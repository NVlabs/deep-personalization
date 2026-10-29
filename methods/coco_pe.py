# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-independent COCO-PE failure-driven prompt evolution."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Optional

from benchmarks.common import (
    GenerationRequest,
    Predictor,
    TextGenerator,
    TextRequest,
    parse_json_object,
)

from .types import MethodExample

SECTIONS = ("when_to_support", "when_to_stay_silent", "how_to_support", "general")
SECTION_TITLES = {
    "when_to_support": "When to proactively support",
    "when_to_stay_silent": "When to stay silent",
    "how_to_support": "How to support (style & routing)",
    "general": "General notes about this user",
}

ParseLabel = Callable[[str], Optional[str]]
InjectMemory = Callable[[GenerationRequest, str], GenerationRequest]


@dataclass
class MemoryBullet:
    section: str
    content: str
    helpful: int
    harmful: int
    born: int


class PromptMemory:
    """Sectioned prompt memory updated with incremental delta operations."""

    def __init__(self) -> None:
        self.bullets: dict[str, MemoryBullet] = {}
        self.next_id = 1
        self.age = 0

    def render(self, *, with_ids: bool = True) -> str:
        if not self.bullets:
            return "(no entries yet — you have not learned anything about this user so far)"
        lines: list[str] = []
        for section in SECTIONS:
            items = [
                (key, value) for key, value in self.bullets.items() if value.section == section
            ]
            if not items:
                continue
            lines.append(f"## {SECTION_TITLES[section]}")
            for key, bullet in sorted(items, key=lambda item: item[1].born):
                identifier = f"[{key}] " if with_ids else ""
                lines.append(f"- {identifier}{bullet.content}")
            lines.append("")
        return "\n".join(lines).strip()

    def vote(self, helpful: list[str], harmful: list[str]) -> None:
        for identifier in helpful:
            if identifier in self.bullets:
                self.bullets[identifier].helpful += 1
        for identifier in harmful:
            if identifier in self.bullets:
                self.bullets[identifier].harmful += 1

    def apply(self, operations: list[dict[str, Any]], max_operations: int) -> int:
        applied = 0
        for operation in operations[:max_operations]:
            kind = operation.get("op")
            content = operation.get("content")
            content = content.strip() if isinstance(content, str) else ""
            if kind == "add" and content:
                normalized = re.sub(r"\W+", " ", content.lower()).strip()
                if any(
                    re.sub(r"\W+", " ", bullet.content.lower()).strip() == normalized
                    for bullet in self.bullets.values()
                ):
                    continue
                section = operation.get("section")
                section = section if section in SECTIONS else "general"
                identifier = f"m-{self.next_id:03d}"
                self.next_id += 1
                self.age += 1
                self.bullets[identifier] = MemoryBullet(section, content, 0, 0, self.age)
                applied += 1
            elif kind == "update" and operation.get("id") in self.bullets and content:
                self.bullets[operation["id"]].content = content
                applied += 1
            elif kind == "delete" and operation.get("id") in self.bullets:
                del self.bullets[operation["id"]]
                applied += 1
        return applied

    def refine(self, max_bullets: int) -> int:
        count = max(0, len(self.bullets) - max_bullets)
        ranked = sorted(
            self.bullets.items(),
            key=lambda item: (
                item[1].helpful - item[1].harmful,
                item[1].helpful,
                -item[1].born,
            ),
        )
        for identifier, _ in ranked[:count]:
            del self.bullets[identifier]
        return count

    def to_dict(self) -> dict[str, Any]:
        return {
            "next": self.next_id,
            "age": self.age,
            "bullets": {key: asdict(value) for key, value in self.bullets.items()},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PromptMemory:
        memory = cls()
        # Accept the short-lived release-candidate key as well as the original
        # implementation's state format so existing checkpoints remain usable.
        memory.next_id = int(value.get("next", value.get("next_id")))
        memory.age = int(value["age"])
        memory.bullets = {
            key: MemoryBullet(**bullet) for key, bullet in value.get("bullets", {}).items()
        }
        return memory


def _one(generator: TextGenerator, request: TextRequest) -> str:
    outputs = list(generator.generate_text([request]))
    if len(outputs) != 1 or not isinstance(outputs[0], str):
        raise ValueError("TextGenerator must return one string per request")
    return outputs[0]


class CocoPETrainer:
    """Evolve one user's prompt memory using benchmark-supplied adapters."""

    def __init__(
        self,
        predictor: Predictor,
        text_generator: TextGenerator,
        *,
        parse_label: ParseLabel,
        inject_memory: InjectMemory,
        reflector_system: str,
        reflector_template: str,
        curator_system: str,
        curator_template: str,
        batch_size: int = 8,
        max_bullets: int = 50,
        max_operations: int = 8,
        reflect_correct: int = 2,
        reflection_max_tokens: int = 8192,
        curator_max_tokens: int = 4096,
        seed: int = 42,
    ) -> None:
        self.predictor = predictor
        self.text_generator = text_generator
        self.parse_label = parse_label
        self.inject_memory = inject_memory
        self.reflector_system = reflector_system
        self.reflector_template = reflector_template
        self.curator_system = curator_system
        self.curator_template = curator_template
        self.batch_size = batch_size
        self.max_bullets = max_bullets
        self.max_operations = max_operations
        self.reflect_correct = reflect_correct
        self.reflection_max_tokens = reflection_max_tokens
        self.curator_max_tokens = curator_max_tokens
        self.seed = seed

    def fit(
        self,
        examples: list[MethodExample],
        *,
        epochs: int = 1,
        shuffle: bool = True,
        memory: PromptMemory | None = None,
        progress: Callable[[int, int], None] | None = None,
        checkpoint: Callable[[PromptMemory, dict[str, Any]], None] | None = None,
    ) -> tuple[PromptMemory, list[dict[str, Any]]]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        users = {example.user_id for example in examples}
        if len(users) > 1:
            raise ValueError("COCO-PE learns one user memory at a time")
        memory = memory or PromptMemory()
        rng = random.Random(self.seed)
        audit: list[dict[str, Any]] = []
        completed = 0
        total = len(examples) * epochs
        for epoch in range(1, epochs + 1):
            order = list(range(len(examples)))
            if shuffle:
                rng.shuffle(order)
            epoch_correct = 0
            epoch_seen = 0
            for start in range(0, len(order), self.batch_size):
                batch = [examples[index] for index in order[start : start + self.batch_size]]
                memory_text = memory.render(with_ids=True)
                requests = [self.inject_memory(example.request, memory_text) for example in batch]
                outputs = list(self.predictor.predict(requests))
                if len(outputs) != len(batch):
                    raise ValueError("Predictor returned the wrong number of COCO-PE outputs")
                labels = [self.parse_label(output) for output in outputs]
                correct = [
                    label is not None and label == example.label
                    for label, example in zip(labels, batch)
                ]
                epoch_seen += len(batch)
                epoch_correct += sum(correct)
                wrong = [index for index, value in enumerate(correct) if not value]
                right = [index for index, value in enumerate(correct) if value][
                    : self.reflect_correct
                ]
                targets = sorted(wrong + right)
                reflections = self._reflect(batch, outputs, correct, targets, memory_text)
                for reflection in reflections:
                    memory.vote(
                        reflection.get("helpful_bullet_ids", []),
                        reflection.get("harmful_bullet_ids", []),
                    )
                operations = self._curate(reflections, memory) if reflections else []
                applied = memory.apply(operations, self.max_operations)
                dropped = memory.refine(self.max_bullets)
                record = {
                    "epoch": epoch,
                    "batch": start // self.batch_size + 1,
                    "batch_acc": round(sum(correct) / len(correct), 4),
                    "running_acc": round(epoch_correct / epoch_seen, 4),
                    "n_wrong": len(wrong),
                    "n_reflection_targets": len(targets),
                    "n_reflections": len(reflections),
                    "ops_applied": applied,
                    "bullets_dropped": dropped,
                    "n_bullets": len(memory.bullets),
                }
                audit.append(record)
                if checkpoint:
                    checkpoint(memory, record)
                completed += len(batch)
                if progress:
                    progress(completed, total)
        return memory, audit

    def _reflect(self, examples, outputs, correct, targets, memory_text):
        if not targets:
            return []
        blocks = []
        for index, (example, output, is_correct) in enumerate(zip(examples, outputs, correct)):
            mode = "REFLECT" if index in targets else "CONTEXT_ONLY"
            parsed_prediction = parse_json_object(output)
            prediction = (
                json.dumps(parsed_prediction, ensure_ascii=False, indent=1)
                if parsed_prediction
                else (output[:800] or "(no output)")
            )
            if len(prediction) > 4000:
                prediction = prediction[:4000] + "\n... <prediction truncated>"
            blocks.append(
                f'<example sequence_index="{index}" mode="{mode}">\n'
                f"example_id: {example.id}\n"
                f"assistant_prediction:\n{prediction}\n"
                f"ground_truth_need_support: {example.label}\n"
                f"decision_verdict: {'CORRECT' if is_correct else 'WRONG'}\n"
                "</example>"
            )
        prompt = self.reflector_template.format(
            memory=memory_text,
            examples="\n\n".join(blocks),
        )
        parsed = parse_json_object(
            _one(
                self.text_generator,
                TextRequest(
                    id="coco-pe:reflect",
                    system_prompt=self.reflector_system,
                    user_prompt=prompt,
                    temperature=0.2,
                    max_tokens=self.reflection_max_tokens,
                ),
            )
        )
        raw_reflections = parsed.get("reflections") if parsed else None
        if not isinstance(raw_reflections, list):
            return []
        seen: set[int] = set()
        reflections = []
        for item in raw_reflections:
            index = item.get("sequence_index") if isinstance(item, dict) else None
            if isinstance(index, int) and index in targets and index not in seen:
                seen.add(index)
                item["verdict"] = "correct" if correct[index] else "wrong"
                reflections.append(item)
        return reflections

    def _curate(self, reflections, memory):
        reflection_lines = []
        for index, reflection in enumerate(reflections, 1):
            reflection_lines.append(
                f"Reflection {index} (prediction was {reflection['verdict']}): "
                f"{reflection.get('reflection', '')}\n"
                "  proposed insights: "
                f"{json.dumps(reflection.get('proposed_insights', []))}"
            )
        prompt = self.curator_template.format(
            memory=memory.render(with_ids=True),
            reflections="\n".join(reflection_lines),
        )
        parsed = parse_json_object(
            _one(
                self.text_generator,
                TextRequest(
                    id="coco-pe:curate",
                    system_prompt=self.curator_system.replace(
                        "{max_ops}", str(self.max_operations)
                    ),
                    user_prompt=prompt,
                    temperature=0.2,
                    max_tokens=self.curator_max_tokens,
                ),
            )
        )
        operations = parsed.get("ops") if parsed else None
        return operations if isinstance(operations, list) else []
