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

"""Adapt OSWorld-Persona records and prompts to reusable method types."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from benchmarks.common import GenerationRequest, Predictor, TextGenerator
from methods import CocoLoRADataBuilder, CocoLoRATrainer, CocoPETrainer, LoRATrainingBackend
from methods import induce_persona as run_induction
from methods.types import (
    InductionRecord,
    MethodExample,
    MethodPrediction,
    RevisedTrainingExample,
)

from .data import sample_frames
from .metrics import parse_complete_decision, parse_decision
from .prompts import (
    COCO_LORA_REFLECTION_BLOCK,
    COCO_PE_CURATOR_SYSTEM,
    COCO_PE_CURATOR_TEMPLATE,
    COCO_PE_REFLECTOR_SYSTEM,
    COCO_PE_REFLECTOR_TEMPLATE,
    INDUCTION_DIGESTION_SYSTEM,
    INDUCTION_DIGESTION_USER,
    INDUCTION_PROFILE_SYSTEM,
    INDUCTION_PROFILE_USER,
    LEARNED_MEMORY_BLOCK,
    observer_prompt,
    render_user_prompt,
)
from .schema import Annotation, PredictionRecord


def to_induction_records(examples: list[Annotation]) -> list[InductionRecord]:
    """Apply the benchmark's leakage boundary before the method sees history."""
    return [
        InductionRecord(
            id=example.id,
            user_id=example.persona,
            label=example.want_support,
            context={"observation": example.observation, "user_intent": example.user_intent},
        )
        for example in examples
    ]


def _compact_induction_text(value: object, max_chars: int) -> str:
    # The canonical schema stores JSON arrays as tuples; turn them back into
    # lists so str(value) matches the raw-list serialization used originally.
    if isinstance(value, tuple):
        value = list(value)
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def render_induction_context(records: list[InductionRecord]) -> str:
    """Serialize Personal Context exactly as the original induced baseline."""
    blocks = []
    for index, record in enumerate(records, 1):
        feedback = "support_wanted" if record.label == "yes" else "support_not_wanted"
        blocks.append(
            f'<history id="h-{index:04d}">\n'
            f"observation: {_compact_induction_text(record.context.get('observation'), 350)}\n"
            f"user_intent: {_compact_induction_text(record.context.get('user_intent'), 200)}\n"
            f"assistance_feedback: {feedback}\n"
            "</history>"
        )
    return "\n\n".join(blocks)


def induce_persona(examples: list[Annotation], generator: TextGenerator, **kwargs):
    """Run generic induction with OSWorld-Persona's leakage policy and prompts."""
    return run_induction(
        to_induction_records(examples),
        generator,
        digestion_system=INDUCTION_DIGESTION_SYSTEM,
        digestion_user=INDUCTION_DIGESTION_USER,
        profile_system=INDUCTION_PROFILE_SYSTEM,
        profile_user=INDUCTION_PROFILE_USER,
        render_context=render_induction_context,
        label_order=("yes", "no"),
        **kwargs,
    )


def to_method_predictions(records: list[PredictionRecord]) -> list[MethodPrediction]:
    return [
        MethodPrediction(record.id, record.raw_prediction, record.predicted) for record in records
    ]


def to_method_examples(
    annotations: list[Annotation],
    base_predictions: list[PredictionRecord] | None = None,
    *,
    max_images: int = 8,
) -> list[MethodExample]:
    prior_memory = (
        {record.id: record.previous_recent_memory for record in base_predictions}
        if base_predictions is not None
        else {}
    )
    expected = {example.id for example in annotations}
    if base_predictions is not None and set(prior_memory) != expected:
        raise ValueError("Base predictions must exactly cover the supplied annotations")
    result = []
    for original in annotations:
        example = sample_frames(original, max_images)
        request = GenerationRequest(
            id=example.id,
            system_prompt=observer_prompt(),
            user_prompt=render_user_prompt(example.timestamps, prior_memory.get(example.id)),
            image_paths=example.image_paths,
            metadata={
                "persona": example.persona,
                "session_id": example.session_id,
                "index": example.index,
            },
        )
        result.append(
            MethodExample(
                id=example.id,
                user_id=example.persona,
                label=example.want_support,
                request=request,
                feedback=example.rationale,
            )
        )
    return result


def parse_label(output: str) -> str | None:
    decision = parse_decision(output)
    return decision.need_support if decision else None


def parse_revised_target(output: str) -> tuple[str, str] | None:
    decision = parse_complete_decision(output)
    if decision is None:
        return None
    target = {key: value for key, value in decision.to_dict().items() if value is not None}
    return decision.need_support, json.dumps(target, ensure_ascii=False)


def inject_prompt_memory(request: GenerationRequest, memory: str) -> GenerationRequest:
    return replace(
        request,
        user_prompt=LEARNED_MEMORY_BLOCK.format(memory=memory) + request.user_prompt,
        metadata={**request.metadata, "stage": "coco_pe_generator"},
    )


def render_lora_reflection(
    example: MethodExample, prediction: MethodPrediction
) -> GenerationRequest:
    reflection = COCO_LORA_REFLECTION_BLOCK.format(
        previous_response=prediction.raw_prediction.replace("<image>", "<image_literal>"),
        ground_truth_label=example.label,
        ground_truth_rationale=example.feedback.replace("<image>", "<image_literal>"),
    )
    return replace(
        example.request,
        user_prompt=f"{example.request.user_prompt}\n\n{reflection}",
        metadata={**example.request.metadata, "stage": "coco_lora_reflection"},
    )


def coco_pe_trainer(predictor: Predictor, text_generator: TextGenerator, **kwargs) -> CocoPETrainer:
    return CocoPETrainer(
        predictor,
        text_generator,
        parse_label=parse_label,
        inject_memory=inject_prompt_memory,
        reflector_system=COCO_PE_REFLECTOR_SYSTEM,
        reflector_template=COCO_PE_REFLECTOR_TEMPLATE,
        curator_system=COCO_PE_CURATOR_SYSTEM,
        curator_template=COCO_PE_CURATOR_TEMPLATE,
        **kwargs,
    )


def coco_lora_builder(predictor: Predictor) -> CocoLoRADataBuilder:
    return CocoLoRADataBuilder(
        predictor,
        parse_target=parse_revised_target,
        render_reflection=render_lora_reflection,
    )


def coco_lora_trainer(
    predictor: Predictor, training_backend: LoRATrainingBackend
) -> CocoLoRATrainer:
    """Build the complete OSWorld-Persona COCO-LoRA training pipeline."""
    return CocoLoRATrainer(coco_lora_builder(predictor), training_backend)


def write_revised_data(
    examples: list[RevisedTrainingExample],
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> None:
    root = Path(relative_to).resolve() if relative_to else None
    rows = []
    for revised in examples:
        request = revised.example.request
        images = []
        for image in request.image_paths:
            if root:
                try:
                    images.append(str(image.relative_to(root)))
                    continue
                except ValueError:
                    pass
            images.append(str(image))
        rows.append(
            {
                "id": revised.example.id,
                "persona": revised.example.user_id,
                "session": request.metadata.get("session_id"),
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                    {"role": "assistant", "content": revised.target},
                ],
                "images": images,
                "metadata": {"correction_source": revised.correction_source},
            }
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_revised_data(
    path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> list[RevisedTrainingExample]:
    """Load the framework-neutral revised dataset written by ``write_revised_data``."""
    source = Path(path).expanduser().resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{source}: expected a JSON list")
    media_root = Path(image_root).expanduser().resolve() if image_root else source.parent
    revised_examples = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        location = f"{source}:{index}"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected an object")
        identifier = row.get("id")
        persona = row.get("persona")
        messages = row.get("messages")
        images = row.get("images", [])
        metadata = row.get("metadata", {})
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"{location}: invalid or duplicate id")
        if not isinstance(persona, str) or not persona:
            raise ValueError(f"{location}: persona must be a non-empty string")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{location}: expected system, user, and assistant messages")
        expected_roles = ("system", "user", "assistant")
        contents = []
        for message, role in zip(messages, expected_roles):
            if not isinstance(message, dict) or message.get("role") != role:
                raise ValueError(f"{location}: expected {role!r} message")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{location}: {role} content must be a string")
            contents.append(content)
        if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
            raise ValueError(f"{location}: images must be a string list")
        if not isinstance(metadata, dict):
            raise ValueError(f"{location}: metadata must be an object")
        parsed_target = parse_revised_target(contents[2])
        if parsed_target is None:
            raise ValueError(f"{location}: assistant target is incomplete or invalid")
        correction_source = metadata.get("correction_source")
        if correction_source not in {"direct", "self_reflection"}:
            raise ValueError(f"{location}: invalid correction_source")
        image_paths = tuple(
            candidate if candidate.is_absolute() else (media_root / candidate).resolve()
            for candidate in (Path(item).expanduser() for item in images)
        )
        example = MethodExample(
            id=identifier,
            user_id=persona,
            label=parsed_target[0],
            request=GenerationRequest(
                id=identifier,
                system_prompt=contents[0],
                user_prompt=contents[1],
                image_paths=image_paths,
                metadata={"persona": persona, "session_id": row.get("session")},
            ),
        )
        revised_examples.append(RevisedTrainingExample(example, contents[2], correction_source))
        seen.add(identifier)
    return revised_examples
