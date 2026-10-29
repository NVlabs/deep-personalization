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

"""OSWorld-Persona benchmark runner."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.common import GenerationRequest, Predictor
from methods import PromptCondition

from .data import sample_frames
from .metrics import EvaluationMetrics, compute_metrics, parse_decision
from .prompts import (
    RECENT_OBSERVATION_LIMIT,
    accumulated_memory,
    apply_condition,
    observer_prompt,
    render_user_prompt,
)
from .schema import Annotation, PredictionRecord


@dataclass
class BenchmarkResult:
    metadata: dict[str, Any]
    metrics: EvaluationMetrics
    predictions: list[PredictionRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "metadata": self.metadata,
            "metrics": self.metrics.to_dict(),
            "predictions": [record.to_dict() for record in self.predictions],
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)


class BenchmarkRunner:
    """Run an observer while respecting dependencies inside each session."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        batch_size: int = 32,
        max_images: int = 8,
        session_context: bool = True,
        recent_observation_limit: int = RECENT_OBSERVATION_LIMIT,
        system_prompt: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_images = max_images
        self.session_context = session_context
        self.recent_observation_limit = recent_observation_limit
        self.system_prompt = system_prompt or observer_prompt()

    def run(
        self,
        examples: Sequence[Annotation],
        *,
        condition: PromptCondition | None = None,
        run_name: str = "",
        metadata: dict[str, Any] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        condition = condition or PromptCondition.vanilla()
        queues: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
        session_order: list[tuple[str, str]] = []
        for example in examples:
            key = (example.persona, example.session_id)
            if key not in queues:
                session_order.append(key)
            queues[key].append(example)
        positions = {key: 0 for key in session_order}
        observations: dict[tuple[str, str], list[str]] = defaultdict(list)
        records: list[PredictionRecord] = []

        while len(records) < len(examples):
            active = [key for key in session_order if positions[key] < len(queues[key])]
            for start in range(0, len(active), self.batch_size):
                keys = active[start : start + self.batch_size]
                pending: list[tuple[Annotation, str | None]] = []
                requests: list[GenerationRequest] = []
                for key in keys:
                    example = sample_frames(queues[key][positions[key]], self.max_images)
                    positions[key] += 1
                    previous = (
                        accumulated_memory(observations[key], self.recent_observation_limit)
                        if self.session_context
                        else None
                    )
                    user = render_user_prompt(example.timestamps, previous)
                    system, user = apply_condition(
                        condition, example.persona, self.system_prompt, user
                    )
                    requests.append(
                        GenerationRequest(
                            id=example.id,
                            system_prompt=system,
                            user_prompt=user,
                            image_paths=example.image_paths,
                            metadata={
                                "persona": example.persona,
                                "split": example.split,
                                "session_id": example.session_id,
                                "index": example.index,
                                "method": condition.name,
                            },
                        )
                    )
                    pending.append((example, previous))

                outputs = list(self.predictor.predict(requests))
                if len(outputs) != len(requests):
                    raise ValueError(
                        f"Predictor returned {len(outputs)} outputs for {len(requests)} requests"
                    )
                for (example, previous), raw in zip(pending, outputs):
                    if not isinstance(raw, str):
                        raise TypeError(f"Predictor output for {example.id} is not a string")
                    decision = parse_decision(raw)
                    key = (example.persona, example.session_id)
                    if self.session_context and decision and decision.observation:
                        observations[key].append(decision.observation)
                    records.append(
                        PredictionRecord(
                            id=example.id,
                            persona=example.persona,
                            split=example.split,
                            index=example.index,
                            session_id=example.session_id,
                            ground_truth=example.want_support,
                            raw_prediction=raw,
                            decision=decision,
                            previous_recent_memory=previous,
                            metadata={"method": condition.name},
                        )
                    )
                if progress:
                    progress(len(records), len(examples))

        source_order = {example.id: position for position, example in enumerate(examples)}
        records.sort(key=lambda record: source_order[record.id])
        prompt_digest = hashlib.sha256(self.system_prompt.encode()).hexdigest()
        result_metadata = {
            "benchmark": "osworld-persona",
            "run_name": run_name,
            "method": condition.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "split": examples[0].split if examples else "",
            "personas": sorted({example.persona for example in examples}),
            "num_examples": len(examples),
            "num_sessions": len(session_order),
            "session_context": self.session_context,
            "recent_observation_limit": self.recent_observation_limit,
            "max_images": self.max_images,
            "observer_prompt_sha256": prompt_digest,
            **(metadata or {}),
        }
        return BenchmarkResult(result_metadata, compute_metrics(records), records)
