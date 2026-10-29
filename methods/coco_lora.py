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

"""Benchmark-independent COCO-LoRA data revision and training lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from benchmarks.common import GenerationRequest, Predictor

from .types import MethodExample, MethodPrediction, RevisedTrainingExample

ParseTarget = Callable[[str], Optional[tuple[str, str]]]
RenderReflection = Callable[[MethodExample, MethodPrediction], GenerationRequest]


@dataclass(frozen=True)
class LoRATrainingConfig:
    """Training settings shared by local and hosted LoRA backends."""

    base_model: str
    output_dir: Path
    run_name: str = "coco-lora"
    rank: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 1
    batch_size: int = 4
    max_seq_length: int = 4096
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.base_model.strip():
            raise ValueError("base_model must be non-empty")
        if not self.run_name.strip():
            raise ValueError("run_name must be non-empty")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class TrainedAdapter:
    """Portable provenance for a trained adapter or hosted checkpoint."""

    backend: str
    base_model: str
    adapter_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LoRATrainingBackend(Protocol):
    """Interface implemented by LLaMA-Factory, Tinker, and future trainers."""

    def train(
        self,
        examples: Sequence[RevisedTrainingExample],
        config: LoRATrainingConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> TrainedAdapter:
        """Train one adapter, optionally reporting completed optimizer steps."""
        ...


@dataclass(frozen=True)
class CocoLoRAResult:
    revised_examples: tuple[RevisedTrainingExample, ...]
    revision_audit: tuple[dict[str, Any], ...]
    adapter: TrainedAdapter


class CocoLoRADataBuilder:
    """Keep correct complete outputs and self-reflect all other predictions."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        parse_target: ParseTarget,
        render_reflection: RenderReflection,
    ) -> None:
        self.predictor = predictor
        self.parse_target = parse_target
        self.render_reflection = render_reflection

    def build(
        self,
        examples: list[MethodExample],
        predictions: list[MethodPrediction],
        *,
        batch_size: int = 32,
        fail_on_unresolved: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[RevisedTrainingExample], list[dict[str, Any]]]:
        users = {example.user_id for example in examples}
        if len(users) > 1:
            raise ValueError("COCO-LoRA revises one user's data at a time")
        by_id = {prediction.id: prediction for prediction in predictions}
        if len(by_id) != len(predictions):
            raise ValueError("Duplicate prediction IDs")
        expected = {example.id for example in examples}
        missing, extra = sorted(expected - set(by_id)), sorted(set(by_id) - expected)
        if missing or extra:
            raise ValueError(
                f"Prediction coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )

        accepted: dict[str, RevisedTrainingExample] = {}
        audit: list[dict[str, Any]] = []
        pending: list[tuple[MethodExample, MethodPrediction]] = []
        for example in examples:
            prediction = by_id[example.id]
            parsed = self.parse_target(prediction.raw_prediction)
            if parsed and parsed[0] == example.label:
                accepted[example.id] = RevisedTrainingExample(example, parsed[1], "direct")
                audit.append({"id": example.id, "status": "accepted_direct"})
            else:
                pending.append((example, prediction))

        completed = len(accepted)
        if progress and completed:
            progress(completed, len(examples))

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            requests = [
                self.render_reflection(example, prediction) for example, prediction in batch
            ]
            outputs = list(self.predictor.predict(requests))
            if len(outputs) != len(batch):
                raise ValueError("Predictor returned the wrong number of reflection outputs")
            for (example, _prediction), output in zip(batch, outputs):
                parsed = self.parse_target(output)
                is_accepted = parsed is not None and parsed[0] == example.label
                audit.append(
                    {
                        "id": example.id,
                        "status": "accepted_reflection" if is_accepted else "unresolved",
                        "reflection": output,
                    }
                )
                if is_accepted:
                    accepted[example.id] = RevisedTrainingExample(
                        example, parsed[1], "self_reflection"
                    )
            completed += len(batch)
            if progress:
                progress(completed, len(examples))

        unresolved = [item["id"] for item in audit if item["status"] == "unresolved"]
        if unresolved and fail_on_unresolved:
            raise RuntimeError(f"{len(unresolved)} revisions remain unresolved: {unresolved[:5]}")
        return [accepted[item.id] for item in examples if item.id in accepted], audit


class CocoLoRATrainer:
    """Run COCO-LoRA revision and train the accepted supervision with any backend."""

    def __init__(
        self,
        data_builder: CocoLoRADataBuilder,
        training_backend: LoRATrainingBackend,
    ) -> None:
        self.data_builder = data_builder
        self.training_backend = training_backend

    def train(
        self,
        examples: list[MethodExample],
        predictions: list[MethodPrediction],
        config: LoRATrainingConfig,
        *,
        revision_batch_size: int = 32,
        fail_on_unresolved: bool = True,
        revision_progress: Callable[[int, int], None] | None = None,
        training_progress: Callable[[int, int], None] | None = None,
    ) -> CocoLoRAResult:
        revised, audit = self._revise(
            examples,
            predictions,
            revision_batch_size=revision_batch_size,
            fail_on_unresolved=fail_on_unresolved,
            revision_progress=revision_progress,
        )
        if training_progress is None:
            adapter = self.training_backend.train(revised, config)
        else:
            adapter = self.training_backend.train(
                revised,
                config,
                progress=training_progress,
            )
        return CocoLoRAResult(tuple(revised), tuple(audit), adapter)

    async def train_async(
        self,
        examples: list[MethodExample],
        predictions: list[MethodPrediction],
        config: LoRATrainingConfig,
        *,
        revision_batch_size: int = 32,
        fail_on_unresolved: bool = True,
        revision_progress: Callable[[int, int], None] | None = None,
        training_progress: Callable[[int, int], None] | None = None,
    ) -> CocoLoRAResult:
        """Async entry point for hosted backends such as Tinker."""
        revised, audit = self._revise(
            examples,
            predictions,
            revision_batch_size=revision_batch_size,
            fail_on_unresolved=fail_on_unresolved,
            revision_progress=revision_progress,
        )
        backend_train_async = getattr(self.training_backend, "train_async", None)
        if callable(backend_train_async):
            if training_progress is None:
                adapter = await backend_train_async(revised, config)
            else:
                adapter = await backend_train_async(
                    revised,
                    config,
                    progress=training_progress,
                )
        else:
            kwargs = {"progress": training_progress} if training_progress is not None else {}
            adapter = await asyncio.to_thread(
                self.training_backend.train,
                revised,
                config,
                **kwargs,
            )
        return CocoLoRAResult(tuple(revised), tuple(audit), adapter)

    def _revise(
        self,
        examples: list[MethodExample],
        predictions: list[MethodPrediction],
        *,
        revision_batch_size: int,
        fail_on_unresolved: bool,
        revision_progress: Callable[[int, int], None] | None,
    ) -> tuple[list[RevisedTrainingExample], list[dict[str, Any]]]:
        revised, audit = self.data_builder.build(
            examples,
            predictions,
            batch_size=revision_batch_size,
            fail_on_unresolved=fail_on_unresolved,
            progress=revision_progress,
        )
        if not revised:
            raise RuntimeError("COCO-LoRA produced no accepted training examples")
        return revised, audit
