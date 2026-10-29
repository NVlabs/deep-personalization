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

"""Canonical OSWorld-Persona inputs and outputs schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

NeedSupport = Literal["yes", "no"]


def normalize_need_support(value: Any) -> NeedSupport | None:
    """Normalize common model spellings of the binary decision."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "1", "support", "need_support"}:
        return "yes"
    if normalized in {"no", "n", "false", "0", "none"}:
        return "no"
    return None


@dataclass(frozen=True)
class Annotation:
    """One OSWorld-Persona decision point."""

    id: str
    persona: str
    split: str
    index: int
    session_id: str
    image_paths: tuple[Path, ...]
    timestamps: tuple[str, ...]
    observation: str
    user_intent: str | tuple[str, ...]
    want_support: NeedSupport
    rationale: str
    required_context_for_support: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_paths"] = [str(path) for path in self.image_paths]
        return value


@dataclass(frozen=True)
class Decision:
    """A parsed observer response."""

    need_support: NeedSupport
    observation: str | None = None
    user_intent: str | None = None
    rationale: str | None = None
    suggestion_type: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionRecord:
    """Prediction joined to its ground truth and execution provenance."""

    id: str
    persona: str
    split: str
    index: int
    session_id: str
    ground_truth: NeedSupport
    raw_prediction: str
    decision: Decision | None
    previous_recent_memory: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def predicted(self) -> NeedSupport | None:
        return self.decision.need_support if self.decision else None

    @property
    def correct(self) -> bool:
        return self.predicted is not None and self.predicted == self.ground_truth

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "persona": self.persona,
            "split": self.split,
            "index": self.index,
            "session_id": self.session_id,
            "ground_truth": self.ground_truth,
            "prediction": self.raw_prediction,
            "pred_need_support": self.predicted,
            "parse_ok": self.decision is not None,
            "correct": self.correct,
            "decision": self.decision.to_dict() if self.decision else None,
            "previous_recent_memory": self.previous_recent_memory,
            "error": self.error,
            "metadata": self.metadata,
        }
