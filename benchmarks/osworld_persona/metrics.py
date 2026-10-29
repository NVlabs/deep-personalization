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

"""Parse model decisions and compute OSWorld-Persona intervention metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from benchmarks.common import parse_json_object

from .schema import Decision, PredictionRecord, normalize_need_support


def parse_decision(value: str | dict[str, Any] | None) -> Decision | None:
    """Parse the minimum valid observer decision used for benchmark scoring."""
    parsed = parse_json_object(value)
    if parsed is None:
        return None
    need_support = normalize_need_support(parsed.get("need_support"))
    if need_support is None:
        return None

    def optional_text(key: str) -> str | None:
        item = parsed.get(key)
        return item.strip() if isinstance(item, str) and item.strip() else None

    return Decision(
        need_support=need_support,
        observation=optional_text("observation"),
        user_intent=optional_text("user_intent"),
        rationale=optional_text("rationale"),
        suggestion_type=optional_text("suggestion_type"),
        suggestion=optional_text("suggestion"),
    )


def parse_complete_decision(value: str | dict[str, Any] | None) -> Decision | None:
    """Require all four fields needed for a revised COCO-LoRA target."""
    decision = parse_decision(value)
    if decision is None:
        return None
    if not decision.observation or not decision.user_intent or not decision.rationale:
        return None
    # COCO-LoRA's revised target is deliberately the four-field benchmark
    # schema, even if a legacy model emitted additional suggestion fields.
    return Decision(
        need_support=decision.need_support,
        observation=decision.observation,
        user_intent=decision.user_intent,
        rationale=decision.rationale,
    )


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


@dataclass(frozen=True)
class BinaryMetrics:
    total: int
    scored: int
    parse_failures: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float | None
    strict_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    false_alarm: float | None
    specificity: float | None
    parse_coverage: float | None
    support_positive: int
    support_negative: int

    def to_dict(self) -> dict[str, int | float | None | dict[str, int]]:
        value = asdict(self)
        for key in ("tp", "fp", "fn", "tn"):
            value.pop(key)
        value["confusion"] = {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}
        return value


@dataclass(frozen=True)
class EvaluationMetrics:
    overall: BinaryMetrics
    by_persona: dict[str, BinaryMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        per_persona = {name: metric.to_dict() for name, metric in self.by_persona.items()}
        macro_keys = ("accuracy", "strict_accuracy", "precision", "recall", "f1", "false_alarm")
        macro = {}
        for key in macro_keys:
            values = [getattr(metric, key) for metric in self.by_persona.values()]
            present = [value for value in values if value is not None]
            macro[key] = _rounded(sum(present) / len(present)) if present else None
        return {
            "overall": self.overall.to_dict(),
            "macro_persona": macro,
            "by_persona": per_persona,
        }


def _binary_metrics(records: list[PredictionRecord]) -> BinaryMetrics:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    parse_failures = 0
    for record in records:
        predicted = record.predicted
        if predicted is None:
            parse_failures += 1
            continue
        if predicted == "yes" and record.ground_truth == "yes":
            counts["tp"] += 1
        elif predicted == "yes":
            counts["fp"] += 1
        elif record.ground_truth == "yes":
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    scored = tp + fp + fn + tn
    total = len(records)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return BinaryMetrics(
        total=total,
        scored=scored,
        parse_failures=parse_failures,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=_rounded((tp + tn) / scored) if scored else None,
        strict_accuracy=_rounded((tp + tn) / total) if total else None,
        precision=_rounded(precision),
        recall=_rounded(recall),
        f1=_rounded(f1),
        false_alarm=_rounded(fp / (fp + tn)) if fp + tn else None,
        specificity=_rounded(tn / (fp + tn)) if fp + tn else None,
        parse_coverage=_rounded(scored / total) if total else None,
        support_positive=sum(record.ground_truth == "yes" for record in records),
        support_negative=sum(record.ground_truth == "no" for record in records),
    )


def compute_metrics(records: list[PredictionRecord]) -> EvaluationMetrics:
    personas = sorted({record.persona for record in records})
    return EvaluationMetrics(
        overall=_binary_metrics(records),
        by_persona={
            persona: _binary_metrics([record for record in records if record.persona == persona])
            for persona in personas
        },
    )
