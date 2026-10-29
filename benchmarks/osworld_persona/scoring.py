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

"""Offline scoring for OSWorld-Persona predictions produced by any runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import compute_metrics, parse_decision
from .run import BenchmarkResult
from .schema import Annotation, PredictionRecord


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load an offline prediction file and report malformed rows precisely."""
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def score_external_predictions(
    annotations: list[Annotation],
    rows: list[dict[str, Any]],
    *,
    method: str = "external",
    allow_missing: bool = False,
) -> BenchmarkResult:
    """Align offline predictions with annotations and apply benchmark metrics."""
    by_id = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every prediction row needs a non-empty string id")
        if identifier in by_id:
            raise ValueError(f"Duplicate prediction ID: {identifier}")
        by_id[identifier] = row
    annotation_ids = {annotation.id for annotation in annotations}
    extra = sorted(set(by_id) - annotation_ids)
    missing = [annotation.id for annotation in annotations if annotation.id not in by_id]
    if extra:
        raise ValueError(f"Predictions contain unknown IDs: {extra[:5]}")
    if missing and not allow_missing:
        raise ValueError(f"Predictions are missing {len(missing)} IDs: {missing[:5]}")

    records = []
    for annotation in annotations:
        row = by_id.get(annotation.id, {})
        prediction = row.get("prediction", row.get("output", ""))
        raw = (
            prediction
            if isinstance(prediction, str)
            else json.dumps(prediction, ensure_ascii=False)
        )
        known = {"id", "prediction", "output", "previous_recent_memory", "error", "metadata"}
        provenance = {key: value for key, value in row.items() if key not in known}
        if isinstance(row.get("metadata"), dict):
            provenance.update(row["metadata"])
        records.append(
            PredictionRecord(
                id=annotation.id,
                persona=annotation.persona,
                split=annotation.split,
                index=annotation.index,
                session_id=annotation.session_id,
                ground_truth=annotation.want_support,
                raw_prediction=raw,
                decision=parse_decision(prediction),
                previous_recent_memory=row.get("previous_recent_memory"),
                error=row.get("error") if isinstance(row.get("error"), str) else None,
                metadata=provenance,
            )
        )
    return BenchmarkResult(
        metadata={
            "benchmark": "osworld-persona",
            "method": method,
            "split": annotations[0].split if annotations else "",
            "personas": sorted({annotation.persona for annotation in annotations}),
            "num_examples": len(annotations),
            "prediction_source": "external_jsonl",
        },
        metrics=compute_metrics(records),
        predictions=records,
    )
