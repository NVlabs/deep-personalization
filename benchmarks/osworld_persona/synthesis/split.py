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

"""Leakage-safe train/test splitting for synthesized session records."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any


def record_session_id(record: dict[str, Any]) -> str:
    """Return the generic session ID, accepting legacy synthesis records."""
    session_id = record.get("session_id", record.get("task_id"))
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(
            f"Synthesis record {record.get('example_id', '<unknown>')!r} has no session_id"
        )
    return session_id.strip()


def label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count synthesized binary labels for split provenance."""
    counts = Counter(
        str(record.get("output", {}).get("need_support", "missing")) for record in records
    )
    return dict(sorted(counts.items()))


def split_records_by_session(
    records: list[dict[str, Any]],
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministically split complete sessions, never individual windows."""
    if not records:
        raise ValueError("Cannot split an empty synthesis dataset")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be strictly between zero and one")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    example_ids: set[str] = set()
    sessions: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every synthesis record must be an object")
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("Every synthesis record must have a non-empty example_id")
        if example_id in example_ids:
            raise ValueError(f"Duplicate synthesis example_id: {example_id!r}")
        example_ids.add(example_id)
        session_id = record_session_id(record)
        sessions.setdefault(session_id, []).append(record)

    if len(sessions) < 2:
        raise ValueError("A train/test split requires at least two source sessions")
    shuffled_sessions = sorted(sessions)
    random.Random(seed).shuffle(shuffled_sessions)
    requested_test_sessions = round(len(shuffled_sessions) * test_ratio)
    num_test_sessions = min(len(shuffled_sessions) - 1, max(1, requested_test_sessions))
    test_session_set = set(shuffled_sessions[:num_test_sessions])
    train_session_set = set(shuffled_sessions) - test_session_set

    train = [record for record in records if record_session_id(record) in train_session_set]
    test = [record for record in records if record_session_id(record) in test_session_set]
    train_sessions = sorted(train_session_set)
    test_sessions = sorted(test_session_set)
    manifest = {
        "format": "osworld-persona-synthesis-split-v1",
        "strategy": "session",
        "seed": seed,
        "requested_test_ratio": test_ratio,
        "actual_test_session_ratio": len(test_sessions) / len(sessions),
        "train": {
            "sessions": train_sessions,
            "num_sessions": len(train_sessions),
            "num_examples": len(train),
            "labels": label_counts(train),
        },
        "test": {
            "sessions": test_sessions,
            "num_sessions": len(test_sessions),
            "num_examples": len(test),
            "labels": label_counts(test),
        },
    }
    return train, test, manifest
