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

"""Self-consistency checks for synthesized support decisions."""

from __future__ import annotations

from typing import Any

from .pipeline import Perception, SynthesisPipeline
from .profiles import ToolProfile, UserProfile


def consistency_vote(
    row: dict[str, Any],
    pipeline: SynthesisPipeline,
    *,
    samples: int = 3,
    min_agree: int | None = None,
    temperature: float = 0.4,
) -> dict[str, Any]:
    """Re-run only the personalized label stage and compare it with the original."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    threshold = samples if min_agree is None else min_agree
    if threshold <= 0 or threshold > samples:
        raise ValueError("min_agree must be between 1 and samples")
    output = row["output"]
    inputs = row["input"]
    profiles = row["input_profiles"]
    perception = Perception(
        observation=str(output.get("observation", "")),
        user_intent=output.get("user_intent", ""),
    )
    user = UserProfile(str(profiles.get("user_profile_id", "?")), inputs["user_profile"])
    tool = ToolProfile(str(profiles.get("tool_profile_id", "?")), inputs["tool_profile"])
    original = str(output.get("need_support", "")).strip().lower()
    votes: list[str] = []
    for sample in range(samples):
        result = pipeline.label_window(
            perception,
            user,
            tool,
            request_id=f"{row['example_id']}:consistency:{sample}",
            recent_observation=inputs.get("recent_observation") or None,
            temperature=temperature,
        )
        votes.append(result.need_support if result else "error")
    agree = sum(vote == original for vote in votes)
    return {
        "example_id": row["example_id"],
        "original": original,
        "votes": votes,
        "agree": agree,
        "samples": samples,
        "min_agree": threshold,
        "kept": agree >= threshold,
    }


def attach_consistency(
    rows: list[dict[str, Any]], votes_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the kept rows with their vote audit attached."""
    kept: list[dict[str, Any]] = []
    for row in rows:
        vote = votes_by_id.get(str(row.get("example_id")))
        if not vote or not vote.get("kept"):
            continue
        annotated = dict(row)
        annotated["self_consistency"] = {
            key: vote[key] for key in ("votes", "agree", "samples", "min_agree")
        }
        kept.append(annotated)
    return kept
