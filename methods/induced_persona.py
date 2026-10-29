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

"""Benchmark-independent persona induction."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from benchmarks.common import TextGenerator, TextRequest, parse_json_object

from .types import InductionRecord

RenderContext = Callable[[list[InductionRecord]], str]


@dataclass(frozen=True)
class InducedPersona:
    profile: str
    limitations: tuple[str, ...]
    selected_ids: tuple[str, ...]
    digestion: dict[str, Any]


def select_history(
    records: list[InductionRecord],
    max_per_label: int = 60,
    seed: int = 42,
    label_order: tuple[str, ...] | None = None,
) -> list[InductionRecord]:
    """Balance arbitrary feedback labels deterministically, then restore source order."""
    if max_per_label <= 0:
        raise ValueError("max_per_label must be positive")
    rng = random.Random(seed)
    selected = []
    labels = label_order or tuple(sorted({record.label for record in records}))
    for label in labels:
        bucket = [record for record in records if record.label == label]
        if not bucket:
            raise ValueError(f"History contains no records for required label {label!r}")
        selected.extend(
            bucket if len(bucket) <= max_per_label else rng.sample(bucket, max_per_label)
        )
    order = {record.id: index for index, record in enumerate(records)}
    return sorted(selected, key=lambda record: order[record.id])


def render_personal_context(records: list[InductionRecord]) -> str:
    return "\n".join(
        json.dumps(
            {"history_id": f"h-{index:04d}", **record.context, "feedback": record.label},
            ensure_ascii=False,
        )
        for index, record in enumerate(records, 1)
    )


def _one(generator: TextGenerator, request: TextRequest) -> str:
    outputs = list(generator.generate_text([request]))
    if len(outputs) != 1 or not isinstance(outputs[0], str):
        raise ValueError("TextGenerator must return one string per request")
    return outputs[0]


def induce_persona(
    records: list[InductionRecord],
    generator: TextGenerator,
    *,
    digestion_system: str,
    digestion_user: str,
    profile_system: str,
    profile_user: str,
    render_context: RenderContext = render_personal_context,
    max_per_label: int = 60,
    label_order: tuple[str, ...] | None = None,
    digestion_max_tokens: int = 2048,
    profile_max_tokens: int = 1024,
    seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> InducedPersona:
    users = {record.user_id for record in records}
    if len(users) != 1:
        raise ValueError("Persona induction requires history for exactly one user")
    selected = select_history(
        records,
        max_per_label=max_per_label,
        seed=seed,
        label_order=label_order,
    )
    context = render_context(selected)
    digestion_raw = _one(
        generator,
        TextRequest(
            id="induced-persona:digestion",
            system_prompt=digestion_system,
            user_prompt=digestion_user.format(context=context),
            max_tokens=digestion_max_tokens,
        ),
    )
    digestion = parse_json_object(digestion_raw)
    if not digestion or not isinstance(digestion.get("patterns"), list):
        raise ValueError("Persona digestion did not return the required JSON structure")
    if progress:
        progress(1, 2)
    profile_raw = _one(
        generator,
        TextRequest(
            id="induced-persona:profile",
            system_prompt=profile_system,
            user_prompt=profile_user.format(
                context=context,
                guidance=json.dumps(digestion, ensure_ascii=False, indent=2),
            ),
            max_tokens=profile_max_tokens,
        ),
    )
    profile_obj = parse_json_object(profile_raw)
    profile = profile_obj.get("profile") if profile_obj else None
    limitations = profile_obj.get("limitations", []) if profile_obj else []
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("Profile generation did not return a non-empty profile")
    if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
        raise ValueError("Profile limitations must be a string list")
    if progress:
        progress(2, 2)
    return InducedPersona(
        profile.strip(), tuple(limitations), tuple(record.id for record in selected), digestion
    )
