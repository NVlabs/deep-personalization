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

"""Shared benchmark-neutral data models for personalization methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.common import GenerationRequest


@dataclass(frozen=True)
class MethodExample:
    """A labeled request expressed in a method-neutral form."""

    id: str
    user_id: str
    label: str
    request: GenerationRequest
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodPrediction:
    """The minimum prediction provenance needed by learning methods."""

    id: str
    raw_prediction: str
    predicted_label: str | None


@dataclass(frozen=True)
class RevisedTrainingExample:
    """A framework-neutral revised supervision example."""

    example: MethodExample
    target: str
    correction_source: str


@dataclass(frozen=True)
class InductionRecord:
    """Leakage-reviewed historical context for profile induction."""

    id: str
    user_id: str
    label: str
    context: dict[str, Any]
