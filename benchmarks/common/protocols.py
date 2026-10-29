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

"""Runtime-neutral generation interfaces shared by benchmarks and methods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationRequest:
    """A multimodal observer request independent of any chat template."""

    id: str
    system_prompt: str
    user_prompt: str
    image_paths: tuple[Path, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextRequest:
    """A text-only request used by persona induction and prompt evolution."""

    id: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Predictor(Protocol):
    """Interface implemented by vLLM, Tinker, or hosted inference adapters."""

    def predict(self, requests: Sequence[GenerationRequest]) -> Sequence[str]:
        """Return one raw model response for every request, in input order."""
        ...


@runtime_checkable
class TextGenerator(Protocol):
    """Text-only generation interface used for meta-reasoning stages."""

    def generate_text(self, requests: Sequence[TextRequest]) -> Sequence[str]:
        """Return one raw model response for every request, in input order."""
        ...
