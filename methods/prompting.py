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

"""Prompt-condition descriptors shared across benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

MethodName = Literal["vanilla", "induced_persona", "oracle_persona", "coco_pe", "coco_lora"]


@dataclass(frozen=True)
class PromptCondition:
    """Personalization context; each benchmark decides how to render it."""

    name: MethodName
    profiles: Mapping[str, str] | None = None
    memories: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.name in {"induced_persona", "oracle_persona"} and not self.profiles:
            raise ValueError(f"{self.name} requires a user-to-profile mapping")
        if self.name == "coco_pe" and not self.memories:
            raise ValueError("coco_pe requires a user-to-memory mapping")
        if self.name in {"vanilla", "coco_lora"} and (self.profiles or self.memories):
            raise ValueError(f"{self.name} does not accept prompt-side personalization")

    @classmethod
    def vanilla(cls) -> PromptCondition:
        return cls("vanilla")

    @classmethod
    def induced_persona(cls, profiles: Mapping[str, str]) -> PromptCondition:
        return cls("induced_persona", profiles=dict(profiles))

    @classmethod
    def oracle_persona(cls, profiles: Mapping[str, str]) -> PromptCondition:
        return cls("oracle_persona", profiles=dict(profiles))

    @classmethod
    def coco_pe(cls, memories: Mapping[str, str]) -> PromptCondition:
        return cls("coco_pe", memories=dict(memories))

    @classmethod
    def coco_lora(cls) -> PromptCondition:
        return cls("coco_lora")

    def profile_for(self, user_id: str) -> str:
        return self._value_for(self.profiles, user_id, "profile")

    def memory_for(self, user_id: str) -> str:
        return self._value_for(self.memories, user_id, "memory")

    @staticmethod
    def _value_for(values: Mapping[str, str] | None, user_id: str, kind: str) -> str:
        if values is None or user_id not in values:
            raise ValueError(f"No {kind} configured for user {user_id!r}")
        value = values[user_id]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{kind.title()} for {user_id!r} must be non-empty")
        return value.strip()
