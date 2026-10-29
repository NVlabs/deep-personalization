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

"""Chronology-aware construction of synthesized OSWorld-Persona rows."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frames import Frame
from .pipeline import Perception, SynthesisPipeline
from .profiles import (
    TOOL_PROFILES,
    USER_PROFILES,
    ToolProfile,
    UserProfile,
    sample_profiles,
)


@dataclass(frozen=True)
class SessionSeed:
    """A raw session after video/trajectory frames have been selected."""

    session_id: str
    frames: tuple[Frame, ...]
    instruction: str = ""
    source_kind: str = "trajectory"
    source_path: Path | None = None


@dataclass(frozen=True)
class BuildConfig:
    window_size: int = 5
    window_stride: int = 5
    min_window: int = 2
    max_windows: int = 8
    recent_observations: int = 2
    seed: int = 0
    use_instruction_hint: bool = True
    use_initial_observation: bool = True
    user_profile_ids: tuple[str, ...] = ()
    tool_profile_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.window_stride <= 0:
            raise ValueError("window_size and window_stride must be positive")
        if self.min_window <= 0 or self.min_window > self.window_size:
            raise ValueError("min_window must be between 1 and window_size")
        if self.max_windows < 0 or self.recent_observations < 0:
            raise ValueError("max_windows and recent_observations cannot be negative")
        _resolve_profiles(self.user_profile_ids, USER_PROFILES, "user")
        _resolve_profiles(self.tool_profile_ids, TOOL_PROFILES, "tool")


def _resolve_profiles(
    ids: tuple[str, ...], profiles: tuple[Any, ...], kind: str
) -> tuple[Any, ...]:
    if not ids:
        return ()
    by_id = {profile.id: profile for profile in profiles}
    unknown = [profile_id for profile_id in ids if profile_id not in by_id]
    if unknown:
        raise ValueError(
            f"Unknown {kind} profile(s): {', '.join(unknown)}; valid: {', '.join(by_id)}"
        )
    return tuple(by_id[profile_id] for profile_id in ids)


def frame_windows(
    frames: Iterable[Frame],
    *,
    size: int = 5,
    stride: int = 5,
    minimum: int = 2,
    limit: int = 8,
) -> list[tuple[int, list[Frame]]]:
    """Create indexed chronological windows from one key-frame stream."""
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be positive")
    if minimum <= 0 or minimum > size:
        raise ValueError("minimum must be between 1 and size")
    ordered = list(frames)
    windows: list[tuple[int, list[Frame]]] = []
    offset = 0
    while offset < len(ordered):
        window = ordered[offset : offset + size]
        if len(window) >= minimum:
            windows.append((len(windows), window))
        offset += stride
        if limit > 0 and len(windows) >= limit:
            break
    return windows


def format_recent(observations: list[str], count: int) -> str:
    if count <= 0:
        return ""
    return "\n".join(f"- {observation}" for observation in observations[-count:])


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read valid JSONL rows, tolerating a half-written final line after a crash."""
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def balance_records(rows: list[dict[str, Any]], *, seed: int = 0) -> list[dict[str, Any]]:
    """Downsample the majority binary class without modifying the complete dataset."""
    yes = [row for row in rows if row.get("output", {}).get("need_support") == "yes"]
    no = [row for row in rows if row.get("output", {}).get("need_support") == "no"]
    keep = min(len(yes), len(no))
    if keep == 0:
        return list(rows)
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)
    return sorted(yes[:keep] + no[:keep], key=lambda row: str(row.get("example_id", "")))


class DatasetSynthesizer:
    """Build one session sequentially so recent observations never leak forward."""

    def __init__(
        self,
        pipeline: SynthesisPipeline,
        *,
        config: BuildConfig | None = None,
        model_name: str = "unknown",
    ) -> None:
        self.pipeline = pipeline
        self.config = config or BuildConfig()
        self.model_name = model_name

    def _profile_combinations(
        self, base_id: str
    ) -> tuple[list[tuple[UserProfile, ToolProfile]], bool]:
        users = _resolve_profiles(self.config.user_profile_ids, USER_PROFILES, "user")
        tools = _resolve_profiles(self.config.tool_profile_ids, TOOL_PROFILES, "tool")
        explicit = bool(users or tools)
        sampled_user, sampled_tool = sample_profiles(self.config.seed, base_id)
        selected_users = users or (sampled_user,)
        selected_tools = tools or (sampled_tool,)
        return [(user, tool) for user in selected_users for tool in selected_tools], explicit

    @staticmethod
    def _example_id(base_id: str, user: UserProfile, tool: ToolProfile, explicit: bool) -> str:
        return f"{base_id}::{user.id}__{tool.id}" if explicit else base_id

    def build_session(
        self,
        seed: SessionSeed,
        *,
        existing_by_id: dict[str, dict[str, Any]] | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Synthesize pending rows, reusing stored perceptions during partial resume."""
        prior = existing_by_id or {}
        created: list[dict[str, Any]] = []
        windows = frame_windows(
            seed.frames,
            size=self.config.window_size,
            stride=self.config.window_stride,
            minimum=self.config.min_window,
            limit=self.config.max_windows,
        )
        pending_exists = False
        for index, _ in windows:
            base_id = f"{seed.session_id}::win{index:02d}"
            combinations, explicit = self._profile_combinations(base_id)
            if any(
                self._example_id(base_id, user, tool, explicit) not in prior
                for user, tool in combinations
            ):
                pending_exists = True
                break
        if not pending_exists:
            return created

        recent: list[str] = []
        if (
            seed.instruction
            and self.config.use_instruction_hint
            and self.config.use_initial_observation
        ):
            initial = self.pipeline.initial_observation(
                seed.instruction, request_id=seed.session_id
            )
            if initial:
                recent.append(initial)

        for index, frames in windows:
            base_id = f"{seed.session_id}::win{index:02d}"
            combinations, explicit = self._profile_combinations(base_id)
            recent_text = format_recent(recent, self.config.recent_observations)
            pending = [
                (user, tool)
                for user, tool in combinations
                if self._example_id(base_id, user, tool, explicit) not in prior
            ]
            previous_rows = [
                prior[self._example_id(base_id, user, tool, explicit)]
                for user, tool in combinations
                if self._example_id(base_id, user, tool, explicit) in prior
            ]
            if previous_rows:
                output = previous_rows[0].get("output", {})
                perception = Perception(
                    observation=str(output.get("observation", "")).strip(),
                    user_intent=output.get("user_intent", ""),
                )
            elif pending:
                perception = self.pipeline.perceive_window(
                    frames,
                    request_id=base_id,
                    instruction_hint=(
                        seed.instruction if self.config.use_instruction_hint else None
                    ),
                    recent_observation=recent_text or None,
                )
                if perception is None:
                    continue
            else:
                continue
            recent.append(perception.observation)

            for user, tool in pending:
                example_id = self._example_id(base_id, user, tool, explicit)
                result = self.pipeline.label_window(
                    perception,
                    user,
                    tool,
                    request_id=example_id,
                    recent_observation=recent_text or None,
                )
                if result is None:
                    continue
                source_key = "source_video" if seed.source_kind == "video" else "source_trajectory"
                row: dict[str, Any] = {
                    "example_id": example_id,
                    "session_id": seed.session_id,
                    source_key: str(seed.source_path) if seed.source_path else "",
                    "instruction": seed.instruction,
                    "window": {
                        "index": index,
                        "t_start": frames[0].t_seconds,
                        "t_end": frames[-1].t_seconds,
                        "frames": [
                            {
                                "path": str(frame.path),
                                "t_seconds": frame.t_seconds,
                                "label": frame.label,
                            }
                            for frame in frames
                        ],
                    },
                    "input": {
                        "user_profile": user.text,
                        "tool_profile": tool.text,
                        "recent_observation": recent_text,
                        "image_paths": [str(frame.path) for frame in frames],
                    },
                    "input_profiles": {
                        "user_profile_id": user.id,
                        "tool_profile_id": tool.id,
                    },
                    "output": result.to_output(),
                    "model": self.model_name,
                    "validation": {"status": "pending", "notes": ""},
                }
                created.append(row)
                prior[example_id] = row
                if emit:
                    emit(row)
        return created
