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

"""Two-stage LLM-powered annotation pipeline."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, TypeVar

from benchmarks.common import (
    GenerationRequest,
    Predictor,
    TextGenerator,
    TextRequest,
    parse_json_object,
)

from . import prompts
from .frames import Frame
from .profiles import ToolProfile, UserProfile

T = TypeVar("T")


@dataclass(frozen=True)
class Perception:
    observation: str
    user_intent: object


@dataclass(frozen=True)
class SynthesisResult:
    observation: str
    user_intent: object
    need_support: str
    rationale: str
    suggestion_type: str
    suggestion: str

    def to_output(self) -> dict[str, object]:
        return {
            "observation": self.observation,
            "user_intent": self.user_intent,
            "need_support": self.need_support,
            "rationale": self.rationale,
            "suggestion_type": self.suggestion_type,
            "suggestion": self.suggestion,
        }


def _recent_block(recent_observation: str | None) -> str:
    return (
        prompts.RECENT_OBSERVATION_TEMPLATE.format(recent_observation=recent_observation)
        if recent_observation
        else ""
    )


def _first(outputs: object, request_id: str) -> str:
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence) or len(outputs) != 1:
        raise RuntimeError(f"Backend must return exactly one response for {request_id}")
    if not isinstance(outputs[0], str):
        raise RuntimeError(f"Backend returned non-text output for {request_id}")
    return outputs[0]


def _retry(call: Callable[[], T | None], attempts: int, base_delay: float) -> T | None:
    result = None
    for attempt in range(attempts):
        try:
            result = call()
        except Exception as exc:
            result = None
            print(
                f"Synthesis call failed ({attempt + 1}/{attempts}): "
                f"{type(exc).__name__}: {str(exc)[:200]}",
                file=sys.stderr,
            )
        if result is not None:
            return result
        if attempt + 1 < attempts and base_delay > 0:
            time.sleep(min(base_delay * (2**attempt), 30.0))
    return result


class SynthesisPipeline:
    """Generate grounded perceptions and profile-conditioned support labels.

    The interfaces are the same small contracts used by benchmark methods, so
    perception and labeling may be served by LLaMA-Factory, Tinker, or another
    backend without importing that framework here.
    """

    def __init__(
        self,
        perceiver: Predictor,
        labeler: TextGenerator,
        *,
        attempts: int = 4,
        retry_delay: float = 2.0,
        perceive_temperature: float = 0.4,
        label_temperature: float = 0.4,
        max_tokens: int = 10240,
    ) -> None:
        if attempts <= 0:
            raise ValueError("attempts must be positive")
        self.perceiver = perceiver
        self.labeler = labeler
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.perceive_temperature = perceive_temperature
        self.label_temperature = label_temperature
        self.max_tokens = max_tokens

    def initial_observation(self, instruction: str, *, request_id: str) -> str | None:
        if not instruction.strip():
            return None

        def call() -> str | None:
            response = (
                _first(
                    self.labeler.generate_text(
                        [
                            TextRequest(
                                id=f"{request_id}:initial",
                                system_prompt=prompts.INITIAL_OBSERVATION_SYSTEM,
                                user_prompt=prompts.INITIAL_OBSERVATION_USER.format(
                                    instruction=instruction
                                ),
                                temperature=0.5,
                                max_tokens=self.max_tokens,
                                metadata={"synthesis_stage": "initial_observation"},
                            )
                        ]
                    ),
                    request_id,
                )
                .strip()
                .strip("`")
                .strip()
            )
            return response or None

        return _retry(call, self.attempts, self.retry_delay)

    def perceive_window(
        self,
        frames: list[Frame],
        *,
        request_id: str,
        instruction_hint: str | None = None,
        recent_observation: str | None = None,
    ) -> Perception | None:
        if not frames:
            raise ValueError("A perception window must contain at least one frame")
        count = len(frames)
        screenshot_lines = "\n".join(
            f"  [t={frame.label}] Screenshot {index + 1} of {count}"
            for index, frame in enumerate(frames)
        )
        hint = (
            prompts.INSTRUCTION_HINT_TEMPLATE.format(instruction=instruction_hint)
            if instruction_hint
            else ""
        )
        user_prompt = prompts.PERCEIVE_USER_TEMPLATE.format(
            recent=_recent_block(recent_observation),
            screenshot_lines=screenshot_lines,
            hint=hint,
            image_markers="\n" + "\n".join("<image>" for _ in frames),
        )

        def call() -> Perception | None:
            raw = _first(
                self.perceiver.predict(
                    [
                        GenerationRequest(
                            id=f"{request_id}:perceive",
                            system_prompt=prompts.PERCEIVE_SYSTEM,
                            user_prompt=user_prompt,
                            image_paths=tuple(frame.path for frame in frames),
                            metadata={
                                "synthesis_stage": "perceive",
                                "temperature": self.perceive_temperature,
                                "max_tokens": self.max_tokens,
                            },
                        )
                    ]
                ),
                request_id,
            )
            parsed = parse_json_object(raw)
            if not parsed or not str(parsed.get("observation", "")).strip():
                return None
            return Perception(
                observation=str(parsed["observation"]).strip(),
                user_intent=parsed.get("user_intent", ""),
            )

        return _retry(call, self.attempts, self.retry_delay)

    def label_window(
        self,
        perception: Perception,
        user_profile: UserProfile,
        tool_profile: ToolProfile,
        *,
        request_id: str,
        recent_observation: str | None = None,
        temperature: float | None = None,
    ) -> SynthesisResult | None:
        intent = (
            json.dumps(perception.user_intent)
            if isinstance(perception.user_intent, list)
            else str(perception.user_intent)
        )
        user_prompt = prompts.LABEL_USER_TEMPLATE.format(
            user_profile=user_profile.text,
            tool_profile=tool_profile.text,
            recent=_recent_block(recent_observation),
            observation=perception.observation,
            user_intent=intent,
        )

        def call() -> SynthesisResult | None:
            raw = _first(
                self.labeler.generate_text(
                    [
                        TextRequest(
                            id=f"{request_id}:label",
                            system_prompt=prompts.LABEL_SYSTEM,
                            user_prompt=user_prompt,
                            temperature=(
                                self.label_temperature if temperature is None else temperature
                            ),
                            max_tokens=self.max_tokens,
                            metadata={"synthesis_stage": "label"},
                        )
                    ]
                ),
                request_id,
            )
            parsed = parse_json_object(raw)
            if not parsed or "need_support" not in parsed:
                return None
            need = str(parsed.get("need_support", "")).strip().lower()
            need = "yes" if need in {"yes", "true", "y"} else "no"
            suggestion = str(parsed.get("suggestion", "") or "")
            suggestion_type = str(parsed.get("suggestion_type", "none")).strip()
            if need == "no":
                suggestion_type, suggestion = "none", ""
            elif suggestion_type not in {"none", "direct_message", "prompt_to_agent"}:
                suggestion_type = "direct_message" if suggestion else "none"
            return SynthesisResult(
                observation=perception.observation,
                user_intent=perception.user_intent,
                need_support=need,
                rationale=str(parsed.get("rationale", "")).strip(),
                suggestion_type=suggestion_type,
                suggestion=suggestion,
            )

        return _retry(call, self.attempts, self.retry_delay)

    def synthesize(
        self,
        frames: list[Frame],
        user_profile: UserProfile,
        tool_profile: ToolProfile,
        *,
        request_id: str,
        instruction_hint: str | None = None,
        recent_observation: str | None = None,
    ) -> SynthesisResult | None:
        perception = self.perceive_window(
            frames,
            request_id=request_id,
            instruction_hint=instruction_hint,
            recent_observation=recent_observation,
        )
        if perception is None:
            return None
        return self.label_window(
            perception,
            user_profile,
            tool_profile,
            request_id=request_id,
            recent_observation=recent_observation,
        )
