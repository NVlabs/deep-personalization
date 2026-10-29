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

"""Synthetic user-preference and available-tool profile pools."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class UserProfile:
    """A stable ID and the preference text exposed to the labeler."""

    id: str
    text: str


@dataclass(frozen=True)
class ToolProfile:
    """A stable ID and the tools the proactive assistant may offer."""

    id: str
    text: str


USER_PROFILES = (
    UserProfile(
        "reserved_drafts",
        "The user appreciates occasional proactive suggestions when they can noticeably improve "
        "productivity, but prefers not to receive frequent notifications.\n"
        "Offer help when the expected benefit is moderate or high.\n"
        "The user prefers AI to prepare drafts or intermediate work that they can review and "
        "edit.\nWhenever appropriate, generate partial artifacts instead of only providing advice.",
    ),
    UserProfile(
        "minimal_interruptions",
        "The user strongly dislikes interruptions and wants to stay in flow.\n"
        "Only intervene when the user is clearly blocked or about to make a costly, "
        "hard-to-reverse mistake.\nKeep any message extremely short and skippable.\n"
        "Prefer staying silent when the user is making steady progress on their own.",
    ),
    UserProfile(
        "eager_handson",
        "The user likes a hands-on assistant that actively looks for ways to help.\n"
        "Offer suggestions whenever there is a plausible productivity gain, even a small one.\n"
        "The user is happy to delegate whole sub-tasks to agents and review the results "
        "afterward.\n"
        "Prefer doing the work (drafting, running, automating) over just explaining how.",
    ),
    UserProfile(
        "learning_oriented",
        "The user wants to learn and improve their own skills.\n"
        "When intervening, briefly explain the why, not just the fix.\n"
        "Prefer nudges and guidance over silently taking over the task.\n"
        "Offer help when the user seems stuck or is repeating an ineffective approach.",
    ),
    UserProfile(
        "automation_first",
        "The user values automation and offloading repetitive or mechanical work.\n"
        "Proactively offer to script, batch, or hand off multi-step chores to an agent.\n"
        "For one-off trivial actions the user can finish themselves, stay out of the way.\n"
        "When suggesting, prefer routing the task to a capable agent with a ready-to-run prompt.",
    ),
)

TOOL_PROFILES = (
    ToolProfile("chat_and_code", "Available tools: ChatGPT, Claude Code."),
    ToolProfile("chat_only", "Available tools: ChatGPT."),
    ToolProfile(
        "code_agent",
        "Available tools: Claude Code (a terminal coding agent that can run shell commands, "
        "edit files, and execute scripts).",
    ),
    ToolProfile(
        "browser_and_code",
        "Available tools: a Browser-Use agent (navigates and acts on live web pages), a CLI "
        "agent (runs shell commands and processes files).",
    ),
    ToolProfile(
        "full_suite",
        "Available tools: ChatGPT, Claude Code, a Browser-Use web agent, and a CLI "
        "automation agent.",
    ),
)


def sample_profiles(seed: int, key: object) -> tuple[UserProfile, ToolProfile]:
    """Choose a deterministic profile pair from a stable example key."""
    rng = random.Random(f"{seed}:{key}:profiles")
    return rng.choice(USER_PROFILES), rng.choice(TOOL_PROFILES)


def user_profile_by_id(profile_id: str) -> UserProfile | None:
    return next((profile for profile in USER_PROFILES if profile.id == profile_id), None)


def tool_profile_by_id(profile_id: str) -> ToolProfile | None:
    return next((profile for profile in TOOL_PROFILES if profile.id == profile_id), None)
