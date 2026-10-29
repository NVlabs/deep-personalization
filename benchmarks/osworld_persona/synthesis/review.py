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

"""Render synthesized examples with their frames for human validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _images(frames: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for frame in frames:
        path = Path(frame["path"])
        label = f"t={frame.get('label', '?')}"
        if path.exists():
            lines.append(f"![{label}]({path.resolve().as_uri()})")
        else:
            lines.append(f"> frame missing: `{path}` ({label})")
    return lines


def write_review(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write a Markdown quality-control document for synthesized records."""
    yes = sum(row.get("output", {}).get("need_support") == "yes" for row in rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("input_profiles", {}).get("user_profile_id", "?"))].append(row)
    lines = [
        "# OSWorld-Persona synthesized-data review",
        "",
        f"{len(rows)} examples; need_support: yes={yes}, no={len(rows) - yes}.",
        "",
        "Check grounding, profile consistency, and suggestion quality. Mark each item "
        "accept, edit, or reject.",
        "",
        "## Label distribution by user profile",
        "",
        "| user profile | n | yes | no | direct_message | prompt_to_agent | none |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for profile_id in sorted(grouped):
        profile_rows = grouped[profile_id]
        needs = Counter(row["output"]["need_support"] for row in profile_rows)
        types = Counter(row["output"]["suggestion_type"] for row in profile_rows)
        lines.append(
            f"| `{profile_id}` | {len(profile_rows)} | {needs['yes']} | {needs['no']} | "
            f"{types['direct_message']} | {types['prompt_to_agent']} | {types['none']} |"
        )
    lines.extend(["", "---", ""])
    for index, row in enumerate(rows, 1):
        output = row["output"]
        lines.extend(
            [
                f"## {index}. `{row['example_id']}` — "
                f"{output['need_support']} / {output['suggestion_type']}",
                "",
                f"Profiles: user=`{row['input_profiles']['user_profile_id']}`, "
                f"tool=`{row['input_profiles']['tool_profile_id']}`  ",
                f"Task: {row.get('instruction', '') or '—'}",
                "",
                "**User Preference Memory**",
                "",
                f"```\n{row['input']['user_profile']}\n```",
                "",
                "**Available Agent Profile**",
                "",
                f"```\n{row['input']['tool_profile']}\n```",
                "",
                "**Recent Observation**",
                "",
                f"```\n{row['input'].get('recent_observation', '') or '(none)'}\n```",
                "",
                "**Screen History**",
                "",
            ]
        )
        lines.extend(_images(row.get("window", {}).get("frames", [])))
        lines.extend(
            [
                "",
                "**Output**",
                "",
                f"```json\n{json.dumps(output, indent=2, ensure_ascii=False)}\n```",
                "",
                "**Validation:** [ ] accept  [ ] edit  [ ] reject — notes:",
                "",
                "---",
                "",
            ]
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
