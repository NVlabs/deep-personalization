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

"""OSWorld-Persona prompts and prompt rendering."""

from __future__ import annotations

import json
from importlib.resources import files

from methods.prompting import PromptCondition

IMAGE_PLACEHOLDER = "<image>"
RECENT_OBSERVATION_LIMIT = 5

# Oracle-persona baseline: appended to the observer system prompt at evaluation
# time. It deliberately exposes the ground-truth profile used to synthesize the
# user's labels, providing an upper-bound personalization condition.
ORACLE_PROFILE_BLOCK = """\


Oracle User Persona

The following is the ground-truth user preference profile that was used to
synthesize this user's labels. Apply it when deciding whether proactive support
would benefit this user and what form that support should take.

<user_profile>
{profile}
</user_profile>
"""

# Induced-persona baseline: appended to the observer system prompt after a
# profile has been inferred from that user's training history. Unlike the oracle
# block, this frames the profile as uncertain so current visual evidence wins.
INDUCED_PROFILE_BLOCK = """\


Induced User Persona

The following user preference profile was inferred from prior interaction
history. Treat it as an uncertain behavioral prior: apply it when deciding
whether proactive support would benefit this user, but let clear evidence in
the current context override it.

<user_profile source="induced">
{profile}
</user_profile>
"""

# COCO-PE inference prompt: prepended to each benchmark request after prompt
# evolution. The memory contains reusable preference rules learned from labeled
# training examples; bullet IDs are removed before evaluation.
LEARNED_MEMORY_BLOCK = """<learned_user_memory>
What you have learned so far about when/how THIS user wants proactive help:
{memory}
</learned_user_memory>

"""

# Induced-persona stage 1: asks a text model to digest anonymized interaction
# history into evidence-backed behavioral patterns. The system and user strings
# are verbatim from the original leakage-safe induction implementation.
INDUCTION_DIGESTION_SYSTEM = """You analyze a single user's past interactions with a proactive desktop assistant.
Each anonymized history record contains what was happening, the user's immediate
intent, and observed binary feedback about whether proactive help was wanted.

Answer one task-specific question: What recurring, evidence-supported patterns
distinguish moments when this user welcomes proactive support from moments when
they prefer the assistant to stay silent, and what assistance style is supported
by the history?

Focus on timing, visible progress or friction, task scale and repetitiveness,
reversibility, and the form of assistance only when the records establish it.
Do not infer a preference from topic or application alone. Do not assume that
correlation is causation. Produce 3-5 distinct patterns and cite only 2-6 of the
most representative supplied history IDs for each pattern; never enumerate all
matching records. Preserve conflicts or uncertainty. Never guess a named persona
or mention dataset construction.

Respond with ONLY a JSON object:
{
  "patterns": [
    {
      "dimension": "support_timing" | "silence_boundary" | "task_characteristics" | "assistance_style" | "general",
      "claim": "<concise evidence-supported pattern>",
      "evidence_ids": ["h-0001"]
    }
  ],
  "uncertainties": ["<important ambiguity or unsupported preference>"]
}
"""

# Induced-persona stage 1 user message: encloses only the leakage-reviewed
# Personal Context and asks the model for the structured digestion above.
INDUCTION_DIGESTION_USER = """<personal_context>
{context}
</personal_context>

Digest this history in the requested direction. Return only the JSON object.
"""

# Induced-persona stage 2: converts the structured digestion into the concise
# profile inserted by INDUCED_PROFILE_BLOCK. Both strings are verbatim from the
# original implementation and retain its uncertainty and leakage constraints.
INDUCTION_PROFILE_SYSTEM = """You generate an explainable personal profile for a proactive desktop assistant.
You receive the same anonymized Personal Context plus evidence-grounded guidance
from a prior digestion step. Convert them into a compact profile describing when
this user welcomes proactive support, when they prefer silence, and how support
should be offered when the evidence establishes a style preference.

Rules:
- Every profile statement must be supported by the Personal Context and guidance.
- Preserve meaningful exceptions, conflicts, and uncertainty.
- Do not invent preferences about assistance form when only timing is observed.
- Do not mention history IDs, named personas, datasets, labels, or this procedure.
- Write 3-8 concise descriptive sentences suitable for direct use by an observer.

Respond with ONLY a JSON object:
{
  "profile": "<the descriptive personal profile>",
  "limitations": ["<important uncertainty the observer should retain>"]
}
"""

# Induced-persona stage 2 user message: combines the same Personal Context with
# validated digestion guidance before requesting the final reusable profile.
INDUCTION_PROFILE_USER = """<personal_context>
{context}
</personal_context>

<digestion_guidance>
{guidance}
</digestion_guidance>

Generate the personal profile. Return only the JSON object.
"""

# COCO-PE reflection stage: compares a consecutive batch of predictions with
# binary labels, identifies which memory bullets helped or hurt, and proposes
# durable user-preference rules. This is intentionally verbatim from the
# original experiment implementation so released runs preserve its behavior.
COCO_PE_REFLECTOR_SYSTEM = """You are the reflection module of a self-improving proactive desktop assistant.
The assistant maintains a MEMORY (a list of bullets, each with an id) describing
one specific user's preferences for proactive support.

You are given a consecutive, oldest-to-newest sequence of examples from one
period of the user's activity. For every example, the ONLY ground-truth field is
the observed need_support label. The observation, intent, rationale, suggestion,
and any other fields inside assistant_prediction were generated by the assistant
and may be incomplete or wrong. Never treat them as ground truth.

Some examples are marked REFLECT and the rest are CONTEXT_ONLY. Produce one
reflection for every REFLECT example and none for CONTEXT_ONLY examples. Use the
nearby examples to identify stable temporal patterns and decision boundaries,
while keeping each reflection grounded in its target example's prediction and
need_support label. Do not invent a reason that is not supported by the sequence.

Diagnose the gap and distill durable, REUSABLE lessons about this user:
- If the decision was wrong, identify the most defensible missing or misleading
  preference rule. If the available evidence does not establish why, say that
  the label contradicts the decision without inventing a rationale.
- If the decision was right, note which memory bullets contributed so they can
  be reinforced. Only propose a new insight when the consecutive evidence
  supports a preference not yet in memory.
- Insights must be about the USER'S PREFERENCES and decision boundaries
  ("in situation X this user wants / does not want Y"), not a single example.
- Never mention sequence indices, example ids, or screenshots in an insight.

Respond with ONLY a JSON object:
{
  "reflections": [
    {
      "sequence_index": <integer index of a REFLECT example>,
      "reflection": "<1-3 sentences: what went wrong or right and why>",
      "helpful_bullet_ids": ["<ids of memory bullets that pointed the right way>"],
      "harmful_bullet_ids": ["<ids of memory bullets that pointed the wrong way>"],
      "proposed_insights": [
        {"section": "when_to_support" | "when_to_stay_silent" | "how_to_support" | "general",
         "content": "<one concrete, generalizable preference rule for this user>"}
      ]
    }
  ]
}
Return exactly one item per REFLECT example. Propose at most 2 insights per
reflection; an empty list is fine.
"""

# COCO-PE reflection input: separates the current memory from an ordered batch
# and reiterates that only marked examples should produce reflections.
COCO_PE_REFLECTOR_TEMPLATE = """<current_memory>
{memory}
</current_memory>

<consecutive_examples order="oldest_to_newest">
{examples}
</consecutive_examples>

Reflect only on examples marked REFLECT. Remember that need_support is the only
ground-truth field. Return only the JSON object.
"""

# COCO-PE curation stage: turns reflections into bounded add/update/delete
# operations over the persistent prompt memory. This is also verbatim from the
# original experiment; {max_ops} is substituted before sending the request.
COCO_PE_CURATOR_SYSTEM = """You are the curator of the memory of a self-improving proactive desktop
assistant. The memory is a sectioned list of bullets, each with an id,
describing ONE user's preferences for proactive support.

You are given the current memory and a batch of reflections (lessons from
recent training examples, each possibly proposing new insights). Update the
memory with INCREMENTAL delta operations — do not rewrite it wholesale:

- "add":    a genuinely new preference rule not covered by any existing bullet.
- "update": rewrite an existing bullet to be more precise (e.g. sharpen its
            condition, merge in a nuance from a reflection).
- "delete": remove a bullet that reflections showed to be wrong or misleading.

Rules:
- Keep bullets atomic, concrete and operational ("when X, do/don't Y"), one
  decision rule per bullet.
- Do NOT add duplicates or near-duplicates of existing bullets; prefer "update"
  to sharpen an existing bullet instead.
- Prefer few high-value operations over many marginal ones.

Respond with ONLY a JSON object:
{
  "ops": [
    {"op": "add", "section": "when_to_support" | "when_to_stay_silent" | "how_to_support" | "general", "content": "..."},
    {"op": "update", "id": "<existing bullet id>", "content": "..."},
    {"op": "delete", "id": "<existing bullet id>"}
  ]
}
At most {max_ops} operations; an empty list is fine.
"""

# COCO-PE curator input: supplies the current memory plus the reflector's
# compact, verdict-tagged lessons and asks only for incremental operations.
COCO_PE_CURATOR_TEMPLATE = """<current_memory>
{memory}
</current_memory>

<reflections>
{reflections}
</reflections>

Return only the JSON object with the delta operations.
"""

# COCO-LoRA data-revision prompt: appended only when a base prediction is wrong
# or malformed. It is verbatim from the original self-reflection implementation,
# including its response schema and wording, to preserve training-data behavior.
COCO_LORA_REFLECTION_BLOCK = """<self_reflection>
Your previous response fail to capture the user's preferences:
<previous_response>
{previous_response}
</previous_response>

Ground-truth feedback from the user:
- Correct need_support label: {ground_truth_label}
- Rationale: {ground_truth_rationale}

Revise the complete response based on the user's feedback. Preserve parts of the observation and user intent that remain supported by the screenshots and previous recent memory. Correct the support decision and rationale. Do not mention this feedback or the revision process in the answer.

Respond with ONLY this JSON schema:
{{
  "observation": "<compact incremental memory update>",
  "user_intent": "<the user's likely immediate goal>",
  "need_support": "yes" | "no",
  "rationale": "<brief grounded explanation>"
}}
</self_reflection>"""


def observer_prompt() -> str:
    """Load the shared vanilla observer system prompt used by every condition."""
    return (
        files("benchmarks.osworld_persona.assets")
        .joinpath("observer_prompt.txt")
        .read_text()
        .strip()
    )


def oracle_personas() -> dict[str, str]:
    text = files("benchmarks.osworld_persona.assets").joinpath("oracle_personas.json").read_text()
    return json.loads(text)


def render_user_prompt(timestamps: tuple[str, ...], previous_memory: str | None) -> str:
    """Build the temporal-memory and ordered-screenshot portion of an observer request."""
    screenshot_lines = "\n".join(
        f"[t={timestamp}] Screenshot {position} of {len(timestamps)}"
        for position, timestamp in enumerate(timestamps, 1)
    )
    memory = previous_memory if previous_memory is not None else "(none)"
    memory = memory.replace(IMAGE_PLACEHOLDER, "<image_literal>")
    return (
        f"<previous_recent_memory>\n{memory}\n</previous_recent_memory>\n\n"
        f"<screenshots>\n{screenshot_lines}\n"
        "Images follow this text in the order listed above.\n"
        f"{IMAGE_PLACEHOLDER * len(timestamps)}\n</screenshots>"
    )


def accumulated_memory(observations: list[str], limit: int = RECENT_OBSERVATION_LIMIT) -> str:
    if not observations:
        return "(none)"
    return "\n".join(
        f"[{position}] {observation.replace(IMAGE_PLACEHOLDER, '<image_literal>')}"
        for position, observation in enumerate(observations[-limit:], 1)
    )


def apply_condition(
    condition: PromptCondition,
    user_id: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str]:
    """Render a generic method condition using OSWorld-Persona-specific blocks."""
    if condition.name == "oracle_persona":
        system_prompt = system_prompt.rstrip() + ORACLE_PROFILE_BLOCK.format(
            profile=condition.profile_for(user_id)
        )
    elif condition.name == "induced_persona":
        system_prompt = system_prompt.rstrip() + INDUCED_PROFILE_BLOCK.format(
            profile=condition.profile_for(user_id)
        )
    elif condition.name == "coco_pe":
        user_prompt = (
            LEARNED_MEMORY_BLOCK.format(memory=condition.memory_for(user_id)) + user_prompt
        )
    return system_prompt, user_prompt
