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

"""Load, validate, and sample OSWorld-Persona data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schema import Annotation, normalize_need_support

DEFAULT_PERSONAS = (
    "automation_first",
    "eager_handson",
    "learning_oriented",
    "reserved_drafts",
)
_TIMESTAMP = re.compile(r"\d+:\d{2}\Z")


def resolve_dataset_root(
    data_root: str | Path | None = None,
    *,
    dataset_id: str | None = None,
    revision: str | None = None,
) -> Path:
    """Resolve a local snapshot, optionally downloading it from Hugging Face."""
    if data_root is not None:
        path = Path(data_root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        return path

    if not dataset_id:
        raise ValueError(
            "Pass --data-root or the published Hugging Face repository ID, "
            "for example --dataset-id <organization>/osworld-persona."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face client is missing; run `uv sync` or pass --data-root."
        ) from exc
    return Path(
        snapshot_download(repo_id=dataset_id, repo_type="dataset", revision=revision)
    ).resolve()


def _annotation_file(root: Path, persona: str, split: str) -> Path:
    direct_candidates = (
        root / persona / split / "annotations.json",
        root / "data" / persona / split / "annotations.json",
        root / "osworld-persona" / persona / split / "annotations.json",
    )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.glob(f"**/{persona}/{split}/annotations.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find {persona}/{split}/annotations.json below {root}")
    raise ValueError(f"Ambiguous dataset layout for {persona}/{split}: {matches}")


def _text(row: dict[str, Any], field: str, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {field} must be a non-empty string")
    return value.strip()


def _user_intent(row: dict[str, Any], location: str) -> str | tuple[str, ...]:
    """Preserve the released schema's string and multi-intent list variants."""
    value = row.get("user_intent")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item.strip() for item in value)
    ):
        return tuple(item.strip() for item in value)
    raise ValueError(f"{location}: user_intent must be a string or non-empty string list")


def load_persona_split(
    root: str | Path,
    persona: str,
    split: str,
    *,
    validate_images: bool = True,
) -> list[Annotation]:
    root_path = Path(root).expanduser().resolve()
    annotation_path = _annotation_file(root_path, persona, split)
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {annotation_path}")

    examples: list[Annotation] = []
    for index, row in enumerate(rows):
        location = f"{annotation_path}:{index}"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: annotation must be an object")
        images = row.get("images")
        timestamps = row.get("timestamps")
        session = row.get("session")
        if (
            not isinstance(images, list)
            or not images
            or not all(isinstance(x, str) for x in images)
        ):
            raise ValueError(f"{location}: images must be a non-empty string list")
        if not isinstance(timestamps, list) or len(timestamps) != len(images):
            raise ValueError(f"{location}: expected one timestamp per image")
        if not all(isinstance(x, str) and _TIMESTAMP.fullmatch(x) for x in timestamps):
            raise ValueError(f"{location}: timestamps must use elapsed mm:ss strings")
        if not isinstance(session, str) or not session:
            raise ValueError(f"{location}: session must be a non-empty string")
        if {Path(name).parent.name for name in images} != {session}:
            raise ValueError(f"{location}: image directories do not match session {session!r}")
        image_paths = tuple((annotation_path.parent / name).resolve() for name in images)
        if validate_images:
            missing = [path for path in image_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{location}: missing image {missing[0]}")
        label = normalize_need_support(row.get("want_support"))
        if label is None:
            raise ValueError(f"{location}: invalid want_support label")
        required_context = row.get("required_context_for_support", [])
        if not isinstance(required_context, list) or not all(
            isinstance(item, str) for item in required_context
        ):
            raise ValueError(f"{location}: required_context_for_support must be a string list")

        examples.append(
            Annotation(
                id=f"{persona}:{split}:{index:06d}",
                persona=persona,
                split=split,
                index=index,
                session_id=session,
                image_paths=image_paths,
                timestamps=tuple(timestamps),
                observation=_text(row, "summarized_observation", location),
                # Ten current training rows contain multiple intents as a list.
                user_intent=_user_intent(row, location),
                want_support=label,
                rationale=_text(row, "rationale", location),
                required_context_for_support=tuple(required_context),
            )
        )
    return examples


def load_split(
    root: str | Path,
    split: str = "test",
    personas: tuple[str, ...] | list[str] | None = None,
    *,
    validate_images: bool = True,
) -> list[Annotation]:
    """Load personas in a deterministic order while preserving source order."""
    selected = tuple(personas) if personas is not None else DEFAULT_PERSONAS
    examples: list[Annotation] = []
    for persona in selected:
        examples.extend(load_persona_split(root, persona, split, validate_images=validate_images))
    return examples


def sample_frames(example: Annotation, max_images: int) -> Annotation:
    """Evenly cap images while keeping first and last frames and timestamps aligned."""
    count = len(example.image_paths)
    if max_images <= 0 or count <= max_images:
        return example
    if max_images == 1:
        indices = [0]
    else:
        indices = sorted({round(i * (count - 1) / (max_images - 1)) for i in range(max_images)})
    return Annotation(
        **{
            **example.__dict__,
            "image_paths": tuple(example.image_paths[index] for index in indices),
            "timestamps": tuple(example.timestamps[index] for index in indices),
        }
    )
