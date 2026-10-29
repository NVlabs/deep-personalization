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

"""MSE key-frame selection for recordings and screenshot trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_DIFF = 100000.0
DEFAULT_VIDEO_MSE_THRESHOLD = 8000.0
DEFAULT_TRAJECTORY_MSE_THRESHOLD = 2000.0
TRAJECTORY_FILENAMES = ("trajectory.jsonl", "frames.jsonl", "traj.jsonl")


@dataclass(frozen=True)
class Frame:
    """One chronological screenshot and its elapsed-time label."""

    path: Path
    t_seconds: float
    label: str


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration_s: float


def _vision() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Frame selection dependencies are missing; install them with `uv sync`."
        ) from exc
    return cv2, np


def _label_for(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _mse(a: Any, b: Any) -> float:
    """Match the deployed sensor's per-pixel (not per-channel) MSE."""
    _, np = _vision()
    if a is None or b is None or a.shape != b.shape:
        return MAX_DIFF
    error = np.sum((a.astype("float") - b.astype("float")) ** 2)
    return float(error / float(a.shape[0] * a.shape[1]))


def _evenly_limit(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return items
    _, np = _vision()
    indices = np.linspace(0, len(items) - 1, limit)
    return [items[int(round(index))] for index in indices]


def probe_video(video_path: str | Path) -> VideoInfo:
    cv2, _ = _vision()
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or fps > 240:
            fps = 30.0
        return VideoInfo(path, fps, frame_count, frame_count / fps if frame_count else 0.0)
    finally:
        capture.release()


def detect_key_frame_times(
    video_path: str | Path,
    *,
    sample_interval: float = 1.0,
    mse_threshold: float = DEFAULT_VIDEO_MSE_THRESHOLD,
    max_key_frames: int = 0,
) -> list[float]:
    """Densely sample a video and retain screen changes relative to the last kept frame."""
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    cv2, _ = _vision()
    path = Path(video_path)
    info = probe_video(path)
    if info.duration_s <= 0:
        return []
    capture = cv2.VideoCapture(str(path))
    times: list[float] = []
    last_kept = None
    try:
        timestamp = 0.0
        while timestamp < info.duration_s:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, image = capture.read()
            if (
                ok
                and image is not None
                and (last_kept is None or _mse(last_kept, image) > mse_threshold)
            ):
                times.append(round(timestamp, 3))
                last_kept = image
            timestamp += sample_interval
    finally:
        capture.release()
    return _evenly_limit(times, max_key_frames)


def render_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    times: list[float],
    *,
    prefix: str,
) -> list[Frame]:
    """Decode selected video frames once; timestamp information stays in text metadata."""
    cv2, _ = _vision()
    path = Path(video_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frames: list[Frame] = []
    try:
        for timestamp in sorted(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            frame_path = destination / f"{prefix}_{int(timestamp * 1000):08d}ms.png"
            if not cv2.imwrite(str(frame_path), image):
                continue
            frames.append(Frame(frame_path, timestamp, _label_for(timestamp)))
    finally:
        capture.release()
    return frames


def _parse_timestamp(stamp: str) -> datetime | None:
    if not stamp:
        return None
    if "@" in stamp:
        date_part, time_part = stamp.split("@", 1)
        try:
            return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S%f")
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def trajectory_file_for_session(session_dir: str | Path) -> Path | None:
    """Find a normalized or legacy trajectory frame log in a session directory."""
    directory = Path(session_dir)
    return next(
        (directory / name for name in TRAJECTORY_FILENAMES if (directory / name).is_file()),
        None,
    )


def load_instructions(path: str | Path) -> dict[str, str]:
    """Load required session instructions from JSON, JSONL, or a JSON mapping."""
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Instruction file is empty: {source}")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        values = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
    else:
        values = decoded

    if isinstance(values, dict) and not ({"session_id", "id", "task_id"} & values.keys()):
        rows = []
        for session_id, value in values.items():
            if isinstance(value, str):
                rows.append({"session_id": session_id, "instruction": value})
            elif isinstance(value, dict):
                rows.append({"session_id": session_id, **value})
            else:
                raise ValueError(
                    f"{source}: instruction for {session_id!r} must be a string or object"
                )
    elif isinstance(values, dict):
        rows = [values]
    elif isinstance(values, list):
        rows = values
    else:
        raise ValueError(f"{source}: expected a mapping, object, list, or JSONL records")

    instructions: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{index}: expected an instruction object")
        session_id = next(
            (row.get(key) for key in ("session_id", "id", "task_id") if row.get(key)),
            None,
        )
        instruction = row.get("instruction")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"{source}:{index}: session_id must be a non-empty string")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{source}:{index}: instruction must be a non-empty string")
        if session_id in instructions:
            raise ValueError(f"{source}:{index}: duplicate session_id {session_id!r}")
        instructions[session_id] = instruction.strip()
    if not instructions:
        raise ValueError(f"Instruction file contains no usable records: {source}")
    return instructions


def key_frames_from_trajectory(
    trajectory_dir: str | Path,
    *,
    trajectory_file: str | Path | None = None,
    mse_threshold: float = DEFAULT_TRAJECTORY_MSE_THRESHOLD,
    max_key_frames: int = 0,
) -> list[Frame]:
    """Load and MSE-deduplicate screenshots from a normalized trajectory log."""
    cv2, _ = _vision()
    directory = Path(trajectory_dir)
    trajectory_path = (
        Path(trajectory_file) if trajectory_file else trajectory_file_for_session(directory)
    )
    if trajectory_path is None or not trajectory_path.exists():
        return []
    steps: list[tuple[float, Path, float | None, datetime | None]] = []
    for row_index, raw_line in enumerate(trajectory_path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        screenshot = next(
            (
                row.get(key)
                for key in ("image_path", "screenshot_file", "image", "screenshot")
                if row.get(key)
            ),
            None,
        )
        if not isinstance(screenshot, str):
            continue
        image_path = directory / screenshot
        if image_path.exists():
            raw_order = next(
                (row.get(key) for key in ("index", "step_num", "step") if row.get(key) is not None),
                row_index,
            )
            try:
                order = float(raw_order)
            except (TypeError, ValueError):
                order = float(row_index)
            raw_elapsed = next(
                (
                    row.get(key)
                    for key in ("t_seconds", "elapsed_seconds")
                    if row.get(key) is not None
                ),
                None,
            )
            try:
                elapsed = float(raw_elapsed) if raw_elapsed is not None else None
            except (TypeError, ValueError):
                elapsed = None
            raw_timestamp = next(
                (
                    row.get(key)
                    for key in ("timestamp", "action_timestamp")
                    if row.get(key) is not None
                ),
                "",
            )
            if elapsed is None and isinstance(raw_timestamp, (int, float)):
                elapsed = float(raw_timestamp)
                stamp = None
            else:
                stamp = _parse_timestamp(str(raw_timestamp))
            steps.append(
                (
                    order,
                    image_path,
                    elapsed,
                    stamp,
                )
            )
    steps.sort(key=lambda step: step[0])
    start = next((stamp for _, _, _, stamp in steps if stamp is not None), None)
    frames: list[Frame] = []
    last_kept = None
    for order, image_path, elapsed, stamp in steps:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        if last_kept is not None and mse_threshold > 0 and _mse(last_kept, image) <= mse_threshold:
            continue
        last_kept = image
        seconds = (
            max(0.0, elapsed)
            if elapsed is not None
            else max(0.0, (stamp - start).total_seconds())
            if stamp and start
            else max(0.0, order)
        )
        frames.append(Frame(image_path, round(seconds, 3), _label_for(seconds)))
    return _evenly_limit(frames, max_key_frames)
