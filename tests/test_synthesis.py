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

"""Tests for framework-neutral OSWorld-Persona synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.osworld_persona.synthesis import (
    BuildConfig,
    DatasetSynthesizer,
    Frame,
    SessionSeed,
    SynthesisPipeline,
    balance_records,
    frame_windows,
    split_records_by_session,
)
from benchmarks.osworld_persona.synthesis.cli import main as synthesis_main
from benchmarks.osworld_persona.synthesis.frames import (
    key_frames_from_trajectory,
    load_instructions,
)


class FakeBackend:
    def __init__(self) -> None:
        self.multimodal_requests = []
        self.text_requests = []

    def predict(self, requests):
        self.multimodal_requests.extend(requests)
        return [
            '{"observation":"The user edited a document.","user_intent":"finish the document"}'
            for _ in requests
        ]

    def generate_text(self, requests):
        self.text_requests.extend(requests)
        outputs = []
        for request in requests:
            if request.metadata.get("synthesis_stage") == "initial_observation":
                outputs.append("The user opened the document editor.")
            else:
                outputs.append(
                    '{"need_support":"no","rationale":"Steady progress.",'
                    '"suggestion_type":"direct_message","suggestion":"ignored"}'
                )
        return outputs


def _frames(tmp_path: Path, count: int = 4) -> tuple[Frame, ...]:
    return tuple(
        Frame(tmp_path / f"{index}.png", float(index), f"00:0{index}") for index in range(count)
    )


def test_frame_windows_keeps_a_valid_trailing_window(tmp_path: Path) -> None:
    windows = frame_windows(_frames(tmp_path, 7), size=5, stride=5, minimum=2, limit=0)
    assert [len(window) for _, window in windows] == [5, 2]
    assert [index for index, _ in windows] == [0, 1]


def test_pipeline_places_timestamped_labels_before_image_placeholders(tmp_path: Path) -> None:
    backend = FakeBackend()
    pipeline = SynthesisPipeline(backend, backend, attempts=1, retry_delay=0)

    perception = pipeline.perceive_window(
        list(_frames(tmp_path, 2)),
        request_id="task::win00",
        instruction_hint="write a report",
        recent_observation="The user opened the editor.",
    )

    assert perception is not None
    request = backend.multimodal_requests[0]
    assert request.user_prompt.count("<image>") == 2
    assert request.user_prompt.index("[t=00:00]") < request.user_prompt.index("[t=00:01]")
    assert request.metadata["synthesis_stage"] == "perceive"


def test_dataset_builder_reuses_one_perception_for_profile_cross_product(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    pipeline = SynthesisPipeline(backend, backend, attempts=1, retry_delay=0)
    builder = DatasetSynthesizer(
        pipeline,
        config=BuildConfig(
            window_size=2,
            window_stride=2,
            min_window=2,
            max_windows=1,
            use_initial_observation=False,
            user_profile_ids=("automation_first", "learning_oriented"),
            tool_profile_ids=("chat_only",),
        ),
        model_name="test-model",
    )

    rows = builder.build_session(
        SessionSeed("task", _frames(tmp_path, 2), "write a report", "trajectory", tmp_path)
    )

    assert len(rows) == 2
    assert all(row["session_id"] == "task" for row in rows)
    assert all("task_id" not in row for row in rows)
    assert len(backend.multimodal_requests) == 1
    assert len(backend.text_requests) == 2
    assert all(row["output"]["suggestion_type"] == "none" for row in rows)
    assert all(row["output"]["suggestion"] == "" for row in rows)


def test_dataset_builder_resume_reuses_stored_perception(tmp_path: Path) -> None:
    backend = FakeBackend()
    pipeline = SynthesisPipeline(backend, backend, attempts=1, retry_delay=0)
    base_config = dict(
        window_size=2,
        window_stride=2,
        min_window=2,
        max_windows=1,
        use_initial_observation=False,
        tool_profile_ids=("chat_only",),
    )
    first = DatasetSynthesizer(
        pipeline,
        config=BuildConfig(user_profile_ids=("automation_first",), **base_config),
    ).build_session(SessionSeed("task", _frames(tmp_path, 2)))
    backend.multimodal_requests.clear()
    backend.text_requests.clear()

    resumed = DatasetSynthesizer(
        pipeline,
        config=BuildConfig(
            user_profile_ids=("automation_first", "learning_oriented"), **base_config
        ),
    ).build_session(
        SessionSeed("task", _frames(tmp_path, 2)),
        existing_by_id={first[0]["example_id"]: first[0]},
    )

    assert len(resumed) == 1
    assert resumed[0]["input_profiles"]["user_profile_id"] == "learning_oriented"
    assert backend.multimodal_requests == []
    assert len(backend.text_requests) == 1


def test_balance_records_is_even_and_deterministic() -> None:
    rows = [
        {"example_id": f"yes-{index}", "output": {"need_support": "yes"}} for index in range(3)
    ] + [{"example_id": "no-0", "output": {"need_support": "no"}}]
    balanced = balance_records(rows, seed=7)
    assert len(balanced) == 2
    assert {row["output"]["need_support"] for row in balanced} == {"yes", "no"}
    assert balanced == balance_records(rows, seed=7)


def test_trajectory_frame_selection_with_synthesis_extra(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    rows = []
    for index, value in enumerate((0, 0, 255)):
        image_path = tmp_path / f"step_{index}.png"
        cv2.imwrite(str(image_path), np.full((8, 8, 3), value, dtype=np.uint8))
        rows.append(
            {
                "step_num": index,
                "screenshot_file": image_path.name,
                "action_timestamp": f"20260101@12000{index}000000",
            }
        )
    (tmp_path / "traj.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    frames = key_frames_from_trajectory(tmp_path, mse_threshold=2000)

    assert [frame.path.name for frame in frames] == ["step_0.png", "step_2.png"]
    assert [frame.label for frame in frames] == ["00:00", "00:02"]


def test_normalized_trajectory_and_instruction_mapping(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    session = tmp_path / "session-001"
    session.mkdir()
    for index, value in enumerate((0, 255)):
        cv2.imwrite(
            str(session / f"frame-{index}.png"),
            np.full((8, 8, 3), value, dtype=np.uint8),
        )
    (session / "trajectory.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "image_path": f"frame-{index}.png",
                    "t_seconds": seconds,
                }
            )
            + "\n"
            for index, seconds in enumerate((0.0, 4.2))
        )
    )
    instruction_file = tmp_path / "instructions.json"
    instruction_file.write_text(json.dumps({"session-001": "Create the report"}))

    frames = key_frames_from_trajectory(session)

    assert [frame.label for frame in frames] == ["00:00", "00:04"]
    assert load_instructions(instruction_file) == {"session-001": "Create the report"}


def test_split_records_keeps_sessions_together_and_is_deterministic() -> None:
    rows = [
        {
            "example_id": f"{session}::win{window:02d}::{persona}",
            "session_id": session,
            "input_profiles": {"user_profile_id": persona},
            "output": {"need_support": "yes" if window == 0 else "no"},
        }
        for session in ("session-a", "session-b", "session-c", "session-d")
        for window in range(2)
        for persona in ("automation_first", "learning_oriented")
    ]

    train, test, manifest = split_records_by_session(rows, test_ratio=0.25, seed=7)
    repeated = split_records_by_session(list(reversed(rows)), test_ratio=0.25, seed=7)
    train_sessions = {row["session_id"] for row in train}
    test_sessions = {row["session_id"] for row in test}

    assert train_sessions.isdisjoint(test_sessions)
    assert len(test_sessions) == 1
    assert manifest["train"]["num_examples"] == 12
    assert manifest["test"]["num_examples"] == 4
    assert manifest["test"]["labels"] == {"no": 2, "yes": 2}
    assert repeated[2] == manifest


def test_split_records_rejects_a_single_session() -> None:
    with pytest.raises(ValueError, match="at least two"):
        split_records_by_session(
            [
                {
                    "example_id": "only::win00",
                    "session_id": "only",
                    "output": {"need_support": "no"},
                }
            ]
        )


def test_split_cli_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "dataset_consistent.jsonl"
    rows = [
        {
            "example_id": f"{session}::win00",
            "session_id": session,
            "output": {"need_support": label},
        }
        for session, label in (("one", "yes"), ("two", "no"), ("three", "no"))
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output_dir = tmp_path / "split"

    assert (
        synthesis_main(
            [
                "split",
                "--input",
                str(input_path),
                "--out-dir",
                str(output_dir),
                "--test-ratio",
                "0.34",
                "--seed",
                "9",
            ]
        )
        == 0
    )

    train = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (output_dir / "test.jsonl").read_text().splitlines()]
    manifest = json.loads((output_dir / "split_manifest.json").read_text())
    assert len(train) == 2
    assert len(test) == 1
    assert manifest["strategy"] == "session"
    assert manifest["input"] == str(input_path.resolve())
