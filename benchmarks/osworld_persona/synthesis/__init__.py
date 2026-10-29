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

"""Data synthesis pipeline to convert computer-use data into personalized proactive agent datasets."""

from .build import (
    BuildConfig,
    DatasetSynthesizer,
    SessionSeed,
    balance_records,
    frame_windows,
)
from .consistency import attach_consistency, consistency_vote
from .frames import (
    Frame,
    detect_key_frame_times,
    key_frames_from_trajectory,
    load_instructions,
    render_video_frames,
    trajectory_file_for_session,
)
from .pipeline import Perception, SynthesisPipeline, SynthesisResult
from .profiles import TOOL_PROFILES, USER_PROFILES, ToolProfile, UserProfile
from .split import label_counts, record_session_id, split_records_by_session

__all__ = [
    "BuildConfig",
    "DatasetSynthesizer",
    "Frame",
    "Perception",
    "SessionSeed",
    "SynthesisPipeline",
    "SynthesisResult",
    "TOOL_PROFILES",
    "USER_PROFILES",
    "ToolProfile",
    "UserProfile",
    "attach_consistency",
    "balance_records",
    "consistency_vote",
    "detect_key_frame_times",
    "frame_windows",
    "key_frames_from_trajectory",
    "label_counts",
    "load_instructions",
    "record_session_id",
    "render_video_frames",
    "split_records_by_session",
    "trajectory_file_for_session",
]
