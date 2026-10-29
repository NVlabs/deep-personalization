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

"""Benchmark-independent personalization methods."""

from .coco_lora import (
    CocoLoRADataBuilder,
    CocoLoRAResult,
    CocoLoRATrainer,
    LoRATrainingBackend,
    LoRATrainingConfig,
    TrainedAdapter,
)
from .coco_pe import CocoPETrainer, PromptMemory
from .induced_persona import InducedPersona, induce_persona
from .prompting import PromptCondition
from .types import (
    InductionRecord,
    MethodExample,
    MethodPrediction,
    RevisedTrainingExample,
)

__all__ = [
    "CocoLoRADataBuilder",
    "CocoLoRAResult",
    "CocoLoRATrainer",
    "CocoPETrainer",
    "InductionRecord",
    "InducedPersona",
    "MethodExample",
    "MethodPrediction",
    "LoRATrainingBackend",
    "LoRATrainingConfig",
    "PromptCondition",
    "PromptMemory",
    "RevisedTrainingExample",
    "TrainedAdapter",
    "induce_persona",
]
