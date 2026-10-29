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

"""OSWorld-Persona benchmark implementation and method adapters."""

from benchmarks.common import GenerationRequest, Predictor, TextGenerator, TextRequest
from methods import PromptCondition

from .data import DEFAULT_PERSONAS, load_split, resolve_dataset_root
from .method_adapter import (
    coco_lora_builder,
    coco_lora_trainer,
    coco_pe_trainer,
    induce_persona,
    load_revised_data,
    to_method_examples,
    to_method_predictions,
    write_revised_data,
)
from .metrics import BinaryMetrics, EvaluationMetrics, compute_metrics
from .run import BenchmarkResult, BenchmarkRunner
from .schema import Annotation, Decision, PredictionRecord
from .scoring import score_external_predictions

__all__ = [
    "Annotation",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BinaryMetrics",
    "DEFAULT_PERSONAS",
    "Decision",
    "EvaluationMetrics",
    "GenerationRequest",
    "PredictionRecord",
    "Predictor",
    "PromptCondition",
    "TextGenerator",
    "TextRequest",
    "coco_lora_builder",
    "coco_lora_trainer",
    "coco_pe_trainer",
    "compute_metrics",
    "induce_persona",
    "load_split",
    "load_revised_data",
    "resolve_dataset_root",
    "score_external_predictions",
    "to_method_examples",
    "to_method_predictions",
    "write_revised_data",
]
