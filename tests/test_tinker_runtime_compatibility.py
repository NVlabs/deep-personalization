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

"""Offline checks against installed Tinker libraries, not simulated runtimes."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from inference_backends.tinker import _load_runtime as load_inference_runtime
from training_backends.tinker import _load_runtime as load_training_runtime


def test_installed_tinker_interfaces_accept_backend_arguments() -> None:
    from tinker.lib.public_interfaces import SamplingClient, TrainingClient

    runtime = load_inference_runtime()
    calls = [
        (runtime.tinker.ServiceClient.create_sampling_client, {"base_model": "model"}),
        (runtime.tinker.ServiceClient.create_sampling_client, {"model_path": "tinker://test"}),
        (
            runtime.tinker.ServiceClient.create_lora_training_client_async,
            {"base_model": "model", "rank": 32},
        ),
        (
            SamplingClient.sample,
            {
                "prompt": runtime.tinker.ModelInput.from_ints([1]),
                "num_samples": 1,
                "sampling_params": runtime.tinker.SamplingParams(
                    max_tokens=8, temperature=0.0, stop=[1], seed=9
                ),
            },
        ),
        (TrainingClient.forward_backward_async, {"data": [], "loss_fn": "cross_entropy"}),
        (
            TrainingClient.save_weights_for_sampler_async,
            {"name": "test-final", "ttl_seconds": None},
        ),
    ]
    for method, kwargs in calls:
        inspect.signature(method).bind(None, **kwargs)
    # The backend passes AdamParams positionally, so validate that exact call shape.
    inspect.signature(TrainingClient.optim_step_async).bind(
        None, runtime.tinker.AdamParams(learning_rate=1e-4)
    )


def test_installed_renderer_and_training_datum_work_without_downloads() -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokenizer_backend = Tokenizer(
        WordLevel({"[UNK]": 0, "System": 1, "User": 2, "Assistant": 3, ":": 4}, unk_token="[UNK]")
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token="[UNK]")
    inference = load_inference_runtime()
    training = load_training_runtime()
    renderer = inference.get_renderer("role_colon", tokenizer, model_name="offline-test")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    prompt = renderer.build_generation_prompt(messages)
    assert isinstance(prompt, inference.tinker.ModelInput)
    assert prompt.length > 0
    parsed, _termination = renderer.parse_response(tokenizer.encode("answer"))
    assert isinstance(inference.get_text_content(parsed), str)
    datum = training.conversation_to_datum(
        [*messages, {"role": "assistant", "content": "answer"}],
        renderer,
        max_length=128,
        train_on_what=training.train_on_last_assistant,
    )
    assert isinstance(datum, inference.tinker.Datum)
    assert datum.model_input.length > 0
    assert "weights" in datum.loss_fn_inputs


def test_installed_cookbook_recognizes_project_vision_model() -> None:
    runtime = load_inference_runtime()
    assert runtime.get_model_attributes("Qwen/Qwen3.5-9B").is_vl
    assert runtime.get_recommended_renderer_name("Qwen/Qwen3.5-9B")
