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

"""Tests for optional inference adapters without external services."""

from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.common import GenerationRequest, TextRequest
from inference_backends import llamafactory_server as server_module
from inference_backends import tinker as tinker_module
from inference_backends.llamafactory_server import LlamaFactoryServer
from inference_backends.openai_compatible import OpenAICompatiblePredictor, openai_user_content
from inference_backends.tinker import TinkerPredictor


def test_openai_content_preserves_image_positions(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    request = GenerationRequest("id", "system", "before<image>middle<image>after", (first, second))

    content = openai_user_content(request)

    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert content[0]["text"] == "before"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_content_rejects_placeholder_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "one.png"
    image.touch()
    request = GenerationRequest("id", "system", "no placeholder", (image,))
    with pytest.raises(ValueError, match="image placeholders"):
        openai_user_content(request)


def test_openai_predictor_sends_multimodal_chat_request(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(b"screen")
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='{"need_support":"no"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    predictor = OpenAICompatiblePredictor(
        "served-model", base_url="http://localhost:8000/v1", seed=42, client=client
    )
    request = GenerationRequest("id", "observer", "look<image>", (image,))

    assert predictor.predict([request]) == ['{"need_support":"no"}']
    assert calls[0]["model"] == "served-model"
    assert calls[0]["seed"] == 42
    assert calls[0]["messages"][0] == {"role": "system", "content": "observer"}
    assert calls[0]["messages"][1]["content"][1]["type"] == "image_url"


def test_openai_predictor_supports_text_generation() -> None:
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content="generated")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    predictor = OpenAICompatiblePredictor(
        "served-model", base_url="http://localhost:8000/v1", client=client
    )
    request = TextRequest("label", "system", "user", temperature=0.6, max_tokens=321)

    assert predictor.generate_text([request]) == ["generated"]
    assert calls[0]["messages"][1] == {"role": "user", "content": "user"}
    assert calls[0]["temperature"] == 0.6
    assert calls[0]["max_tokens"] == 321


def test_tinker_predictor_renders_native_multimodal_and_text_requests(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(b"screen")
    calls = {"prompts": [], "parameters": []}

    class Future:
        def __init__(self, output: str) -> None:
            self.output = output

        def result(self):
            return SimpleNamespace(
                sequences=[SimpleNamespace(tokens=[self.output])],
            )

    class Renderer:
        def build_generation_prompt(self, messages):
            calls["prompts"].append(messages)
            return messages

        def get_stop_sequences(self):
            return [999]

        def parse_response(self, tokens):
            return {"role": "assistant", "content": tokens[0]}, "stop"

    renderer = Renderer()

    class SamplingClient:
        def get_base_model(self):
            return "Qwen/Qwen3.5-9B"

        def get_tokenizer(self):
            return "tokenizer"

        def sample(self, *, prompt, num_samples, sampling_params):
            assert prompt and num_samples == 1
            calls["parameters"].append(sampling_params)
            return Future(f"result-{len(calls['parameters'])}")

    class ServiceClient:
        def __init__(self, **kwargs):
            assert kwargs == {"api_key": "tml-key"}

        def create_sampling_client(self, **kwargs):
            assert kwargs == {"model_path": "tinker://run/sampler_weights/final"}
            return SamplingClient()

    runtime = SimpleNamespace(
        tinker=SimpleNamespace(
            ServiceClient=ServiceClient,
            SamplingParams=lambda **kwargs: kwargs,
        ),
        get_model_attributes=lambda _model: SimpleNamespace(is_vl=True),
        get_image_processor=lambda model: f"processor:{model}",
        get_recommended_renderer_name=lambda _model: "qwen3_5",
        get_renderer=lambda name, tokenizer, **kwargs: (
            renderer
            if (name, tokenizer, kwargs)
            == (
                "qwen3_5",
                "tokenizer",
                {
                    "image_processor": "processor:Qwen/Qwen3.5-9B",
                    "model_name": "Qwen/Qwen3.5-9B",
                },
            )
            else None
        ),
        get_text_content=lambda message: message["content"],
    )
    monkeypatch.setattr(tinker_module, "_load_runtime", lambda: runtime)

    predictor = TinkerPredictor(
        "tinker://run/sampler_weights/final",
        api_key="tml-key",
        temperature=0.2,
        max_tokens=777,
        max_concurrency=2,
        seed=9,
    )
    generation = GenerationRequest(
        "vision",
        "system",
        "before<image>after",
        (image,),
    )
    text_request = TextRequest("text", "meta-system", "meta-user", 0.6, 321)

    assert predictor.predict([generation]) == ["result-1"]
    assert predictor.generate_text([text_request]) == ["result-2"]
    assert [part["type"] for part in calls["prompts"][0][1]["content"]] == [
        "text",
        "image",
        "text",
    ]
    assert calls["prompts"][0][1]["content"][1]["image"].startswith(
        "data:image/png;base64,"
    )
    assert calls["parameters"] == [
        {"max_tokens": 777, "temperature": 0.2, "stop": [999], "seed": 9},
        {"max_tokens": 321, "temperature": 0.6, "stop": [999], "seed": 9},
    ]


def test_llamafactory_server_manages_only_its_process_group(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "LlamaFactory"
    config = repo / "examples" / "inference" / "qwen3vl.yaml"
    adapter = tmp_path / "adapter"
    config.parent.mkdir(parents=True)
    config.write_text("template: qwen3_5_nothink\n")
    (repo / "pyproject.toml").write_text("[project]\nname='stub'\nversion='0'\n")
    adapter.mkdir()
    calls = {}

    class StubProcess:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            calls["wait_timeout"] = timeout
            return 0

    def stub_popen(command, **kwargs):
        calls["command"] = command
        calls["popen_kwargs"] = kwargs
        return StubProcess()

    class StubResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(server_module.subprocess, "Popen", stub_popen)
    monkeypatch.setattr(server_module, "urlopen", lambda *_args, **_kwargs: StubResponse())
    monkeypatch.setattr(
        server_module.os,
        "killpg",
        lambda pid, sig: calls.setdefault("signals", []).append((pid, sig)),
    )

    server = LlamaFactoryServer(
        repo_path=repo,
        config_file="examples/inference/qwen3vl.yaml",
        base_model="Qwen/base",
        adapter_path=adapter,
        port=8123,
    )
    with server:
        assert server.base_url == "http://127.0.0.1:8123/v1"

    assert "adapter_name_or_path=" + str(adapter) in calls["command"]
    assert calls["popen_kwargs"]["env"]["API_PORT"] == "8123"
    assert calls["popen_kwargs"]["env"]["API_MODEL_NAME"] == "Qwen/base"
    assert calls["popen_kwargs"]["env"]["API_VERBOSE"] == "0"
    assert calls["popen_kwargs"]["start_new_session"] is True
    assert calls["signals"] == [(4321, signal.SIGTERM)]
    assert calls["wait_timeout"] == 30
