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

"""Native Tinker sampling for framework-neutral benchmark requests."""

from __future__ import annotations

import base64
import mimetypes
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from benchmarks.common import GenerationRequest, TextRequest

IMAGE_PLACEHOLDER = "<image>"
_SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
RequestT = TypeVar("RequestT", GenerationRequest, TextRequest)


def _load_runtime() -> SimpleNamespace:
    try:
        import tinker
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.model_info import (
            get_model_attributes,
            get_recommended_renderer_name,
        )
        from tinker_cookbook.renderers import get_renderer, get_text_content
    except ImportError as error:
        raise ImportError(
            "TinkerPredictor requires Python 3.11+ and the dependencies installed "
            "by `uv sync`."
        ) from error
    return SimpleNamespace(
        tinker=tinker,
        get_image_processor=get_image_processor,
        get_model_attributes=get_model_attributes,
        get_recommended_renderer_name=get_recommended_renderer_name,
        get_renderer=get_renderer,
        get_text_content=get_text_content,
    )


def _image_data_url(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark image does not exist: {path}")
    mime_type, _encoding = mimetypes.guess_type(path.name)
    if mime_type not in _SUPPORTED_IMAGE_TYPES:
        supported = ", ".join(sorted(_SUPPORTED_IMAGE_TYPES))
        raise ValueError(f"Unsupported Tinker image type for {path}: expected {supported}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _generation_messages(request: GenerationRequest) -> list[dict[str, Any]]:
    pieces = request.user_prompt.split(IMAGE_PLACEHOLDER)
    if len(pieces) - 1 != len(request.image_paths):
        raise ValueError(
            f"Request {request.id!r} has {len(pieces) - 1} image placeholders "
            f"for {len(request.image_paths)} images"
        )

    content: str | list[dict[str, Any]]
    if request.image_paths:
        parts: list[dict[str, Any]] = []
        for index, piece in enumerate(pieces):
            if piece:
                parts.append({"type": "text", "text": piece})
            if index < len(request.image_paths):
                parts.append(
                    {
                        "type": "image",
                        "image": _image_data_url(request.image_paths[index]),
                    }
                )
        content = parts
    else:
        content = request.user_prompt
    return [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": content},
    ]


class TinkerPredictor:
    """Implement multimodal and text generation through Tinker's native SDK."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        renderer_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_concurrency: int = 1,
        seed: int | None = None,
        service_client: Any | None = None,
    ) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError("Native Tinker inference requires Python 3.11 or newer")
        if not model.strip():
            raise ValueError("model must be non-empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        runtime = _load_runtime()
        if service_client is None:
            client_options = {}
            if api_key:
                client_options["api_key"] = api_key
            if base_url:
                client_options["base_url"] = base_url
            service_client = runtime.tinker.ServiceClient(**client_options)

        if model.startswith("tinker://"):
            sampling_client = service_client.create_sampling_client(model_path=model)
        else:
            sampling_client = service_client.create_sampling_client(base_model=model)
        base_model = sampling_client.get_base_model()
        attributes = runtime.get_model_attributes(base_model)
        image_processor = (
            runtime.get_image_processor(base_model) if attributes.is_vl else None
        )
        renderer_name = renderer_name or runtime.get_recommended_renderer_name(base_model)

        self.model = model
        self.base_model = base_model
        self.supports_images = attributes.is_vl
        self.renderer_name = renderer_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.seed = seed
        self.runtime = runtime
        self.sampling_client = sampling_client
        self.renderer = runtime.get_renderer(
            renderer_name,
            sampling_client.get_tokenizer(),
            image_processor=image_processor,
            model_name=base_model,
        )

    def _submit(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        prompt = self.renderer.build_generation_prompt(messages)
        parameters = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": self.renderer.get_stop_sequences(),
        }
        if self.seed is not None:
            parameters["seed"] = self.seed
        return self.sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=self.runtime.tinker.SamplingParams(**parameters),
        )

    def _resolve(self, request_id: str, future: Any) -> str:
        response = future.result()
        if not response.sequences:
            raise RuntimeError(f"Tinker returned no samples for {request_id}")
        message, _termination = self.renderer.parse_response(response.sequences[0].tokens)
        output = self.runtime.get_text_content(message)
        if not isinstance(output, str):
            raise RuntimeError(f"Tinker returned no text for {request_id}")
        return output

    def _generate(
        self,
        requests: Sequence[RequestT],
        message_builder: Callable[[RequestT], list[dict[str, Any]]],
        settings: Callable[[RequestT], tuple[float, int]],
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(requests), self.max_concurrency):
            batch = requests[start : start + self.max_concurrency]
            pending = []
            for request in batch:
                temperature, max_tokens = settings(request)
                pending.append(
                    (
                        request.id,
                        self._submit(
                            message_builder(request),
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                    )
                )
            outputs.extend(self._resolve(request_id, future) for request_id, future in pending)
        return outputs

    def predict(self, requests: Sequence[GenerationRequest]) -> Sequence[str]:
        if not self.supports_images and any(request.image_paths for request in requests):
            raise ValueError(f"Tinker model {self.base_model!r} does not support image input")
        return self._generate(
            requests,
            _generation_messages,
            lambda _request: (self.temperature, self.max_tokens),
        )

    def generate_text(self, requests: Sequence[TextRequest]) -> Sequence[str]:
        return self._generate(
            requests,
            lambda request: [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            lambda request: (request.temperature, request.max_tokens),
        )
