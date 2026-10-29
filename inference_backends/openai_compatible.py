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

"""OpenAI-compatible multimodal inference adapter.

This adapter works with the public LLaMA-Factory API as well as other servers
that implement OpenAI's chat-completions request shape. Images are embedded as
data URLs so the benchmark process and model server need not share a filesystem.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.common import GenerationRequest, TextRequest

IMAGE_PLACEHOLDER = "<image>"
_IMAGE_MIME_TYPES = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in _IMAGE_MIME_TYPES:
        supported = ", ".join(sorted(_IMAGE_MIME_TYPES))
        raise ValueError(
            f"Unsupported image type {suffix!r} for {path}; expected one of {supported}"
        )
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark image does not exist: {path}") from exc
    return f"data:{_IMAGE_MIME_TYPES[suffix]};base64,{encoded}"


def openai_user_content(request: GenerationRequest) -> list[dict[str, Any]]:
    """Convert one request while preserving every image placeholder's position."""
    text_parts = request.user_prompt.split(IMAGE_PLACEHOLDER)
    if len(text_parts) - 1 != len(request.image_paths):
        raise ValueError(
            f"Request {request.id!r} has {len(text_parts) - 1} image placeholders "
            f"for {len(request.image_paths)} images"
        )

    content: list[dict[str, Any]] = []
    for index, text in enumerate(text_parts):
        if text:
            content.append({"type": "text", "text": text})
        if index < len(request.image_paths):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(request.image_paths[index])},
                }
            )
    return content


class OpenAICompatiblePredictor:
    """Send benchmark requests to an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 600.0,
        max_retries: int = 2,
        max_concurrency: int = 1,
        seed: int | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.seed = seed
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The OpenAI client is missing; install project dependencies with `uv sync`."
                ) from exc
            client = OpenAI(
                base_url=base_url,
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "0"),
                timeout=timeout,
                max_retries=max_retries,
            )
        self.client = client

    def _complete(
        self,
        *,
        request_id: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        parameters = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.seed is not None:
            parameters["seed"] = self.seed
        response = self.client.chat.completions.create(**parameters)
        if not response.choices:
            raise RuntimeError(f"Inference server returned no choices for {request_id}")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError(f"Inference server returned no text for {request_id}")
        return content

    def _predict_one(self, request: GenerationRequest) -> str:
        return self._complete(
            request_id=request.id,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": openai_user_content(request)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _generate_text_one(self, request: TextRequest) -> str:
        return self._complete(
            request_id=request.id,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    def predict(self, requests: Sequence[GenerationRequest]) -> Sequence[str]:
        if self.max_concurrency == 1 or len(requests) <= 1:
            return [self._predict_one(request) for request in requests]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(requests))) as executor:
            return list(executor.map(self._predict_one, requests))

    def generate_text(self, requests: Sequence[TextRequest]) -> Sequence[str]:
        """Generate text for method and synthesis stages through the same API."""
        if self.max_concurrency == 1 or len(requests) <= 1:
            return [self._generate_text_one(request) for request in requests]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(requests))) as executor:
            return list(executor.map(self._generate_text_one, requests))
