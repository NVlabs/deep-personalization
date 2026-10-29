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

"""Tinker implementation of the Coco-LoRA training backend."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from methods import LoRATrainingConfig, TrainedAdapter
from methods.types import RevisedTrainingExample

IMAGE_PLACEHOLDER = "<image>"


def _load_runtime() -> SimpleNamespace:
    try:
        import tinker
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import TrainOnWhat, get_renderer
        from tinker_cookbook.supervised.data import conversation_to_datum
    except ImportError as error:
        raise ImportError(
            "TinkerBackend requires Python 3.11+ and the dependencies installed by `uv sync`."
        ) from error
    return SimpleNamespace(
        tinker=tinker,
        get_image_processor=get_image_processor,
        get_recommended_renderer_name=get_recommended_renderer_name,
        get_renderer=get_renderer,
        train_on_last_assistant=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        conversation_to_datum=conversation_to_datum,
    )


def _image_data_url(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"Unsupported Tinker image type for {path}: {mime_type}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def to_tinker_conversation(revised: RevisedTrainingExample) -> list[dict[str, Any]]:
    """Render one generic multimodal example as an OpenAI-style Tinker conversation."""
    request = revised.example.request
    pieces = request.user_prompt.split(IMAGE_PLACEHOLDER)
    if len(pieces) - 1 != len(request.image_paths):
        raise ValueError(
            f"{request.id} has {len(request.image_paths)} images but "
            f"{len(pieces) - 1} image placeholders"
        )
    if request.image_paths:
        user_content: str | list[dict[str, Any]] = []
        for index, piece in enumerate(pieces):
            if piece:
                user_content.append({"type": "text", "text": piece})
            if index < len(request.image_paths):
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(request.image_paths[index])},
                    }
                )
    else:
        user_content = request.user_prompt
    return [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": revised.target},
    ]


class TinkerBackend:
    """Run multimodal LoRA SFT through Tinker's hosted training API."""

    def __init__(
        self,
        *,
        renderer_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        checkpoint_name: str | None = None,
        checkpoint_ttl_seconds: int | None = None,
        service_client: Any = None,
    ) -> None:
        self.renderer_name = renderer_name
        self.base_url = base_url
        self.api_key = api_key
        self.checkpoint_name = checkpoint_name
        self.checkpoint_ttl_seconds = checkpoint_ttl_seconds
        self.service_client = service_client

    def train(
        self,
        examples: Sequence[RevisedTrainingExample],
        config: LoRATrainingConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> TrainedAdapter:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.train_async(examples, config, progress=progress))
        raise RuntimeError(
            "TinkerBackend.train() cannot run inside an event loop; await train_async()"
        )

    async def train_async(
        self,
        examples: Sequence[RevisedTrainingExample],
        config: LoRATrainingConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> TrainedAdapter:
        if not examples:
            raise ValueError("Tinker training requires at least one example")
        if sys.version_info < (3, 11):
            raise RuntimeError("Tinker training requires Python 3.11 or newer")
        runtime = _load_runtime()

        output_dir = config.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite training output: {output_dir}")
        output_dir.mkdir(parents=True)

        client_options = {}
        if self.base_url:
            client_options["base_url"] = self.base_url
        if self.api_key:
            client_options["api_key"] = self.api_key
        service_client = self.service_client or runtime.tinker.ServiceClient(**client_options)
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.base_model,
            rank=config.rank,
        )
        tokenizer = training_client.get_tokenizer()
        renderer_name = self.renderer_name or runtime.get_recommended_renderer_name(
            config.base_model
        )
        renderer_kwargs = {}
        if any(revised.example.request.image_paths for revised in examples):
            renderer_kwargs["image_processor"] = runtime.get_image_processor(config.base_model)
        renderer = runtime.get_renderer(renderer_name, tokenizer, **renderer_kwargs)
        data = [
            runtime.conversation_to_datum(
                to_tinker_conversation(revised),
                renderer,
                max_length=config.max_seq_length,
                train_on_what=runtime.train_on_last_assistant,
            )
            for revised in examples
        ]

        rng = random.Random(config.seed)
        steps = 0
        steps_per_epoch = (len(data) + config.batch_size - 1) // config.batch_size
        total_steps = config.num_epochs * steps_per_epoch
        for _epoch in range(config.num_epochs):
            order = list(range(len(data)))
            rng.shuffle(order)
            for start in range(0, len(order), config.batch_size):
                batch = [data[index] for index in order[start : start + config.batch_size]]
                forward_future = await training_client.forward_backward_async(
                    batch, loss_fn="cross_entropy"
                )
                optimizer_future = await training_client.optim_step_async(
                    runtime.tinker.AdamParams(learning_rate=config.learning_rate)
                )
                await forward_future.result_async()
                await optimizer_future.result_async()
                steps += 1
                if progress:
                    progress(steps, total_steps)

        checkpoint_name = self.checkpoint_name or f"{config.run_name}-final"
        # A sampler checkpoint can be loaded by TinkerPredictor. A training-state
        # checkpoint is only suitable for resuming optimization, not sampling.
        save_future = await training_client.save_weights_for_sampler_async(
            checkpoint_name,
            ttl_seconds=self.checkpoint_ttl_seconds,
        )
        save_result = await save_future.result_async()
        adapter_uri = str(save_result.path)
        metadata = {
            "renderer_name": renderer_name,
            "num_examples": len(examples),
            "steps": steps,
            "checkpoint_name": checkpoint_name,
        }
        (output_dir / "training_artifact.json").write_text(
            json.dumps(
                {
                    "backend": "tinker",
                    "base_model": config.base_model,
                    "adapter_uri": adapter_uri,
                    "metadata": metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return TrainedAdapter("tinker", config.base_model, adapter_uri, metadata)
