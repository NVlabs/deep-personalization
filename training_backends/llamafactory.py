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

"""LLaMA-Factory implementation of the COCO-LoRA training backend."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Event, Thread
from typing import Any

from methods import LoRATrainingConfig, TrainedAdapter
from methods.types import RevisedTrainingExample

CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], None]


def _default_runner(command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> None:
    subprocess.run(list(command), cwd=cwd, env=dict(env), check=True)


def _watch_training_log(
    path: Path,
    stop: Event,
    progress: Callable[[int, int], None],
) -> None:
    """Forward LLaMA-Factory trainer-log entries as optimizer-step progress."""
    offset = 0
    pending = b""

    def consume() -> None:
        nonlocal offset, pending
        try:
            size = path.stat().st_size
            if size < offset:
                offset = 0
                pending = b""
            with path.open("rb") as source:
                source.seek(offset)
                chunk = source.read()
                offset = source.tell()
        except FileNotFoundError:
            return
        if not chunk:
            return
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for raw_line in lines:
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            completed = record.get("current_steps", record.get("step"))
            total = record.get("total_steps")
            if (
                isinstance(completed, int)
                and isinstance(total, int)
                and 0 <= completed <= total
                and total > 0
            ):
                progress(completed, total)

    while not stop.wait(0.2):
        consume()
    consume()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    if not slug:
        raise ValueError("run_name must contain at least one letter or number")
    return slug


def _sharegpt_rows(examples: Sequence[RevisedTrainingExample]) -> list[dict[str, Any]]:
    return [
        {
            "id": revised.example.id,
            "messages": [
                {"role": "system", "content": revised.example.request.system_prompt},
                {"role": "user", "content": revised.example.request.user_prompt},
                {"role": "assistant", "content": revised.target},
            ],
            "images": [
                str(path.expanduser().resolve()) for path in revised.example.request.image_paths
            ],
        }
        for revised in examples
    ]


class LlamaFactoryTrainingBackend:
    """Launch LoRA SFT through an unmodified LLaMA-Factory checkout."""

    def __init__(
        self,
        repo_path: str | Path,
        config_file: str | Path,
        *,
        launcher: Sequence[str] | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        lora_target: str = "all",
        extra_overrides: Mapping[str, str | int | float | bool] | None = None,
        env: Mapping[str, str] | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        raw_config = Path(config_file).expanduser()
        self.config_file = (
            raw_config.resolve()
            if raw_config.is_absolute()
            else (self.repo_path / raw_config).resolve()
        )
        self.launcher = tuple(
            launcher
            or (
                "uv",
                "run",
                "--project",
                str(self.repo_path),
                "llamafactory-cli",
                "train",
            )
        )
        if not self.launcher:
            raise ValueError("launcher must be non-empty")
        if lora_alpha is not None and lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0 <= lora_dropout < 1:
            raise ValueError("lora_dropout must be between zero and one")
        if not lora_target.strip():
            raise ValueError("lora_target must be non-empty")
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target = lora_target
        self.extra_overrides = dict(extra_overrides or {})
        self.env = dict(env or {})
        self.runner = runner or _default_runner

    def train(
        self,
        examples: Sequence[RevisedTrainingExample],
        config: LoRATrainingConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> TrainedAdapter:
        if not examples:
            raise ValueError("LLaMA-Factory training requires at least one example")
        project_file = self.repo_path / "pyproject.toml"
        if not project_file.is_file():
            raise FileNotFoundError(f"LLaMA-Factory pyproject.toml not found: {project_file}")
        if not self.config_file.is_file():
            raise FileNotFoundError(f"LLaMA-Factory config not found: {self.config_file}")

        owned_overrides = {
            "model_name_or_path",
            "stage",
            "do_train",
            "finetuning_type",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target",
            "dataset",
            "dataset_dir",
            "media_dir",
            "output_dir",
            "run_name",
            "cutoff_len",
            "per_device_train_batch_size",
            "learning_rate",
            "num_train_epochs",
            "seed",
            "data_seed",
            "disable_tqdm",
            "logging_steps",
            "logging_strategy",
        }
        for key in self.extra_overrides:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid LLaMA-Factory override name: {key!r}")
            if key in owned_overrides:
                raise ValueError(f"Use LoRATrainingConfig instead of overriding {key!r}")

        output_dir = config.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite training output: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        input_dir = output_dir.parent / f".{output_dir.name}-coco-lora-input"
        if input_dir.exists():
            raise FileExistsError(f"Refusing to overwrite prepared training data: {input_dir}")
        input_dir.mkdir()

        dataset_name = f"coco_lora_{_slug(config.run_name).replace('-', '_')}"
        dataset_path = input_dir / f"{dataset_name}.json"
        dataset_path.write_text(
            json.dumps(_sharegpt_rows(examples), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dataset_info = {
            dataset_name: {
                "file_name": dataset_path.name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
        }
        (input_dir / "dataset_info.json").write_text(
            json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        overrides: dict[str, str | int | float | bool] = {
            "model_name_or_path": config.base_model,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": config.rank,
            "lora_alpha": self.lora_alpha if self.lora_alpha is not None else 2 * config.rank,
            "lora_dropout": self.lora_dropout,
            "lora_target": self.lora_target,
            "dataset": dataset_name,
            "dataset_dir": input_dir,
            "media_dir": "/",
            "output_dir": output_dir,
            "run_name": config.run_name,
            "cutoff_len": config.max_seq_length,
            "per_device_train_batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_epochs,
            "seed": config.seed,
            "data_seed": config.seed,
        }
        overrides.update(self.extra_overrides)
        if progress:
            # Log every optimizer step for the watcher and avoid displaying a
            # second progress bar from Transformers in the child process.
            overrides.update(
                disable_tqdm=True,
                logging_steps=1,
                logging_strategy="steps",
            )
        command = [
            *self.launcher,
            str(self.config_file),
            *[
                f"{key}={str(value).lower() if isinstance(value, bool) else value}"
                for key, value in overrides.items()
            ],
        ]
        environment = {**os.environ, **self.env}
        stop = Event()
        watcher = None
        if progress:
            watcher = Thread(
                target=_watch_training_log,
                args=(output_dir / "trainer_log.jsonl", stop, progress),
                daemon=True,
            )
            watcher.start()
        try:
            self.runner(command, self.repo_path, environment)
        finally:
            if watcher:
                stop.set()
                watcher.join()
        return TrainedAdapter(
            backend="llamafactory",
            base_model=config.base_model,
            adapter_uri=str(output_dir),
            metadata={
                "command": command,
                "dataset_path": str(dataset_path),
                "num_examples": len(examples),
            },
        )
