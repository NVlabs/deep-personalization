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

"""Manage an upstream LLaMA-Factory API server for a local evaluation run."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LlamaFactoryServer:
    """Start, await, and stop the exact LLaMA-Factory process launched here."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        config_file: str | Path,
        base_model: str,
        adapter_path: str | Path,
        template: str = "qwen3_5_nothink",
        infer_backend: str = "huggingface",
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str | None = None,
        startup_timeout: float = 600.0,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        raw_config = Path(config_file).expanduser()
        self.config_file = (
            raw_config.resolve()
            if raw_config.is_absolute()
            else (self.repo_path / raw_config).resolve()
        )
        self.adapter_path = Path(adapter_path).expanduser().resolve()
        self.base_model = base_model
        self.template = template
        self.infer_backend = infer_backend
        self.host = host
        self.port = port
        self.api_key = api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None

        if not self.base_model.strip():
            raise ValueError("base_model must be non-empty")
        if not self.template.strip():
            raise ValueError("template must be non-empty")
        if not self.infer_backend.strip():
            raise ValueError("infer_backend must be non-empty")
        if not self.host.strip():
            raise ValueError("host must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def command(self) -> list[str]:
        return [
            "uv",
            "run",
            "--project",
            str(self.repo_path),
            "llamafactory-cli",
            "api",
            str(self.config_file),
            f"model_name_or_path={self.base_model}",
            f"adapter_name_or_path={self.adapter_path}",
            f"template={self.template}",
            f"infer_backend={self.infer_backend}",
        ]

    def start(self) -> LlamaFactoryServer:
        if self.process is not None:
            raise RuntimeError("LLaMA-Factory server has already been started")
        if not (self.repo_path / "pyproject.toml").is_file():
            raise FileNotFoundError(f"LLaMA-Factory checkout not found: {self.repo_path}")
        if not self.config_file.is_file():
            raise FileNotFoundError(f"LLaMA-Factory inference config not found: {self.config_file}")
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"COCO-LoRA adapter not found: {self.adapter_path}")

        environment = dict(os.environ)
        environment["API_HOST"] = self.host
        environment["API_PORT"] = str(self.port)
        environment["API_MODEL_NAME"] = self.base_model
        # LLaMA-Factory otherwise logs the complete multimodal request, including
        # megabytes of base64-encoded screenshot data, for every example.
        environment["API_VERBOSE"] = "0"
        if self.api_key:
            environment["API_KEY"] = self.api_key
        self.process = subprocess.Popen(
            self.command,
            cwd=self.repo_path,
            env=environment,
            start_new_session=True,
        )
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_until_ready(self) -> None:
        if self.process is None:
            raise RuntimeError("LLaMA-Factory server has not been started")
        deadline = time.monotonic() + self.startup_timeout
        models_url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"LLaMA-Factory server exited before becoming ready (code {return_code})"
                )
            try:
                with urlopen(Request(models_url, headers=headers), timeout=5) as response:
                    if 200 <= response.status < 300:
                        return
            except HTTPError as error:
                if error.code in {401, 403}:
                    raise RuntimeError(
                        f"LLaMA-Factory rejected the readiness probe at {models_url}; "
                        "check --api-key"
                    ) from error
                last_error = error
            except (URLError, TimeoutError) as error:
                last_error = error
            time.sleep(2)
        detail = f": {last_error}" if last_error else ""
        raise TimeoutError(
            f"LLaMA-Factory server was not ready at {models_url} after "
            f"{self.startup_timeout:g} seconds{detail}"
        )

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def __enter__(self) -> LlamaFactoryServer:
        return self.start()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.stop()
