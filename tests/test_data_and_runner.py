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

"""OSWorld-Persona data and runner tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import benchmarks.osworld_persona.cli as cli_module
import inference_backends
import training_backends
from benchmarks.osworld_persona.cli import main
from benchmarks.osworld_persona.data import load_split
from benchmarks.osworld_persona.run import BenchmarkRunner
from methods import PromptMemory


def _write_dataset(root: Path) -> None:
    rows = [
        {
            "images": ["images/001/a.png", "images/001/b.png"],
            "timestamps": ["00:00", "00:10"],
            "session": "001",
            "summarized_observation": "first",
            "user_intent": "work",
            "want_support": "no",
            "rationale": "progress",
            "required_context_for_support": [],
        },
        {
            "images": ["images/002/a.png"],
            "timestamps": ["00:00"],
            "session": "002",
            "summarized_observation": "other",
            "user_intent": "work",
            "want_support": "yes",
            "rationale": "blocked",
            "required_context_for_support": ["error"],
        },
        {
            "images": ["images/001/c.png"],
            "timestamps": ["00:20"],
            "session": "001",
            "summarized_observation": "second",
            "user_intent": "work",
            "want_support": "no",
            "rationale": "progress",
            "required_context_for_support": [],
        },
    ]
    split = root / "automation_first" / "test"
    split.mkdir(parents=True)
    image_names = (
        "images/001/a.png",
        "images/001/b.png",
        "images/001/c.png",
        "images/002/a.png",
    )
    for relative in image_names:
        image = split / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
    (split / "annotations.json").write_text(json.dumps(rows), encoding="utf-8")


class RecordingPredictor:
    def __init__(self) -> None:
        self.batches = []

    def predict(self, requests):
        self.batches.append(requests)
        outputs = []
        for request in requests:
            outputs.append(
                json.dumps(
                    {
                        "observation": f"memory for {request.id}",
                        "user_intent": "work",
                        "rationale": "ok",
                        "need_support": "yes" if ":000001" in request.id else "no",
                    }
                )
            )
        return outputs


def test_load_and_run_sessions_in_dependency_order(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    examples = load_split(tmp_path, personas=["automation_first"])
    predictor = RecordingPredictor()
    progress = []
    result = BenchmarkRunner(predictor, batch_size=8).run(
        examples, progress=lambda completed, total: progress.append((completed, total))
    )

    assert [len(batch) for batch in predictor.batches] == [2, 1]
    assert "memory for automation_first:test:000000" in predictor.batches[1][0].user_prompt
    assert result.metrics.overall.accuracy == 1.0
    assert result.metrics.overall.parse_coverage == 1.0
    assert [record.id for record in result.predictions] == [example.id for example in examples]
    assert progress == [(2, 3), (3, 3)]


def test_loader_rejects_session_image_mismatch(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    path = tmp_path / "automation_first" / "test" / "annotations.json"
    rows = json.loads(path.read_text())
    rows[0]["session"] = "wrong"
    path.write_text(json.dumps(rows))
    try:
        load_split(tmp_path, personas=["automation_first"])
    except ValueError as error:
        assert "do not match session" in str(error)
    else:
        raise AssertionError("invalid session was accepted")


def test_run_cli_writes_native_benchmark_result(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)

    class StubPredictor:
        def __init__(self, model, **kwargs) -> None:
            assert model == "served-model"
            assert kwargs["base_url"] == "http://localhost:8000/v1"

        def predict(self, requests):
            return [
                json.dumps(
                    {
                        "observation": "current activity",
                        "user_intent": "work",
                        "rationale": "no interruption",
                        "need_support": "no",
                    }
                )
                for _ in requests
            ]

    monkeypatch.setattr(inference_backends, "OpenAICompatiblePredictor", StubPredictor)
    output = tmp_path / "result.json"
    exit_code = main(
        [
            "run",
            "--data-root",
            str(tmp_path),
            "--personas",
            "automation_first",
            "--condition",
            "vanilla",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "served-model",
            "--limit",
            "2",
            "--output",
            str(output),
        ]
    )

    result = json.loads(output.read_text())
    assert exit_code == 0
    assert result["metadata"]["method"] == "vanilla"
    assert result["metadata"]["num_examples"] == 2
    assert len(result["predictions"]) == 2


def test_run_cli_selects_native_tinker_backend(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)
    calls = {}

    class StubPredictor:
        def __init__(self, model, **kwargs) -> None:
            calls["model"] = model
            calls["kwargs"] = kwargs

        def predict(self, requests):
            return ['{"need_support":"no"}' for _request in requests]

    monkeypatch.setattr(inference_backends, "TinkerPredictor", StubPredictor)
    output = tmp_path / "tinker-result.json"

    assert (
        main(
            [
                "run",
                "--backend",
                "tinker",
                "--data-root",
                str(tmp_path),
                "--personas",
                "automation_first",
                "--condition",
                "vanilla",
                "--model",
                "Qwen/Qwen3.5-9B",
                "--api-key",
                "tml-key",
                "--tinker-renderer",
                "qwen3_5_disable_thinking",
                "--limit",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls["model"] == "Qwen/Qwen3.5-9B"
    assert calls["kwargs"]["api_key"] == "tml-key"
    assert calls["kwargs"]["base_url"] is None
    assert calls["kwargs"]["renderer_name"] == "qwen3_5_disable_thinking"
    assert json.loads(output.read_text())["metadata"]["backend"] == "tinker"


def test_induce_persona_cli_writes_condition_mapping_and_audit(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)

    class StubGenerator:
        def __init__(self, model, **kwargs) -> None:
            assert model == "served-model"
            assert kwargs["base_url"] == "http://localhost:8000/v1"

    def stub_induce(history, _generator, **kwargs):
        assert len(history) == 2
        assert kwargs["max_per_label"] == 12
        assert kwargs["seed"] == 7
        kwargs["progress"](1, 2)
        kwargs["progress"](2, 2)
        return SimpleNamespace(
            profile="The user prefers concise help.",
            limitations=("Limited history",),
            selected_ids=tuple(example.id for example in history),
            digestion={"patterns": []},
        )

    monkeypatch.setattr(inference_backends, "OpenAICompatiblePredictor", StubGenerator)
    monkeypatch.setattr(cli_module, "induce_persona", stub_induce)
    output = tmp_path / "artifacts" / "profiles.json"
    audit_output = tmp_path / "artifacts" / "audit.json"

    exit_code = main(
        [
            "induce-persona",
            "--data-root",
            str(tmp_path),
            "--split",
            "test",
            "--personas",
            "automation_first",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "served-model",
            "--max-per-label",
            "12",
            "--limit",
            "2",
            "--seed",
            "7",
            "--output",
            str(output),
            "--audit-output",
            str(audit_output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text()) == {"automation_first": "The user prefers concise help."}
    audit = json.loads(audit_output.read_text())["automation_first"]
    assert audit["limitations"] == ["Limited history"]
    assert len(audit["selected_ids"]) == 2


def test_coco_pe_cli_supports_per_persona_smoke_limit(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)

    class StubBackend:
        def __init__(self, model, **kwargs) -> None:
            assert model == "served-model"
            assert kwargs["base_url"] == "http://localhost:8000/v1"

    class StubRunner:
        def __init__(self, _backend, **kwargs) -> None:
            assert kwargs["max_images"] == 4

        def run(self, history, **_kwargs):
            assert len(history) == 2
            return SimpleNamespace(predictions=["base-one", "base-two"])

    class StubTrainer:
        def fit(self, examples, **kwargs):
            assert examples == ["adapted"]
            assert kwargs["epochs"] == 1
            assert kwargs["shuffle"] is True
            assert isinstance(kwargs["memory"], PromptMemory)
            assert callable(kwargs["checkpoint"])
            memory = kwargs["memory"]
            memory.apply(
                [
                    {
                        "op": "add",
                        "section": "when_to_support",
                        "content": "Offer help after repeated errors.",
                    }
                ],
                1,
            )
            record = {"batch": 1, "n_reflection_targets": 1, "n_reflections": 1}
            kwargs["checkpoint"](memory, record)
            kwargs["progress"](2, 2)
            return memory, [record]

    def stub_coco_pe_trainer(_predictor, _generator, **kwargs):
        assert kwargs["batch_size"] == 2
        return StubTrainer()

    def stub_to_method_examples(history, predictions, **kwargs):
        assert len(history) == 2
        assert predictions == ["base-one", "base-two"]
        assert kwargs == {"max_images": 4}
        return ["adapted"]

    monkeypatch.setattr(inference_backends, "OpenAICompatiblePredictor", StubBackend)
    monkeypatch.setattr(cli_module, "BenchmarkRunner", StubRunner)
    monkeypatch.setattr(cli_module, "coco_pe_trainer", stub_coco_pe_trainer)
    monkeypatch.setattr(cli_module, "to_method_examples", stub_to_method_examples)
    output = tmp_path / "artifacts" / "memories.json"
    audit_output = tmp_path / "artifacts" / "memory-audit.json"

    exit_code = main(
        [
            "evolve-coco-pe",
            "--data-root",
            str(tmp_path),
            "--split",
            "test",
            "--personas",
            "automation_first",
            "--limit",
            "2",
            "--evolution-batch-size",
            "2",
            "--max-images",
            "4",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "served-model",
            "--output",
            str(output),
            "--audit-output",
            str(audit_output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text()) == {
        "automation_first": ("## When to proactively support\n- Offer help after repeated errors.")
    }
    assert json.loads(audit_output.read_text())["automation_first"][0]["batch"] == 1
    checkpoint_dir = output.parent / "memories-checkpoints" / "automation_first"
    assert json.loads((checkpoint_dir / "memory_state.json").read_text())["next"] == 2
    assert (checkpoint_dir / "memory.md").is_file()
    assert json.loads((checkpoint_dir / "progress.jsonl").read_text())["batch"] == 1


def test_prepare_coco_lora_cli_writes_reusable_data_and_audit(tmp_path: Path, monkeypatch) -> None:
    _write_dataset(tmp_path)

    class StubInferenceBackend:
        def __init__(self, model, **kwargs) -> None:
            assert model == "served-model"
            assert kwargs["base_url"] == "http://localhost:8000/v1"
            assert kwargs["seed"] == 9

    class StubRunner:
        def __init__(self, _backend, **kwargs) -> None:
            assert kwargs["max_images"] == 4

        def run(self, history, **kwargs):
            assert len(history) == 2
            kwargs["progress"](2, 2)
            return SimpleNamespace(predictions=["base-one", "base-two"])

    class StubBuilder:
        def build(self, examples, predictions, **kwargs):
            assert examples == ["adapted"]
            assert predictions == ["method-predictions"]
            assert kwargs["batch_size"] == 2
            assert kwargs["fail_on_unresolved"] is False
            kwargs["progress"](2, 2)
            return (
                ["revised"],
                [
                    {"id": "one", "status": "accepted_direct"},
                    {"id": "two", "status": "unresolved"},
                ],
            )

    monkeypatch.setattr(inference_backends, "OpenAICompatiblePredictor", StubInferenceBackend)
    monkeypatch.setattr(cli_module, "BenchmarkRunner", StubRunner)
    monkeypatch.setattr(cli_module, "coco_lora_builder", lambda *_args: StubBuilder())
    monkeypatch.setattr(cli_module, "to_method_examples", lambda *_args, **_kwargs: ["adapted"])
    monkeypatch.setattr(
        cli_module,
        "to_method_predictions",
        lambda _predictions: ["method-predictions"],
    )

    def stub_write_revised_data(examples, path) -> None:
        assert examples == ["revised"]
        Path(path).write_text("[]\n")

    monkeypatch.setattr(cli_module, "write_revised_data", stub_write_revised_data)

    output_dir = tmp_path / "artifacts" / "coco-lora-prepared"
    exit_code = main(
        [
            "prepare-coco-lora",
            "--data-root",
            str(tmp_path),
            "--split",
            "test",
            "--personas",
            "automation_first",
            "--limit",
            "2",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "served-model",
            "--output-dir",
            str(output_dir),
            "--revision-batch-size",
            "2",
            "--max-images",
            "4",
            "--seed",
            "9",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["personas"]["automation_first"]["file"] == "automation_first.json"
    assert manifest["personas"]["automation_first"]["revision_statuses"] == {
        "accepted_direct": 1,
        "unresolved": 1,
    }
    audit = json.loads((output_dir / "revision-audit.json").read_text())
    assert len(audit["automation_first"]) == 2


def test_train_coco_lora_cli_consumes_prepared_data_without_inference(
    tmp_path: Path, monkeypatch
) -> None:
    prepared_item = SimpleNamespace(example=SimpleNamespace(user_id="automation_first"))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "automation_first.json").write_text("[]\n")
    (prepared_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "osworld-persona-coco-lora-v1",
                "personas": {"automation_first": {"file": "automation_first.json"}},
            }
        )
    )

    class StubTrainingBackend:
        def __init__(self, repo_path, config_file) -> None:
            assert Path(repo_path) == tmp_path / "LlamaFactory"
            assert Path(config_file).name == "qwen3_5_lora.yaml"

        def train(self, revised, config, *, progress=None):
            assert revised == [prepared_item]
            assert config.base_model == "training-model"
            assert config.output_dir.name == "automation_first"
            assert config.num_epochs == 1
            assert config.seed == 9
            assert progress is not None
            progress(1, 1)
            return SimpleNamespace(
                adapter_uri="adapter://smoke",
                backend="llamafactory",
                base_model=config.base_model,
                metadata={"steps": 1},
            )

    monkeypatch.setattr(training_backends, "LlamaFactoryTrainingBackend", StubTrainingBackend)
    monkeypatch.setattr(cli_module, "load_revised_data", lambda _path: [prepared_item])
    output_dir = tmp_path / "trained"
    exit_code = main(
        [
            "train-coco-lora",
            "--prepared-dir",
            str(prepared_dir),
            "--base-model",
            "training-model",
            "--llamafactory-root",
            str(tmp_path / "LlamaFactory"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--seed",
            "9",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    index = json.loads((output_dir / "index.json").read_text())
    assert index["automation_first"]["adapter_uri"] == "adapter://smoke"
    assert index["automation_first"]["prepared_examples"] == 1


def test_train_coco_lora_cli_passes_tinker_credentials(tmp_path: Path, monkeypatch) -> None:
    prepared_item = SimpleNamespace(example=SimpleNamespace(user_id="automation_first"))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "automation_first.json").write_text("[]\n")
    (prepared_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "osworld-persona-coco-lora-v1",
                "personas": {"automation_first": {"file": "automation_first.json"}},
            }
        )
    )

    class StubTinkerBackend:
        def __init__(self, **kwargs) -> None:
            assert kwargs["api_key"] == "tml-key"
            assert kwargs["renderer_name"] == "qwen3_5_disable_thinking"

        def train(self, revised, config, *, progress=None):
            assert revised == [prepared_item]
            assert progress is not None
            progress(1, 1)
            return SimpleNamespace(
                adapter_uri="tinker://run/sampler_weights/final",
                backend="tinker",
                base_model=config.base_model,
                metadata={},
            )

    monkeypatch.setattr(training_backends, "TinkerBackend", StubTinkerBackend)
    monkeypatch.setattr(cli_module, "load_revised_data", lambda _path: [prepared_item])
    output_dir = tmp_path / "trained"

    assert (
        main(
            [
                "train-coco-lora",
                "--training-backend",
                "tinker",
                "--prepared-dir",
                str(prepared_dir),
                "--base-model",
                "Qwen/Qwen3.5-9B",
                "--tinker-api-key",
                "tml-key",
                "--tinker-renderer",
                "qwen3_5_disable_thinking",
                "--output-dir",
                str(output_dir),
                "--no-progress",
            ]
        )
        == 0
    )
    assert json.loads((output_dir / "index.json").read_text())["automation_first"][
        "adapter_uri"
    ].endswith("/sampler_weights/final")


def test_train_coco_lora_cli_resumes_completed_persona(tmp_path: Path, monkeypatch) -> None:
    prepared_item = SimpleNamespace(example=SimpleNamespace(user_id="automation_first"))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared_file = prepared_dir / "automation_first.json"
    prepared_file.write_text("[]\n")
    (prepared_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "osworld-persona-coco-lora-v1",
                "personas": {"automation_first": {"file": prepared_file.name}},
            }
        )
    )
    output_dir = tmp_path / "trained"
    persona_output = output_dir / "automation_first"
    persona_output.mkdir(parents=True)
    (persona_output / "training_artifact.json").write_text("{}\n")
    existing_entry = {
        "adapter_uri": "tinker://run/sampler_weights/final",
        "backend": "tinker",
        "base_model": "Qwen/Qwen3.5-9B",
        "prepared_examples": 1,
        "prepared_data": str(prepared_file.resolve()),
        "metadata": {"steps": 3},
    }
    (output_dir / "index.json").write_text(json.dumps({"automation_first": existing_entry}))

    class StubTinkerBackend:
        def __init__(self, **_kwargs) -> None:
            pass

        def train(self, *_args, **_kwargs):
            raise AssertionError("a completed persona adapter should be reused")

    monkeypatch.setattr(training_backends, "TinkerBackend", StubTinkerBackend)
    monkeypatch.setattr(cli_module, "load_revised_data", lambda _path: [prepared_item])

    assert (
        main(
            [
                "train-coco-lora",
                "--training-backend",
                "tinker",
                "--prepared-dir",
                str(prepared_dir),
                "--base-model",
                "Qwen/Qwen3.5-9B",
                "--output-dir",
                str(output_dir),
                "--existing-output",
                "resume",
                "--no-progress",
            ]
        )
        == 0
    )
    assert json.loads((output_dir / "index.json").read_text())["automation_first"] == existing_entry


def test_train_coco_lora_cli_overwrites_incomplete_persona(tmp_path: Path, monkeypatch) -> None:
    prepared_item = SimpleNamespace(example=SimpleNamespace(user_id="automation_first"))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared_file = prepared_dir / "automation_first.json"
    prepared_file.write_text("[]\n")
    (prepared_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "osworld-persona-coco-lora-v1",
                "personas": {"automation_first": {"file": prepared_file.name}},
            }
        )
    )
    output_dir = tmp_path / "trained"
    persona_output = output_dir / "automation_first"
    persona_output.mkdir(parents=True)
    stale_file = persona_output / "partial-training-output"
    stale_file.write_text("stale")

    class StubTinkerBackend:
        def __init__(self, **_kwargs) -> None:
            pass

        def train(self, revised, config, *, progress=None):
            assert revised == [prepared_item]
            assert not stale_file.exists()
            assert not config.output_dir.exists()
            config.output_dir.mkdir()
            (config.output_dir / "training_artifact.json").write_text("{}\n")
            if progress:
                progress(1, 1)
            return SimpleNamespace(
                adapter_uri="tinker://new-run/sampler_weights/final",
                backend="tinker",
                base_model=config.base_model,
                metadata={"steps": 1},
            )

    monkeypatch.setattr(training_backends, "TinkerBackend", StubTinkerBackend)
    monkeypatch.setattr(cli_module, "load_revised_data", lambda _path: [prepared_item])

    assert (
        main(
            [
                "train-coco-lora",
                "--training-backend",
                "tinker",
                "--prepared-dir",
                str(prepared_dir),
                "--base-model",
                "Qwen/Qwen3.5-9B",
                "--output-dir",
                str(output_dir),
                "--existing-output",
                "overwrite",
                "--no-progress",
            ]
        )
        == 0
    )
    index = json.loads((output_dir / "index.json").read_text())
    assert index["automation_first"]["adapter_uri"] == ("tinker://new-run/sampler_weights/final")


def test_train_coco_lora_cli_overwrites_stale_llamafactory_input(
    tmp_path: Path, monkeypatch
) -> None:
    prepared_item = SimpleNamespace(example=SimpleNamespace(user_id="automation_first"))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared_file = prepared_dir / "automation_first.json"
    prepared_file.write_text("[]\n")
    (prepared_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "osworld-persona-coco-lora-v1",
                "personas": {"automation_first": {"file": prepared_file.name}},
            }
        )
    )
    output_dir = tmp_path / "trained"
    output_dir.mkdir()
    stale_input = output_dir / ".automation_first-coco-lora-input"
    stale_input.mkdir()
    (stale_input / "dataset_info.json").write_text("{}\n")

    class StubTrainingBackend:
        def __init__(self, repo_path, config_file) -> None:
            assert Path(repo_path) == tmp_path / "LlamaFactory"
            assert Path(config_file).name == "qwen3_5_lora.yaml"

        def train(self, revised, config, *, progress=None):
            assert revised == [prepared_item]
            assert not stale_input.exists()
            config.output_dir.mkdir()
            (config.output_dir / "adapter_config.json").write_text("{}\n")
            if progress:
                progress(1, 1)
            return SimpleNamespace(
                adapter_uri=str(config.output_dir),
                backend="llamafactory",
                base_model=config.base_model,
                metadata={"steps": 1},
            )

    monkeypatch.setattr(training_backends, "LlamaFactoryTrainingBackend", StubTrainingBackend)
    monkeypatch.setattr(cli_module, "load_revised_data", lambda _path: [prepared_item])

    assert (
        main(
            [
                "train-coco-lora",
                "--prepared-dir",
                str(prepared_dir),
                "--base-model",
                "Qwen/Qwen3.5-9B",
                "--llamafactory-root",
                str(tmp_path / "LlamaFactory"),
                "--output-dir",
                str(output_dir),
                "--existing-output",
                "overwrite",
                "--no-progress",
            ]
        )
        == 0
    )
    assert not stale_input.exists()
    assert (output_dir / "automation_first" / "adapter_config.json").is_file()


def test_train_coco_lora_cli_rejects_manifest_path_escape(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "osworld-persona-coco-lora-v1",
                "personas": {"automation_first": {"file": "../outside.json"}},
            }
        )
    )

    try:
        main(
            [
                "train-coco-lora",
                "--prepared-dir",
                str(prepared_dir),
                "--base-model",
                "training-model",
                "--llamafactory-root",
                str(tmp_path / "LlamaFactory"),
                "--output-dir",
                str(tmp_path / "trained"),
                "--no-progress",
            ]
        )
    except ValueError as error:
        assert "escapes its directory" in str(error)
    else:
        raise AssertionError("manifest path traversal was accepted")


def test_eval_coco_lora_cli_manages_server_and_runs_matching_condition(
    tmp_path: Path, monkeypatch
) -> None:
    calls = {}

    class StubServer:
        adapter_path = tmp_path / "adapter"
        base_url = "http://127.0.0.1:8123/v1"
        api_key = "local-key"

        def __init__(self, **kwargs):
            calls["server"] = kwargs

        def __enter__(self):
            calls["entered"] = True
            return self

        def __exit__(self, *_args):
            calls["exited"] = True

    def stub_run(args):
        calls["run"] = args
        return 0

    monkeypatch.setattr(inference_backends, "LlamaFactoryServer", StubServer)
    monkeypatch.setattr(cli_module, "_run", stub_run)

    output = tmp_path / "result.json"
    assert (
        main(
            [
                "eval-coco-lora",
                "--data-root",
                str(tmp_path / "data"),
                "--personas",
                "automation_first",
                "--limit",
                "10",
                "--adapter-path",
                str(tmp_path / "adapter"),
                "--llamafactory-root",
                str(tmp_path / "LlamaFactory"),
                "--model",
                "Qwen/base",
                "--port",
                "8123",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert calls["server"]["adapter_path"] == tmp_path / "adapter"
    assert calls["server"]["port"] == 8123
    assert calls["entered"] and calls["exited"]
    assert calls["run"].condition == "coco_lora"
    assert calls["run"].base_url == "http://127.0.0.1:8123/v1"
    assert calls["run"].api_key == "local-key"
    assert calls["run"].limit == 10
