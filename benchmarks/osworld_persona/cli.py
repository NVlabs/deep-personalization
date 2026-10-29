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

"""OSWorld-Persona command-line utilities."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from functools import partial
from pathlib import Path

from tqdm.auto import tqdm

from methods import LoRATrainingConfig, PromptCondition, PromptMemory

from .data import DEFAULT_PERSONAS, load_split, resolve_dataset_root
from .method_adapter import (
    coco_lora_builder,
    coco_pe_trainer,
    induce_persona,
    load_revised_data,
    to_method_examples,
    to_method_predictions,
    write_revised_data,
)
from .prompts import oracle_personas
from .run import BenchmarkRunner
from .scoring import load_jsonl, score_external_predictions


def _add_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    personas_required: bool = False,
) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-root", type=Path)
    source.add_argument("--dataset-id", help="Hugging Face dataset repository ID")
    parser.add_argument("--revision", help="Hugging Face revision")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None if personas_required else list(DEFAULT_PERSONAS),
        required=personas_required,
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Validate annotations without requiring the image files",
    )


def _load(args: argparse.Namespace):
    root = resolve_dataset_root(
        args.data_root,
        dataset_id=args.dataset_id,
        revision=args.revision,
    )
    return root, load_split(
        root,
        split=args.split,
        personas=args.personas,
        validate_images=not args.skip_image_validation,
    )


def _validate(args: argparse.Namespace) -> int:
    root, examples = _load(args)
    labels = Counter(example.want_support for example in examples)
    sessions = {(example.persona, example.session_id) for example in examples}
    by_persona = {
        persona: {
            "examples": sum(example.persona == persona for example in examples),
            "support_yes": sum(
                example.persona == persona and example.want_support == "yes" for example in examples
            ),
            "sessions": len(
                {example.session_id for example in examples if example.persona == persona}
            ),
        }
        for persona in args.personas
    }
    print(
        json.dumps(
            {
                "data_root": str(root),
                "split": args.split,
                "examples": len(examples),
                "sessions": len(sessions),
                "labels": dict(sorted(labels.items())),
                "multi_user_intent": sum(
                    isinstance(example.user_intent, tuple) for example in examples
                ),
                "by_persona": by_persona,
                "images_validated": not args.skip_image_validation,
            },
            indent=2,
        )
    )
    return 0


def _export(args: argparse.Namespace) -> int:
    _root, examples = _load(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for example in examples:
            destination.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} canonical examples to {args.output}")
    return 0


def _score(args: argparse.Namespace) -> int:
    _root, examples = _load(args)
    result = score_external_predictions(
        examples,
        load_jsonl(args.predictions),
        method=args.method,
        allow_missing=args.allow_missing,
    )
    result.write(args.output)
    print(json.dumps(result.metrics.to_dict(), indent=2))
    print(f"Wrote results to {args.output}")
    return 0


def _load_string_mapping(path: Path, kind: str) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path}: expected a non-empty JSON object mapping personas to {kind}")
    if not all(
        isinstance(key, str) and isinstance(item, str) and item.strip()
        for key, item in value.items()
    ):
        raise ValueError(f"{path}: every persona and {kind} must be a non-empty string")
    return value


def _prompt_condition(args: argparse.Namespace) -> PromptCondition:
    if args.condition == "vanilla":
        return PromptCondition.vanilla()
    if args.condition == "oracle_persona":
        return PromptCondition.oracle_persona(oracle_personas())
    if args.condition == "coco_lora":
        return PromptCondition.coco_lora()
    if args.condition_file is None:
        raise ValueError(f"--condition-file is required for {args.condition}")
    values = _load_string_mapping(
        args.condition_file,
        "profiles" if args.condition == "induced_persona" else "memories",
    )
    if args.condition == "induced_persona":
        return PromptCondition.induced_persona(values)
    return PromptCondition.coco_pe(values)


def _set_progress(bar, completed: int) -> None:
    """Advance a tqdm bar to an absolute completed-item count."""
    bar.update(max(0, completed - bar.n))


def _set_offset_progress(bar, offset: int, completed: int, _total: int) -> None:
    _set_progress(bar, offset + completed)


def _resumable_adapter_entry(
    index: dict,
    persona: str,
    *,
    backend: str,
    base_model: str,
    prepared_file: Path,
    prepared_examples: int,
    output_dir: Path,
) -> dict | None:
    """Return a completed, compatible adapter entry or report it as incomplete."""
    entry = index.get(persona)
    if not isinstance(entry, dict):
        return None
    marker = (
        output_dir / "training_artifact.json"
        if backend == "tinker"
        else output_dir / "adapter_config.json"
    )
    if not marker.is_file():
        return None
    if entry.get("backend") != backend or entry.get("base_model") != base_model:
        return None
    if not isinstance(entry.get("adapter_uri"), str) or not entry["adapter_uri"]:
        return None
    recorded_data = entry.get("prepared_data")
    if not isinstance(recorded_data, str) or Path(recorded_data).resolve() != prepared_file:
        return None
    if entry.get("prepared_examples") != prepared_examples:
        return None
    return entry


def _existing_output_action(path: Path, policy: str, *, resumable: bool) -> str:
    """Let the caller explicitly reuse or replace an existing persona output."""
    guidance = (
        "Use --existing-output resume to reuse a completed adapter, "
        "--existing-output overwrite to delete it and retrain, or choose a new --output-dir."
        if resumable
        else "This output is incomplete. Use --existing-output overwrite to delete it and "
        "retrain, or choose a new --output-dir."
    )
    if policy == "error":
        raise FileExistsError(f"Training output already exists: {path}. {guidance}")
    if policy == "resume":
        if resumable:
            return "resume"
        raise FileExistsError(f"Cannot resume incomplete training output: {path}. {guidance}")
    if policy == "overwrite":
        return "overwrite"
    if not sys.stdin.isatty():
        raise FileExistsError(
            f"Training output already exists and input is non-interactive: {path}. {guidance}"
        )

    choices = "resume, delete, exit" if resumable else "delete, exit"
    while True:
        answer = input(f"Training output {path} already exists. Choose [{choices}]: ").strip()
        normalized = answer.lower().replace("_", "-").replace(" ", "-")
        if normalized == "resume" and resumable:
            return "resume"
        if normalized in {"delete", "overwrite", "start-over"}:
            return "overwrite"
        if normalized in {"exit", "quit", "abort"}:
            print("Training cancelled; choose a different --output-dir to keep both runs.")
            raise SystemExit(0)
        print(f"Please choose one of: {choices}.")


def _delete_training_output(path: Path, output_root: Path) -> None:
    """Delete one explicitly selected persona output without following symlinks."""
    if path.parent != output_root:
        raise ValueError(f"Refusing to delete output outside {output_root}: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _inference_backend(args: argparse.Namespace):
    """Build the selected runtime without leaking it into benchmark code."""
    from inference_backends import OpenAICompatiblePredictor, TinkerPredictor

    common = {
        "temperature": getattr(args, "temperature", 0.0),
        "max_tokens": getattr(args, "max_tokens", 1024),
        "max_concurrency": args.concurrency,
        "seed": getattr(args, "seed", None),
    }
    if args.backend == "tinker":
        return TinkerPredictor(
            args.model,
            api_key=args.api_key,
            base_url=args.tinker_base_url,
            renderer_name=args.tinker_renderer,
            **common,
        )
    if args.base_url is None:
        raise ValueError("--base-url is required for the OpenAI-compatible backend")
    return OpenAICompatiblePredictor(
        args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
        **common,
    )


def _run(args: argparse.Namespace) -> int:
    _root, examples = _load(args)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        examples = examples[: args.limit]
    predictor = _inference_backend(args)
    condition = _prompt_condition(args)
    with tqdm(
        total=len(examples),
        desc=f"Evaluate {condition.name}",
        unit="example",
        disable=args.no_progress,
    ) as bar:
        result = BenchmarkRunner(
            predictor,
            batch_size=args.batch_size,
            max_images=args.max_images,
            session_context=args.session_context,
        ).run(
            examples,
            condition=condition,
            run_name=args.run_name,
            metadata={
                "backend": args.backend,
                "base_url": (args.tinker_base_url if args.backend == "tinker" else args.base_url),
                "model": args.model,
            },
            progress=lambda completed, _total: _set_progress(bar, completed),
        )
    result.write(args.output)
    print(json.dumps(result.metrics.to_dict(), indent=2))
    print(f"Wrote results to {args.output}")
    return 0


def _eval_coco_lora(args: argparse.Namespace) -> int:
    """Serve one local adapter, evaluate it, and release the server resources."""
    from inference_backends import LlamaFactoryServer

    if len(args.personas) != 1:
        raise ValueError("eval-coco-lora requires exactly one persona and its matching adapter")
    server = LlamaFactoryServer(
        repo_path=args.llamafactory_root,
        config_file=args.inference_config,
        base_model=args.model,
        adapter_path=args.adapter_path,
        template=args.template,
        infer_backend=args.infer_backend,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        startup_timeout=args.startup_timeout,
    )
    print(f"Starting LLaMA-Factory for {args.personas[0]} from {server.adapter_path}")
    with server:
        print(f"LLaMA-Factory is ready at {server.base_url}")
        args.base_url = server.base_url
        args.api_key = server.api_key
        return _run(args)


def _induce_persona(args: argparse.Namespace) -> int:
    """Learn one profile per selected persona and write a CLI-ready mapping."""
    _root, examples = _load(args)
    generator = _inference_backend(args)
    profiles: dict[str, str] = {}
    audit: dict[str, dict] = {}
    with tqdm(
        total=2 * len(args.personas),
        desc="Induce personas",
        unit="stage",
        disable=args.no_progress,
    ) as bar:
        for persona_index, persona in enumerate(args.personas):
            history = [example for example in examples if example.persona == persona]
            if args.limit is not None:
                if args.limit <= 0:
                    raise ValueError("--limit must be positive")
                history = history[: args.limit]
            if not history:
                raise ValueError(f"No {args.split} examples found for persona {persona!r}")
            result = induce_persona(
                history,
                generator,
                max_per_label=args.max_per_label,
                seed=args.seed,
                progress=lambda completed, _total, offset=2 * persona_index: _set_progress(
                    bar, offset + completed
                ),
            )
            profiles[persona] = result.profile
            audit[persona] = {
                "limitations": list(result.limitations),
                "selected_ids": list(result.selected_ids),
                "digestion": result.digestion,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote induced persona mapping to {args.output}")
    return 0


def _evolve_coco_pe(args: argparse.Namespace) -> int:
    """Learn one COCO-PE prompt memory per selected persona."""
    _root, examples = _load(args)
    backend = _inference_backend(args)
    memories: dict[str, str] = {}
    audits: dict[str, list[dict]] = {}
    for persona in args.personas:
        history = [example for example in examples if example.persona == persona]
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            history = history[: args.limit]
        if not history:
            raise ValueError(f"No {args.split} examples found for persona {persona!r}")

        with tqdm(
            total=len(history) * (args.epochs + 1),
            desc=f"COCO-PE {persona}",
            unit="example",
            disable=args.no_progress,
        ) as bar:
            base_predictions = (
                BenchmarkRunner(
                    backend,
                    batch_size=args.inference_batch_size,
                    max_images=args.max_images,
                )
                .run(
                    history,
                    run_name=f"coco-pe-base-{persona}",
                    progress=lambda completed, _total: _set_progress(bar, completed),
                )
                .predictions
            )
            trainer = coco_pe_trainer(
                backend,
                backend,
                batch_size=args.evolution_batch_size,
                max_bullets=args.max_bullets,
                max_operations=args.max_operations,
                reflect_correct=args.reflect_correct,
                reflection_max_tokens=args.reflection_max_tokens,
                curator_max_tokens=args.curator_max_tokens,
                seed=args.seed,
            )
            evolution_offset = len(history)
            checkpoint_root = args.checkpoint_dir or (
                args.output.parent / f"{args.output.stem}-checkpoints"
            )
            checkpoint_dir = checkpoint_root / persona
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_path = checkpoint_dir / "memory_state.json"
            markdown_path = checkpoint_dir / "memory.md"
            progress_path = checkpoint_dir / "progress.jsonl"
            initial_memory = (
                PromptMemory.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
                if args.resume and state_path.exists()
                else PromptMemory()
            )
            with progress_path.open("a" if args.resume else "w", encoding="utf-8") as log:

                def checkpoint(
                    current_memory,
                    record,
                    progress_log=log,
                    state_destination=state_path,
                    markdown_destination=markdown_path,
                ) -> None:
                    progress_log.write(json.dumps(record) + "\n")
                    progress_log.flush()
                    state_destination.write_text(
                        json.dumps(current_memory.to_dict(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    markdown_destination.write_text(
                        current_memory.render(with_ids=False) + "\n",
                        encoding="utf-8",
                    )

                memory, audit = trainer.fit(
                    to_method_examples(history, base_predictions, max_images=args.max_images),
                    epochs=args.epochs,
                    shuffle=not args.no_shuffle,
                    memory=initial_memory,
                    progress=partial(_set_offset_progress, bar, evolution_offset),
                    checkpoint=checkpoint,
                )
        memories[persona] = memory.render(with_ids=False)
        audits[persona] = audit
        print(
            f"Evolved COCO-PE memory for {persona} from {len(history)} examples "
            f"({len(memory.bullets)} bullets)"
        )
        if sum(item.get("n_reflection_targets", 0) for item in audit) and not sum(
            item.get("n_reflections", 0) for item in audit
        ):
            print(
                f"WARNING: {persona}: the reflector returned no valid reflections; "
                f"inspect the server's structured-output behavior and {progress_path}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(memories, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audits, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote COCO-PE memory mapping to {args.output}")
    return 0


def _prepare_coco_lora(args: argparse.Namespace) -> int:
    """Use an inference backend to build reusable revised COCO-LoRA data."""
    data_root, examples = _load(args)
    inference_backend = _inference_backend(args)
    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to mix with an existing prepared-data directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    audit_path = output_root / "revision-audit.json"
    manifest = {
        "format": "osworld-persona-coco-lora-v1",
        "data_root": str(data_root),
        "split": args.split,
        "served_model": args.model,
        "personas": {},
    }
    audits: dict[str, list[dict]] = {}

    def write_artifacts() -> None:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        audit_path.write_text(
            json.dumps(audits, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for persona in args.personas:
        history = [example for example in examples if example.persona == persona]
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            history = history[: args.limit]
        if not history:
            raise ValueError(f"No {args.split} examples found for persona {persona!r}")

        with tqdm(
            total=2 * len(history),
            desc=f"COCO-LoRA prepare {persona}",
            unit="example",
            disable=args.no_progress,
        ) as bar:
            base_predictions = (
                BenchmarkRunner(
                    inference_backend,
                    batch_size=args.inference_batch_size,
                    max_images=args.max_images,
                )
                .run(
                    history,
                    run_name=f"coco-lora-base-{persona}",
                    progress=lambda completed, _total: _set_progress(bar, completed),
                )
                .predictions
            )
            revised, audit = coco_lora_builder(inference_backend).build(
                to_method_examples(history, base_predictions, max_images=args.max_images),
                to_method_predictions(base_predictions),
                batch_size=args.revision_batch_size,
                fail_on_unresolved=args.fail_on_unresolved,
                progress=partial(_set_offset_progress, bar, len(history)),
            )
        if not revised:
            raise RuntimeError(f"COCO-LoRA produced no accepted examples for {persona}")

        data_file = f"{persona}.json"
        write_revised_data(revised, output_root / data_file)
        status_counts = Counter(item["status"] for item in audit)
        manifest["personas"][persona] = {
            "file": data_file,
            "source_examples": len(history),
            "revised_examples": len(revised),
            "revision_statuses": dict(sorted(status_counts.items())),
        }
        audits[persona] = audit
        write_artifacts()
        print(
            f"Prepared COCO-LoRA data for {persona}: "
            f"{len(revised)}/{len(history)} accepted examples"
        )

    print(f"Wrote COCO-LoRA manifest to {manifest_path}")
    print(f"Wrote COCO-LoRA revision audit to {audit_path}")
    return 0


def _train_coco_lora(args: argparse.Namespace) -> int:
    """Train COCO-LoRA adapters from prepared data without inference calls."""
    from training_backends import LlamaFactoryTrainingBackend, TinkerBackend

    prepared_root = args.prepared_dir.expanduser().resolve()
    manifest_path = prepared_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared COCO-LoRA manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != "osworld-persona-coco-lora-v1":
        raise ValueError(f"Invalid COCO-LoRA manifest: {manifest_path}")
    prepared_personas = manifest.get("personas")
    if not isinstance(prepared_personas, dict) or not prepared_personas:
        raise ValueError(f"COCO-LoRA manifest contains no personas: {manifest_path}")
    personas = args.personas or list(prepared_personas)
    missing = [persona for persona in personas if persona not in prepared_personas]
    if missing:
        raise ValueError(f"Prepared data is missing personas: {', '.join(missing)}")

    if args.training_backend == "llamafactory":
        if args.llamafactory_root is None:
            raise ValueError("--llamafactory-root is required for the LLaMA-Factory backend")
        training_backend = LlamaFactoryTrainingBackend(
            repo_path=args.llamafactory_root,
            config_file=args.training_config,
        )
    else:
        training_backend = TinkerBackend(
            renderer_name=args.tinker_renderer,
            base_url=args.tinker_base_url,
            api_key=args.tinker_api_key,
            checkpoint_ttl_seconds=args.tinker_checkpoint_ttl,
        )

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output or output_root / "index.json"
    if index_path.is_file():
        existing_index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(existing_index, dict):
            raise ValueError(f"Invalid COCO-LoRA adapter index: {index_path}")
        index: dict[str, dict] = existing_index
    else:
        index = {}
    for persona in personas:
        entry = prepared_personas[persona]
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError(f"Invalid prepared-data entry for {persona!r}")
        prepared_file = (prepared_root / entry["file"]).resolve()
        try:
            prepared_file.relative_to(prepared_root)
        except ValueError as error:
            raise ValueError(
                f"Prepared-data file for {persona!r} escapes its directory: {entry['file']!r}"
            ) from error
        revised = load_revised_data(prepared_file)
        if not revised or {item.example.user_id for item in revised} != {persona}:
            raise ValueError(f"Prepared file for {persona!r} is empty or contains another persona")
        persona_output = (output_root / persona).resolve()
        if persona_output.parent != output_root:
            raise ValueError(f"Invalid persona output path for {persona!r}: {persona_output}")
        related_outputs = [persona_output]
        if args.training_backend == "llamafactory":
            related_outputs.append(output_root / f".{persona_output.name}-coco-lora-input")
        existing_outputs = [path for path in related_outputs if path.exists() or path.is_symlink()]
        if existing_outputs:
            completed_entry = _resumable_adapter_entry(
                index,
                persona,
                backend=args.training_backend,
                base_model=args.base_model,
                prepared_file=prepared_file,
                prepared_examples=len(revised),
                output_dir=persona_output,
            )
            action = _existing_output_action(
                existing_outputs[0],
                args.existing_output,
                resumable=completed_entry is not None,
            )
            if action == "resume":
                print(
                    f"Reusing completed COCO-LoRA adapter for {persona}: "
                    f"{completed_entry['adapter_uri']}"
                )
                continue
            for existing_output in existing_outputs:
                _delete_training_output(existing_output, output_root)
            index.pop(persona, None)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                json.dumps(index, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        training_config = LoRATrainingConfig(
            base_model=args.base_model,
            output_dir=persona_output,
            run_name=f"{args.run_name_prefix}-{persona}",
            rank=args.rank,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.training_batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )
        progress_total = (
            args.epochs
            * ((len(revised) + args.training_batch_size - 1) // args.training_batch_size)
            if args.training_backend == "tinker"
            else None
        )
        with tqdm(
            total=progress_total,
            desc=f"COCO-LoRA train {persona}",
            unit="step",
            disable=args.no_progress,
        ) as bar:

            def update_training_progress(completed: int, total: int) -> None:
                if bar.total != total:
                    bar.total = total
                _set_progress(bar, completed)

            adapter = training_backend.train(
                revised,
                training_config,
                progress=update_training_progress,
            )
        index[persona] = {
            "adapter_uri": adapter.adapter_uri,
            "backend": adapter.backend,
            "base_model": adapter.base_model,
            "prepared_examples": len(revised),
            "prepared_data": str(prepared_file),
            "metadata": adapter.metadata,
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Trained COCO-LoRA adapter for {persona}: {adapter.adapter_uri}")

    print(f"Wrote COCO-LoRA adapter index to {index_path}")
    return 0


def _add_inference_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("openai-compatible", "tinker"),
        default="openai-compatible",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API root including /v1",
    )
    parser.add_argument(
        "--api-key",
        help="API key; defaults to the backend's environment-based credentials",
    )
    parser.add_argument(
        "--tinker-base-url",
        help="custom native Tinker service URL",
    )
    parser.add_argument(
        "--tinker-renderer",
        help="native Tinker renderer (default: recommended for the base model)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osworld-persona", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate data and print statistics")
    _add_dataset_arguments(validate)
    validate.set_defaults(func=_validate)

    export = subparsers.add_parser("export", help="export canonical framework-neutral JSONL")
    _add_dataset_arguments(export)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(func=_export)

    score = subparsers.add_parser("score", help="score externally produced prediction JSONL")
    _add_dataset_arguments(score)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--method", default="external")
    score.add_argument(
        "--allow-missing",
        action="store_true",
        help="represent missing IDs as parse failures instead of rejecting the file",
    )
    score.set_defaults(func=_score)

    run = subparsers.add_parser("run", help="run the benchmark through an inference backend")
    _add_dataset_arguments(run)
    run.add_argument(
        "--condition",
        choices=("vanilla", "induced_persona", "oracle_persona", "coco_pe", "coco_lora"),
        default="vanilla",
    )
    run.add_argument(
        "--condition-file",
        type=Path,
        help="JSON persona-to-profile or persona-to-memory mapping",
    )
    _add_inference_backend_arguments(run)
    run.add_argument("--model", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-name", default="")
    run.add_argument("--limit", type=int)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-images", type=int, default=8)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    run.add_argument(
        "--session-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="carry model observations forward within each session (default: enabled)",
    )
    run.set_defaults(func=_run)

    eval_lora = subparsers.add_parser(
        "eval-coco-lora",
        help="serve one local COCO-LoRA adapter, evaluate it, then stop the server",
    )
    _add_dataset_arguments(eval_lora, personas_required=True)
    eval_lora.set_defaults(
        func=_eval_coco_lora,
        split="test",
        condition="coco_lora",
        condition_file=None,
        backend="llamafactory",
        base_url=None,
        run_name="coco-lora",
    )
    eval_lora.add_argument("--adapter-path", type=Path, required=True)
    eval_lora.add_argument("--llamafactory-root", type=Path, required=True)
    eval_lora.add_argument(
        "--inference-config",
        type=Path,
        default=Path("examples/inference/qwen3vl.yaml"),
        help="LLaMA-Factory inference YAML, relative to its checkout by default",
    )
    eval_lora.add_argument("--model", required=True, help="base-model checkpoint identifier")
    eval_lora.add_argument("--template", default="qwen3_5_nothink")
    eval_lora.add_argument("--infer-backend", default="huggingface")
    eval_lora.add_argument("--host", default="127.0.0.1")
    eval_lora.add_argument("--port", type=int, default=8000)
    eval_lora.add_argument("--api-key")
    eval_lora.add_argument("--startup-timeout", type=float, default=600.0)
    eval_lora.add_argument("--output", type=Path, required=True)
    eval_lora.add_argument("--limit", type=int)
    eval_lora.add_argument("--batch-size", type=int, default=32)
    eval_lora.add_argument("--max-images", type=int, default=8)
    eval_lora.add_argument("--max-tokens", type=int, default=1024)
    eval_lora.add_argument("--temperature", type=float, default=0.0)
    eval_lora.add_argument("--timeout", type=float, default=600.0)
    eval_lora.add_argument("--max-retries", type=int, default=2)
    eval_lora.add_argument("--concurrency", type=int, default=1)
    eval_lora.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    eval_lora.add_argument(
        "--session-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="carry model observations forward within each session (default: enabled)",
    )

    induce = subparsers.add_parser(
        "induce-persona",
        help="learn persona profiles from a labeled split",
    )
    _add_dataset_arguments(induce)
    induce.set_defaults(func=_induce_persona, split="train")
    _add_inference_backend_arguments(induce)
    induce.add_argument("--model", required=True)
    induce.add_argument("--output", type=Path, required=True)
    induce.add_argument(
        "--audit-output",
        type=Path,
        help="optional JSON containing selected IDs, limitations, and digestion",
    )
    induce.add_argument("--max-per-label", type=int, default=60)
    induce.add_argument(
        "--limit",
        type=int,
        help="use at most this many examples per persona for a smoke test",
    )
    induce.add_argument("--seed", type=int, default=42)
    induce.add_argument("--timeout", type=float, default=600.0)
    induce.add_argument("--max-retries", type=int, default=2)
    induce.add_argument("--concurrency", type=int, default=1)
    induce.add_argument("--no-progress", action="store_true", help="disable the progress bar")

    coco_pe = subparsers.add_parser(
        "evolve-coco-pe",
        help="learn COCO-PE prompt memories from a labeled split",
    )
    _add_dataset_arguments(coco_pe)
    coco_pe.set_defaults(func=_evolve_coco_pe, split="train")
    _add_inference_backend_arguments(coco_pe)
    coco_pe.add_argument("--model", required=True)
    coco_pe.add_argument("--output", type=Path, required=True)
    coco_pe.add_argument("--audit-output", type=Path, help="optional JSON evolution trace")
    coco_pe.add_argument(
        "--limit",
        type=int,
        help="use at most this many examples per persona for a smoke test",
    )
    coco_pe.add_argument("--epochs", type=int, default=1)
    coco_pe.add_argument(
        "--no-shuffle",
        action="store_true",
        help="preserve source order (the original implementation shuffles by default)",
    )
    coco_pe.add_argument("--evolution-batch-size", type=int, default=8)
    coco_pe.add_argument("--inference-batch-size", type=int, default=32)
    coco_pe.add_argument("--max-images", type=int, default=8)
    coco_pe.add_argument("--max-bullets", type=int, default=50)
    coco_pe.add_argument("--max-operations", type=int, default=8)
    coco_pe.add_argument("--reflect-correct", type=int, default=2)
    coco_pe.add_argument("--seed", type=int, default=42)
    coco_pe.add_argument("--max-tokens", type=int, default=4096)
    coco_pe.add_argument("--reflection-max-tokens", type=int, default=8192)
    coco_pe.add_argument("--curator-max-tokens", type=int, default=4096)
    coco_pe.add_argument("--temperature", type=float, default=0.0)
    coco_pe.add_argument("--timeout", type=float, default=600.0)
    coco_pe.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="retries after the initial API attempt (four attempts total by default)",
    )
    coco_pe.add_argument("--concurrency", type=int, default=8)
    coco_pe.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="per-persona state directory (default: <output-stem>-checkpoints beside output)",
    )
    coco_pe.add_argument(
        "--resume",
        action="store_true",
        help="start from existing per-persona memory_state.json files and append progress logs",
    )
    coco_pe.add_argument("--no-progress", action="store_true", help="disable the progress bar")

    prepare_lora = subparsers.add_parser(
        "prepare-coco-lora",
        help="generate and save revised COCO-LoRA supervision",
    )
    _add_dataset_arguments(prepare_lora)
    prepare_lora.set_defaults(func=_prepare_coco_lora, split="train")
    _add_inference_backend_arguments(prepare_lora)
    prepare_lora.add_argument("--model", required=True, help="served base-model name")
    prepare_lora.add_argument("--output-dir", type=Path, required=True)
    prepare_lora.add_argument(
        "--limit",
        type=int,
        help="use at most this many examples per persona for a smoke test",
    )
    prepare_lora.add_argument("--inference-batch-size", type=int, default=32)
    prepare_lora.add_argument("--revision-batch-size", type=int, default=32)
    prepare_lora.add_argument("--max-images", type=int, default=8)
    prepare_lora.add_argument("--max-tokens", type=int, default=1024)
    prepare_lora.add_argument("--temperature", type=float, default=0.0)
    prepare_lora.add_argument("--timeout", type=float, default=600.0)
    prepare_lora.add_argument("--max-retries", type=int, default=2)
    prepare_lora.add_argument("--concurrency", type=int, default=8)
    prepare_lora.add_argument("--seed", type=int, default=42)
    prepare_lora.add_argument(
        "--fail-on-unresolved",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="abort if any requested revision remains invalid",
    )
    prepare_lora.add_argument("--no-progress", action="store_true", help="disable the progress bar")

    train_lora = subparsers.add_parser(
        "train-coco-lora",
        help="train COCO-LoRA adapters from prepared supervision without inference",
    )
    train_lora.set_defaults(func=_train_coco_lora)
    train_lora.add_argument("--prepared-dir", type=Path, required=True)
    train_lora.add_argument(
        "--personas",
        nargs="+",
        help="train only these prepared personas (default: every persona in the manifest)",
    )
    train_lora.add_argument("--base-model", required=True, help="training checkpoint identifier")
    train_lora.add_argument(
        "--training-backend",
        choices=("llamafactory", "tinker"),
        default="llamafactory",
    )
    train_lora.add_argument("--llamafactory-root", type=Path)
    train_lora.add_argument(
        "--training-config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "training_backends"
            / "configs"
            / "qwen3_5_lora.yaml"
        ),
        help="LLaMA-Factory YAML (default: bundled Qwen3.5 LoRA config)",
    )
    train_lora.add_argument("--tinker-base-url")
    train_lora.add_argument(
        "--tinker-api-key",
        help="defaults to TINKER_API_KEY or configured Tinker credentials",
    )
    train_lora.add_argument("--tinker-renderer")
    train_lora.add_argument("--tinker-checkpoint-ttl", type=int)
    train_lora.add_argument("--output-dir", type=Path, required=True)
    train_lora.add_argument("--output", type=Path, help="adapter index JSON")
    train_lora.add_argument(
        "--existing-output",
        choices=("ask", "resume", "overwrite", "error"),
        default="ask",
        help=(
            "what to do with an existing persona output: ask interactively, reuse a "
            "completed adapter, delete and retrain it, or fail (default: ask)"
        ),
    )
    train_lora.add_argument("--rank", type=int, default=32)
    train_lora.add_argument("--learning-rate", type=float, default=1e-5)
    train_lora.add_argument("--epochs", type=int, default=3)
    train_lora.add_argument("--training-batch-size", type=int, default=1)
    train_lora.add_argument("--max-seq-length", type=int, default=4096)
    train_lora.add_argument("--run-name-prefix", default="coco-lora")
    train_lora.add_argument("--seed", type=int, default=42)
    train_lora.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
