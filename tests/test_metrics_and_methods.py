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

"""Metrics and cross-benchmark method tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from benchmarks.common import GenerationRequest
from benchmarks.osworld_persona.method_adapter import (
    coco_lora_builder,
    coco_lora_trainer,
    coco_pe_trainer,
    load_revised_data,
    render_induction_context,
    to_induction_records,
    to_method_examples,
    to_method_predictions,
    write_revised_data,
)
from benchmarks.osworld_persona.method_adapter import (
    induce_persona as induce_osworld_persona,
)
from benchmarks.osworld_persona.metrics import compute_metrics, parse_decision
from benchmarks.osworld_persona.prompts import (
    COCO_LORA_REFLECTION_BLOCK,
    COCO_PE_CURATOR_SYSTEM,
    COCO_PE_CURATOR_TEMPLATE,
    COCO_PE_REFLECTOR_SYSTEM,
    COCO_PE_REFLECTOR_TEMPLATE,
    INDUCED_PROFILE_BLOCK,
    INDUCTION_DIGESTION_SYSTEM,
    INDUCTION_DIGESTION_USER,
    INDUCTION_PROFILE_SYSTEM,
    INDUCTION_PROFILE_USER,
    LEARNED_MEMORY_BLOCK,
    ORACLE_PROFILE_BLOCK,
    accumulated_memory,
    apply_condition,
    observer_prompt,
    render_user_prompt,
)
from benchmarks.osworld_persona.schema import Annotation, PredictionRecord
from methods import (
    InductionRecord,
    LoRATrainingConfig,
    MethodExample,
    PromptCondition,
    PromptMemory,
    RevisedTrainingExample,
    TrainedAdapter,
)
from methods.induced_persona import render_personal_context
from training_backends import LlamaFactoryTrainingBackend, TinkerBackend, to_tinker_conversation


def _annotation(index: int, label: str = "yes") -> Annotation:
    return Annotation(
        id=f"automation_first:train:{index:06d}",
        persona="automation_first",
        split="train",
        index=index,
        session_id="001",
        image_paths=(Path(f"/{index}.png"),),
        timestamps=("00:00",),
        observation="visible observation",
        user_intent="visible intent",
        want_support=label,
        rationale="private rationale",
    )


def _record(example: Annotation, raw: str) -> PredictionRecord:
    return PredictionRecord(
        id=example.id,
        persona=example.persona,
        split=example.split,
        index=example.index,
        session_id=example.session_id,
        ground_truth=example.want_support,
        raw_prediction=raw,
        decision=parse_decision(raw),
        previous_recent_memory="prior state",
    )


def test_metrics_report_parse_coverage_and_strict_accuracy() -> None:
    yes = _annotation(0, "yes")
    no = _annotation(1, "no")
    records = [_record(yes, '{"need_support":"yes"}'), _record(no, "not json")]
    metrics = compute_metrics(records).overall
    assert metrics.accuracy == 1.0
    assert metrics.strict_accuracy == 0.5
    assert metrics.parse_coverage == 0.5
    assert metrics.parse_failures == 1


def test_personal_context_excludes_leaky_fields() -> None:
    context = render_personal_context(to_induction_records([_annotation(0)]))
    assert "visible observation" in context
    assert "visible intent" in context
    assert "feedback" in context
    assert "private rationale" not in context
    assert "automation_first" not in context


def test_prompt_conditions_select_per_persona_context() -> None:
    oracle = PromptCondition.oracle_persona({"automation_first": "automate chores"})
    evolved = PromptCondition.coco_pe({"automation_first": "offer scripts"})
    assert oracle.profile_for("automation_first") == "automate chores"
    assert evolved.memory_for("automation_first") == "offer scripts"


def test_prompt_memory_applies_delta_operations_and_refines() -> None:
    memory = PromptMemory()
    assert (
        memory.apply(
            [
                {"op": "add", "section": "when_to_support", "content": "Help on errors."},
                {"op": "add", "section": "general", "content": "Prefer drafts."},
            ],
            5,
        )
        == 2
    )
    memory.vote(["m-001"], ["m-002"])
    assert memory.refine(1) == 1
    assert "Help on errors" in memory.render(with_ids=False)
    assert "Prefer drafts" not in memory.render(with_ids=False)


def test_prompt_memory_tie_break_matches_original_grow_and_refine() -> None:
    memory = PromptMemory()
    memory.apply(
        [
            {"op": "add", "section": "general", "content": "Older rule."},
            {"op": "add", "section": "general", "content": "Newer rule."},
        ],
        2,
    )

    assert memory.refine(1) == 1
    assert list(memory.bullets) == ["m-001"]


def test_coco_pe_prompts_remain_verbatim_from_original_experiment() -> None:
    expected_sha256 = (
        "2277ff29eaac8d2ade89e7c1018db1699442e76268038a247a6e570156575fba",
        "46642f8e75045c960682093d71fac3843d7fd8b9a969ce964fecc90e7df5a4ca",
        "c7b6972321a8d8e79e66f988d55b4c9fe166f24ec2cf2d81e91277fe900ada30",
        "06db4def4575e406905af2927b8150370a24a682d723488a1d3e8d300cf5499a",
    )
    prompts = (
        COCO_PE_REFLECTOR_SYSTEM,
        COCO_PE_REFLECTOR_TEMPLATE,
        COCO_PE_CURATOR_SYSTEM,
        COCO_PE_CURATOR_TEMPLATE,
    )

    assert tuple(hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts) == (
        expected_sha256
    )


def test_other_prompts_remain_verbatim_from_original_experiment() -> None:
    expected_sha256 = (
        "17840ffb93eae3dd25662db5844845a646864627790321150f8fa98eec776465",
        "60646e1616dfdc1bf7f0fe5829248909b368ff1fbc8355cbfbc90cdc5b12cb30",
        "5535aa994ea3797d0672a0732d2d57ddacc7ed81a95e08aa69096bd452249c44",
        "99d58e373dc4a7e54720cb3a4fe8bca67ac5558fe3e18b45ed705e5d127d195f",
        "ee40ce354decccc53a136adcc54d1776f39921d8e01e92de5684dc1a8fd1467f",
        "7418d7e3b74329d5e7974a55bf8b93cc41c7c1687ca2bcedda386c79aa8507fd",
        "25c5e88b0789201fe143e2fa12b8626b9a0adfac962318228c83c62877f0d4d7",
        "c091cd9a94a212ba081ae12f9eff4fa8cab01cd7a5f61dbaaddc42017206ab16",
        "26d1e81899ac64a22e671db46f55315a2e8bd3555d6a5dcfe5d0dbf609257fef",
    )
    prompts = (
        ORACLE_PROFILE_BLOCK,
        INDUCED_PROFILE_BLOCK,
        LEARNED_MEMORY_BLOCK,
        INDUCTION_DIGESTION_SYSTEM,
        INDUCTION_DIGESTION_USER,
        INDUCTION_PROFILE_SYSTEM,
        INDUCTION_PROFILE_USER,
        COCO_LORA_REFLECTION_BLOCK,
        observer_prompt(),
    )

    assert tuple(hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts) == (
        expected_sha256
    )

    assets = Path(__file__).parents[1] / "benchmarks" / "osworld_persona" / "assets"
    assert hashlib.sha256((assets / "observer_prompt.txt").read_bytes()).hexdigest() == (
        "b3c88346d6f5deb2ff9d0e9538db4fd405ad4c6ee851fd0e47a8c5dcc51f1136"
    )
    assert hashlib.sha256((assets / "oracle_personas.json").read_bytes()).hexdigest() == (
        "6dac5d365a354c72f6ca134c9250bcb6acac60a6917448cdd4892d540fa66257"
    )


def test_prompt_rendering_matches_original_experiment_layout() -> None:
    user_prompt = render_user_prompt(("00:00", "00:10"), "prior<image>state")
    assert user_prompt == (
        "<previous_recent_memory>\nprior<image_literal>state\n"
        "</previous_recent_memory>\n\n<screenshots>\n"
        "[t=00:00] Screenshot 1 of 2\n"
        "[t=00:10] Screenshot 2 of 2\n"
        "Images follow this text in the order listed above.\n"
        "<image><image>\n</screenshots>"
    )
    assert accumulated_memory(["first", "second<image>"]) == (
        "[1] first\n[2] second<image_literal>"
    )

    oracle = PromptCondition.oracle_persona({"user": "profile"})
    system, user = apply_condition(oracle, "user", "observer\n", "request")
    assert system == "observer" + ORACLE_PROFILE_BLOCK.format(profile="profile")
    assert user == "request"


def test_osworld_induction_uses_original_context_and_user_templates() -> None:
    class Generator:
        def __init__(self) -> None:
            self.requests = []

        def generate_text(self, requests):
            request = requests[0]
            self.requests.append(request)
            if request.id.endswith("digestion"):
                return ['{"patterns":[{"evidence_ids":["h-0001"]}]}']
            return ['{"profile":"A grounded profile.","limitations":[]}']

    generator = Generator()
    examples = [_annotation(0, "yes"), _annotation(1, "no")]
    result = induce_osworld_persona(examples, generator)

    expected_context = render_induction_context(to_induction_records(examples))
    digestion, profile = generator.requests
    assert digestion.user_prompt == INDUCTION_DIGESTION_USER.format(context=expected_context)
    assert profile.user_prompt == INDUCTION_PROFILE_USER.format(
        context=expected_context,
        guidance=json.dumps(result.digestion, ensure_ascii=False, indent=2),
    )
    assert "assistance_feedback: support_wanted" in expected_context
    assert "assistance_feedback: support_not_wanted" in expected_context
    assert digestion.max_tokens == 2048
    assert profile.max_tokens == 1024

    multi_intent = to_induction_records([_annotation(2, "yes")])[0]
    multi_intent = InductionRecord(
        multi_intent.id,
        multi_intent.user_id,
        multi_intent.label,
        {**multi_intent.context, "user_intent": ("first", "second")},
    )
    assert "user_intent: ['first', 'second']" in render_induction_context([multi_intent])


class CocoPEBackend:
    def __init__(self) -> None:
        self.text_requests = []

    def predict(self, requests):
        return ['{"observation":"work","need_support":"no"}' for _ in requests]

    def generate_text(self, requests):
        request = requests[0]
        self.text_requests.append(request)
        if request.id == "coco-pe:reflect":
            return [
                json.dumps(
                    {
                        "reflections": [
                            {
                                "sequence_index": 0,
                                "reflection": "The decision missed a stable preference.",
                                "helpful_bullet_ids": [],
                                "harmful_bullet_ids": [],
                                "proposed_insights": [
                                    {
                                        "section": "when_to_support",
                                        "content": "Offer help when repetitive work stalls.",
                                    }
                                ],
                            }
                        ]
                    }
                )
            ]
        return [
            json.dumps(
                {
                    "ops": [
                        {
                            "op": "add",
                            "section": "when_to_support",
                            "content": "Offer help when repetitive work stalls.",
                        }
                    ]
                }
            )
        ]


def test_coco_pe_matches_original_reflector_curator_contract() -> None:
    backend = CocoPEBackend()
    example = MethodExample(
        id="example-1",
        user_id="user-1",
        label="yes",
        request=GenerationRequest("example-1", "observer", "screen state", ()),
    )
    checkpoints = []

    memory, audit = coco_pe_trainer(backend, backend).fit(
        [example],
        shuffle=False,
        checkpoint=lambda state, record: checkpoints.append((state.to_dict(), record.copy())),
    )

    assert memory.render(with_ids=False) == (
        "## When to proactively support\n- Offer help when repetitive work stalls."
    )
    assert audit == [
        {
            "epoch": 1,
            "batch": 1,
            "batch_acc": 0.0,
            "running_acc": 0.0,
            "n_wrong": 1,
            "n_reflection_targets": 1,
            "n_reflections": 1,
            "ops_applied": 1,
            "bullets_dropped": 0,
            "n_bullets": 1,
        }
    ]
    reflection_request, curator_request = backend.text_requests
    assert '<consecutive_examples order="oldest_to_newest">' in reflection_request.user_prompt
    assert "example_id: example-1" in reflection_request.user_prompt
    assert "ground_truth_need_support: yes" in reflection_request.user_prompt
    assert '"observation": "work"' in reflection_request.user_prompt
    assert "At most 8 operations" in curator_request.system_prompt
    assert "Reflection 1 (prediction was wrong)" in curator_request.user_prompt
    assert '"verdict"' not in curator_request.user_prompt
    assert checkpoints[0][0]["next"] == 2


def test_methods_do_not_import_a_specific_benchmark() -> None:
    methods_root = Path(__file__).parents[1] / "methods"
    source = "\n".join(path.read_text() for path in methods_root.glob("*.py"))
    assert "benchmarks.osworld_persona" not in source


class ReflectionPredictor:
    def predict(self, requests):
        return [
            json.dumps(
                {
                    "observation": "revised observation",
                    "user_intent": "revised intent",
                    "need_support": "yes",
                    "rationale": "revised rationale",
                }
            )
            for _ in requests
        ]


def test_coco_lora_keeps_correct_and_revises_wrong() -> None:
    first, second = _annotation(0), _annotation(1)
    complete = json.dumps(
        {
            "observation": "o",
            "user_intent": "i",
            "need_support": "yes",
            "rationale": "r",
        }
    )
    predictions = [_record(first, complete), _record(second, '{"need_support":"no"}')]
    revised, audit = coco_lora_builder(ReflectionPredictor()).build(
        to_method_examples([first, second], predictions),
        to_method_predictions(predictions),
    )
    assert [example.correction_source for example in revised] == ["direct", "self_reflection"]
    assert [item["status"] for item in audit] == ["accepted_direct", "accepted_reflection"]


def test_coco_lora_revised_data_round_trip(tmp_path: Path) -> None:
    example = to_method_examples([_annotation(0)])[0]
    target = json.dumps(
        {
            "observation": "o",
            "user_intent": "i",
            "need_support": "yes",
            "rationale": "r",
        }
    )
    path = tmp_path / "automation_first.json"
    write_revised_data([RevisedTrainingExample(example, target, "direct")], path)

    loaded = load_revised_data(path)

    assert len(loaded) == 1
    assert loaded[0].example.id == example.id
    assert loaded[0].example.user_id == "automation_first"
    assert loaded[0].example.request.image_paths == example.request.image_paths
    assert loaded[0].target == target
    assert loaded[0].correction_source == "direct"


class RecordingTrainingBackend:
    def __init__(self) -> None:
        self.examples = []
        self.config = None

    def train(self, examples, config):
        self.examples = list(examples)
        self.config = config
        return TrainedAdapter("recording", config.base_model, "adapter://trained")


def test_coco_lora_trainer_runs_revision_then_training(tmp_path: Path) -> None:
    first, second = _annotation(0), _annotation(1)
    complete = json.dumps(
        {
            "observation": "o",
            "user_intent": "i",
            "need_support": "yes",
            "rationale": "r",
        }
    )
    predictions = [_record(first, complete), _record(second, '{"need_support":"no"}')]
    training_backend = RecordingTrainingBackend()
    config = LoRATrainingConfig("Qwen/Qwen3.5-9B", tmp_path / "adapter")
    result = coco_lora_trainer(ReflectionPredictor(), training_backend).train(
        to_method_examples([first, second], predictions),
        to_method_predictions(predictions),
        config,
    )

    assert result.adapter.adapter_uri == "adapter://trained"
    assert [item.correction_source for item in training_backend.examples] == [
        "direct",
        "self_reflection",
    ]
    assert training_backend.config == config


def test_llamafactory_backend_prepares_dataset_and_safe_command(tmp_path: Path) -> None:
    repo = tmp_path / "LlamaFactory"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'llamafactory'\n")
    config_file = repo / "train.yaml"
    config_file.write_text("stage: sft\n")
    commands = []

    def record_command(command, cwd, env):
        commands.append((list(command), cwd, env))
        output_dir.mkdir()
        with (output_dir / "trainer_log.jsonl").open("w") as log:
            for step in range(1, 4):
                log.write(json.dumps({"current_steps": step, "total_steps": 3}) + "\n")

    method_example = to_method_examples([_annotation(0)])[0]
    revised = RevisedTrainingExample(method_example, '{"need_support":"yes"}', "direct")
    output_dir = tmp_path / "adapter"
    progress = []
    artifact = LlamaFactoryTrainingBackend(repo, config_file, runner=record_command).train(
        [revised],
        LoRATrainingConfig("Qwen/Qwen3.5-9B", output_dir, run_name="persona-one"),
        progress=lambda completed, total: progress.append((completed, total)),
    )

    dataset_path = Path(artifact.metadata["dataset_path"])
    rows = json.loads(dataset_path.read_text())
    assert rows[0]["messages"][-1]["content"] == '{"need_support":"yes"}'
    assert commands[0][1] == repo.resolve()
    assert commands[0][0][:5] == [
        "uv",
        "run",
        "--project",
        str(repo.resolve()),
        "llamafactory-cli",
    ]
    assert commands[0][0][5] == "train"
    assert f"output_dir={output_dir.resolve()}" in commands[0][0]
    assert "finetuning_type=lora" in commands[0][0]
    assert "disable_tqdm=true" in commands[0][0]
    assert "logging_steps=1" in commands[0][0]
    assert "logging_strategy=steps" in commands[0][0]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert artifact.adapter_uri == str(output_dir.resolve())


def test_tinker_conversation_preserves_image_position(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(b"png bytes for serialization test")
    example = MethodExample(
        id="example",
        user_id="user",
        label="yes",
        request=GenerationRequest(
            id="example",
            system_prompt="system",
            user_prompt="before<image>after",
            image_paths=(image,),
        ),
    )
    conversation = to_tinker_conversation(
        RevisedTrainingExample(example, '{"need_support":"yes"}', "direct")
    )

    content = conversation[1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_tinker_backend_runs_sft_steps_and_saves_adapter(tmp_path: Path, monkeypatch) -> None:
    revised = RevisedTrainingExample(
        MethodExample(
            id="one",
            user_id="user",
            label="yes",
            request=GenerationRequest("one", "system", "user", ()),
        ),
        '{"need_support":"yes"}',
        "direct",
    )

    class Future:
        def __init__(self, value=None) -> None:
            self.value = value

        async def result_async(self):
            return self.value

    class TrainingClient:
        def __init__(self) -> None:
            self.steps = 0

        def get_tokenizer(self):
            return "tokenizer"

        async def forward_backward_async(self, batch, loss_fn):
            assert batch and loss_fn == "cross_entropy"
            self.steps += 1
            return Future()

        async def optim_step_async(self, params):
            assert params["learning_rate"] == 1e-4
            return Future()

        async def save_weights_for_sampler_async(self, name, ttl_seconds):
            assert name == "test-run-final"
            assert ttl_seconds is None
            return Future(SimpleNamespace(path="tinker://run/sampler_weights/final"))

    training_client = TrainingClient()

    class ServiceClient:
        async def create_lora_training_client_async(self, base_model, rank):
            assert base_model == "Qwen/Qwen3.5-9B" and rank == 32
            return training_client

    runtime = SimpleNamespace(
        tinker=SimpleNamespace(AdamParams=lambda **kwargs: kwargs),
        get_recommended_renderer_name=lambda _model: "qwen3_5",
        get_renderer=lambda name, tokenizer, **kwargs: (name, tokenizer, kwargs),
        get_image_processor=lambda _model: None,
        train_on_last_assistant="last",
        conversation_to_datum=lambda conversation, renderer, **kwargs: (
            conversation,
            renderer,
            kwargs,
        ),
    )
    monkeypatch.setattr("training_backends.tinker._load_runtime", lambda: runtime)
    config = LoRATrainingConfig(
        "Qwen/Qwen3.5-9B",
        tmp_path / "tinker-output",
        run_name="test-run",
        num_epochs=2,
    )
    progress = []
    artifact = asyncio.run(
        TinkerBackend(service_client=ServiceClient()).train_async(
            [revised],
            config,
            progress=lambda completed, total: progress.append((completed, total)),
        )
    )

    assert training_client.steps == 2
    assert progress == [(1, 2), (2, 2)]
    assert artifact.adapter_uri == "tinker://run/sampler_weights/final"
    assert (config.output_dir / "training_artifact.json").is_file()
