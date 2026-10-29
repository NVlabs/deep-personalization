# OSWorld-Persona

OSWorld-Persona evaluates whether a proactive desktop agent learns **when a particular user wants support**. Each example contains a short chronological screen history, an inferred current intent, and a binary `want_support` label. The positive class is an interrupt-worthy moment.

## Data

Download [collaborative-agents/OSWorld-Persona](https://huggingface.co/datasets/collaborative-agents/OSWorld-Persona) from Hugging Face:

```bash
uv run hf download collaborative-agents/OSWorld-Persona \
  --repo-type dataset \
  --local-dir data/osworld-persona
```

## Run the benchmark

[`cli.py`](cli.py) is the go-to entry point, exposed as `uv run
osworld-persona`. It supports:

- `validate`: validate the dataset and print split statistics.
- `export`: convert a split to canonical, framework-neutral JSONL.
- `run`: evaluate a prompting condition through the selected inference backend.
- `score`: score externally generated predictions using benchmark metrics.
- `induce-persona`: infer persona profiles from labeled training histories.
- `evolve-coco-pe`: evolve prompt memories from prediction feedback.
- `prepare-coco-lora`, `train-coco-lora`, `eval-coco-lora`: revise training
  targets, train persona adapters, and evaluate a local adapter.

### Vanilla prompting

**LLaMA-Factory.** Start the model server in one terminal:

```bash
cd "<llamafactory-root>"
uv run llamafactory-cli api \
  examples/inference/qwen3vl.yaml \
  model_name_or_path="<base-model>" \
  template="<suitable-template, e.g. qwen3_5_nothink>" \
  infer_backend=huggingface
```

Run the benchmark in another terminal:

```bash
uv run osworld-persona run \
  --data-root "<data-root>" \
  --split test \
  --condition vanilla \
  --base-url "<llamafactory-base-url>/v1" \
  --model "<base-model>" \
  --output results/vanilla.json
```

**Tinker.** Sample the supported vision model through Tinker's native SDK:

```bash
uv run osworld-persona run \
  --backend tinker \
  --data-root "<data-root>" \
  --split test \
  --condition vanilla \
  --api-key "<tinker-api-key>" \
  --model "<tinker-base-model>" \
  --output results/vanilla-tinker.json
```

Native Tinker inference accepts either a supported base-model name or a `tinker://.../sampler_weights/...` URI through `--model`. Use `--tinker-renderer` only when overriding the model's recommended renderer.

### Oracle persona

**LLaMA-Factory.** Start the base-model server as shown under vanilla prompting, then run:

```bash
uv run osworld-persona run \
  --data-root "<data-root>" \
  --split test \
  --condition oracle_persona \
  --base-url "<llamafactory-base-url>/v1" \
  --model "<base-model>" \
  --output results/oracle-persona.json
```

**Tinker.** Use native sampling with the base model:

```bash
uv run osworld-persona run \
  --backend tinker \
  --data-root "<data-root>" \
  --split test \
  --condition oracle_persona \
  --api-key "<tinker-api-key>" \
  --model "<tinker-base-model>" \
  --output results/oracle-persona-tinker.json
```

### Induced persona

Infer one leakage-safe profile per persona from the training split, then use that mapping during evaluation:

**LLaMA-Factory.** Start the base-model server as shown under vanilla prompting, then run both stages against it:

```bash
uv run osworld-persona induce-persona \
  --data-root "<data-root>" \
  --split train \
  --base-url "<llamafactory-base-url>/v1" \
  --model "<base-model>" \
  --output artifacts/induced-personas.json \
  --audit-output artifacts/induced-personas-audit.json

uv run osworld-persona run \
  --data-root "<data-root>" \
  --split test \
  --condition induced_persona \
  --condition-file artifacts/induced-personas.json \
  --base-url "<llamafactory-base-url>/v1" \
  --model "<base-model>" \
  --output results/induced-persona.json
```

**Tinker.** Run both stages through native sampling:

```bash
uv run osworld-persona induce-persona \
  --backend tinker \
  --data-root "<data-root>" \
  --split train \
  --api-key "<tinker-api-key>" \
  --model "<tinker-base-model>" \
  --output artifacts/induced-personas-tinker.json \
  --audit-output artifacts/induced-personas-tinker-audit.json

uv run osworld-persona run \
  --backend tinker \
  --data-root "<data-root>" \
  --split test \
  --condition induced_persona \
  --condition-file artifacts/induced-personas-tinker.json \
  --api-key "<tinker-api-key>" \
  --model "<tinker-base-model>" \
  --output results/induced-persona-tinker.json
```

### Coco-PE

Evolve one prompt memory per persona on the training split, then inject it at evaluation time:

**LLaMA-Factory.** Start the base-model server as shown under vanilla prompting, then run both stages against it:

```bash
uv run osworld-persona evolve-coco-pe \
  --data-root "<data-root>" \
  --split train \
  --base-url "<llamafactory-base-url>/v1" \
  --model "<base-model>" \
  --output artifacts/coco-pe-memories.json \
  --audit-output artifacts/coco-pe-audit.json

uv run osworld-persona run \
  --data-root "<data-root>" \
  --split test \
  --condition coco_pe \
  --condition-file artifacts/coco-pe-memories.json \
  --base-url "<llamafactory-base-url>/v1" \
  --model "<base-model>" \
  --output results/coco-pe.json
```

**Tinker.** Run both stages through native sampling:

```bash
uv run osworld-persona evolve-coco-pe \
  --backend tinker \
  --data-root "<data-root>" \
  --split train \
  --api-key "<tinker-api-key>" \
  --model "<tinker-base-model>" \
  --output artifacts/coco-pe-memories-tinker.json \
  --audit-output artifacts/coco-pe-tinker-audit.json

uv run osworld-persona run \
  --backend tinker \
  --data-root "<data-root>" \
  --split test \
  --condition coco_pe \
  --condition-file artifacts/coco-pe-memories-tinker.json \
  --api-key "<tinker-api-key>" \
  --model "<tinker-base-model>" \
  --output results/coco-pe-tinker.json
```

Use `--resume` with either evolution command to continue from Coco-PE checkpoints.

### Coco-LoRA

First generate revised supervision while the base-model endpoint is running:

**LLaMA-Factory.** Start the base-model server as shown under vanilla prompting, then prepare the training data:

```bash
uv run osworld-persona prepare-coco-lora \
  --data-root "<data-root>" \
  --split train \
  --base-url "<llamafactory-base-url>/v1" \
  --model "<base-model>" \
  --output-dir artifacts/coco-lora-prepared
```

With the revised data, train with an unmodified LLaMA-Factory checkout:

```bash
uv run osworld-persona train-coco-lora \
  --prepared-dir artifacts/coco-lora-prepared \
  --base-model "<base-model>" \
  --llamafactory-root "<llamafactory-root>" \
  --output-dir artifacts/coco-lora
```

Note: By default this command uses the [bundled LLaMA-Factory config](../../training_backends/configs/qwen3_5_lora.yaml), which enables BF16 training and sets `gradient_accumulation_steps: 8`. `--training-batch-size` is the per-device micro-batch size, so its default value of 1 gives an effective batch size of 8 on one GPU. Pass `--training-config <config.yaml>` to use a different LLaMA-Factory configuration.

After training, evaluate every persona sequentially. Each persona has its own adapter, so the command starts and stops a matching LLaMA-Factory server for each run.

```bash
for persona in \
  automation_first \
  eager_handson \
  learning_oriented \
  reserved_drafts
do
  uv run osworld-persona eval-coco-lora \
    --data-root "<data-root>" \
    --split test \
    --personas "$persona" \
    --adapter-path "artifacts/coco-lora/$persona" \
    --llamafactory-root "<llamafactory-root>" \
    --model "<base-model>" \
    --output "results/coco-lora-$persona.json"
done
```

Combine the predictions and compute aggregate metrics:

```bash
jq -c \
  '.predictions[] | {id, prediction: .raw_prediction}' \
  results/coco-lora-{automation_first,eager_handson,learning_oriented,reserved_drafts}.json \
  > results/coco-lora-all.jsonl

uv run osworld-persona score \
  --data-root "<data-root>" \
  --split test \
  --predictions results/coco-lora-all.jsonl \
  --method coco-lora \
  --output results/coco-lora-all.json
```

**Tinker.** Prepare supervision through native sampling from the base model:

```bash
uv run osworld-persona prepare-coco-lora \
  --backend tinker \
  --data-root "<data-root>" \
  --split train \
  --api-key "<tinker-api-key>" \
  --model "<tinker-base-model>" \
  --output-dir artifacts/coco-lora-tinker-prepared
```

With the revised data, train with Tinker backend:

```bash
uv run osworld-persona train-coco-lora \
  --training-backend tinker \
  --prepared-dir artifacts/coco-lora-tinker-prepared \
  --base-model "<tinker-base-model>" \
  --tinker-api-key "<tinker-api-key>" \
  --output-dir artifacts/coco-lora-tinker
```

The resulting `index.json` contains one sampler-weight URI per persona. Evaluate all persona adapters through native sampling:

```bash
for persona in \
  automation_first \
  eager_handson \
  learning_oriented \
  reserved_drafts
do
  adapter_uri="$(jq -r --arg persona "$persona" \
    '.[$persona].adapter_uri' artifacts/coco-lora-tinker/index.json)"

  uv run osworld-persona run \
    --backend tinker \
    --data-root "<data-root>" \
    --split test \
    --personas "$persona" \
    --condition coco_lora \
    --api-key "<tinker-api-key>" \
    --model "$adapter_uri" \
    --output "results/coco-lora-tinker-$persona.json"
done
```

Combine the predictions and compute aggregate metrics:

```bash
jq -c \
  '.predictions[] | {id, prediction}' \
  results/coco-lora-tinker-{automation_first,eager_handson,learning_oriented,reserved_drafts}.json \
  > results/coco-lora-tinker-all.jsonl

uv run osworld-persona score \
  --data-root "<data-root>" \
  --split test \
  --predictions results/coco-lora-tinker-all.jsonl \
  --method coco-lora \
  --output results/coco-lora-tinker-all.json
```

### Score external predictions

Predictions are JSONL objects keyed by stable benchmark ID:

```json
{"id":"automation_first:test:000000","prediction":"{\"need_support\":\"no\"}"}
```

The `prediction` may be the raw model string or a decoded JSON object. Extra fields are retained as provenance.

```bash
uv run osworld-persona score \
  --data-root "<data-root>" \
  --split test \
  --predictions "<predictions.jsonl>" \
  --output "<results.json>"
```
