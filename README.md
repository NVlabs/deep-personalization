# Deep Personalization

Deep personalization is a closed-loop process in which an agent continuously observes user activity, retains useful context, makes personalized intervention decisions or responses, and adapts those decisions based on explicit or implicit user feedback.

This repository provides benchmarks, personalization methods, and backend-independent tooling for training and evaluating deep personalization.

## Install

```bash
uv sync
```

Python 3.11+ is required, including for Tinker inference and training.

## Benchmarks

### OSWorld-Persona

OSWorld-Persona measures whether an agent learns when each user wants proactive support. It contains four personas with session-disjoint train and test data.

Download [collaborative-agents/OSWorld-Persona](https://huggingface.co/datasets/collaborative-agents/OSWorld-Persona):

```bash
uv run hf download collaborative-agents/OSWorld-Persona \
  --repo-type dataset \
  --local-dir data/osworld-persona
```

- [Benchmark format, metrics, and API](benchmarks/osworld_persona)
- [Data synthesis pipeline](benchmarks/osworld_persona/synthesis)


## Methods

We currently support the following methods:

| Method            | Personalization                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| Vanilla prompting | Prompt the model with the prediction prompt directly                                                  |
| Oracle persona    | Inject ground-truth persona to the prediction prompt                                                  |
| Induced persona   | Infer user persona from labeled training history and inject inferred persona to the prediction prompt |
| Coco-PE           | Evolve the prediction prompt from prediction feedback                                                 |
| Coco-LoRA         | Persona-specific LoRA trained on self-revised supervision                                             |

### Add a New Methods

1. Implement the method under `methods/` using `methods/types.py` and the shared
   predictor/trainer protocols.
2. Add benchmark-specific prompts and conversions to
   `benchmarks/<name>/method_adapter.py`.
3. Expose preparation and evaluation through the benchmark CLI.
4. Add a backend adapter only when the method needs a new inference or training
   framework.

## Method Backend

Method code defines how personalization is learned or applied; a backend runs the model. Backends implement one of the shared interfaces:

- `Predictor` handles multimodal `GenerationRequest` batches.
- `TextGenerator` handles text-only induction, reflection, and prompt evolution.
- `LoRATrainingBackend` trains revised examples and returns a `TrainedAdapter`.

The repository provides [OpenAI-compatible and native Tinker inference backends](inference_backends) and [LLaMA-Factory and Tinker training backends](training_backends).

## How to Benchmark a Method

Each benchmark runs a three-stage pipeline:

**Prepare → Predict → Score**

**Prepare:** Load and validate benchmark splits, then construct any method-specific artifact. Vanilla and oracle prompting require no learned artifact; induced persona and Coco-PE learn prompt context; Coco-LoRA revises training targets and trains an adapter through a method backend.

**Predict:** The benchmark combines each example, its prompts, and the method artifact into a `GenerationRequest`. The inference backend returns one raw response per stable example ID while the benchmark preserves required session or user ordering.

**Score:** The benchmark parses raw responses, aligns them with ground truth by stable ID, and computes its metrics.

See the [OSWorld-Persona guide](benchmarks/osworld_persona) for complete commands for every supported method and external scoring.

## External Integration
Deep Personalization is integrated into [Coco](https://github.com/collaborative-agents/coco), a proactive agent that grows with you.


## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

Please cite our paper if you use this code/data or part of it in your work:
```bibtex
@article{shao2026deep,
  title={Deep Personalization: Continuous Context Observation for Local Proactive Agents},
  author={Shao, Yijia and Bian, Song and Guo, Junxian and Jiang, Jindong and Dong, Zhen and Mao, Huizi and Yang, Diyi and Han, Song},
  journal={arXiv preprint arXiv:2611.13457},
  year={2026}
}
```