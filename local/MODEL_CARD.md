---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: mlx
pipeline_tag: text-generation
language:
  - en
tags:
  - mlx
  - lora
  - router
  - text-classification
  - request-routing
datasets:
  - SolvrenAI/class1-router-synthetic
---

<!-- DRAFT — private repo, pending review. -->

# Class 1 Decider router (Qwen3-0.6B, MLX)

A small request router. It reads one user request and replies with one of four routes, so a
lightweight layer in front of LLM calls can decide which model or path handles it:

| Route | Meaning |
| --- | --- |
| `small-model` | simple, well-scoped, low-risk text work (reformat, short summary, extract, translate, classify) |
| `frontier-model` | complex multi-step reasoning, long-form drafting, nuanced analysis |
| `retrieval` | needs specific internal documents, records, or current external sources |
| `refuse-or-escalate` | high-stakes individualized advice, another person's private records, or anything that needs a qualified human |

This is a showcase of the "Class 1 Decider" pattern from Solvren AI (AI forward deployment). It is a
demo trained on synthetic data, not a production safety system.

## Files

- **Repository root:** the fused model, with the LoRA adapter merged into the base weights. It loads
  standalone with `mlx_lm.load`.
- **`adapter/`:** the LoRA adapter on its own (`adapters.safetensors`, 5.5MB, plus
  `adapter_config.json`), to apply on top of `mlx-community/Qwen3-0.6B-bf16`.

## Model details

- **Base model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0), loaded as the
  MLX conversion `mlx-community/Qwen3-0.6B-bf16`.
- **Method:** LoRA, rank 8, last 8 transformer layers, 1.44M trainable parameters (0.24% of 596M),
  loss masked to the assistant label. The adapter was then fused with `mlx_lm fuse`.
- **Framework:** [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.32.2 on Apple Silicon.
- **License:** Apache-2.0, inherited from the base model. The training code and dataset are MIT-0.

## Routing prompt

The model was trained with this exact system prompt, and it is part of the input contract. Use it
verbatim. It is copied from `local/gen_router_data.py`:

```python
SYSTEM = (
    "You are a request router. Classify the user's request into exactly one route and "
    "reply with only the route name.\n"
    "small-model: simple, well-scoped, low-risk text work (reformat, short summary, "
    "extract, translate, classify).\n"
    "frontier-model: complex multi-step reasoning, long-form drafting, nuanced analysis.\n"
    "retrieval: needs specific internal documents, records, or current external sources.\n"
    "refuse-or-escalate: high-stakes individualized advice, another person's private "
    "records, or anything that needs a qualified human."
)
```

## Use

```python
from mlx_lm import load, generate

SYSTEM = (
    "You are a request router. Classify the user's request into exactly one route and "
    "reply with only the route name.\n"
    "small-model: simple, well-scoped, low-risk text work (reformat, short summary, "
    "extract, translate, classify).\n"
    "frontier-model: complex multi-step reasoning, long-form drafting, nuanced analysis.\n"
    "retrieval: needs specific internal documents, records, or current external sources.\n"
    "refuse-or-escalate: high-stakes individualized advice, another person's private "
    "records, or anything that needs a qualified human."
)

model, tok = load("SolvrenAI/class1-router-qwen3-0.6b-mlx")
msgs = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Summarize this lab report in one sentence."},
]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
print(generate(model, tok, prompt=prompt, max_tokens=8))  # -> small-model
```

To use the adapter instead of the fused model, download the repository and call
`load("mlx-community/Qwen3-0.6B-bf16", adapter_path="<local path>/adapter")`.

## Training data

The training data is [SolvrenAI/class1-router-synthetic](https://huggingface.co/datasets/SolvrenAI/class1-router-synthetic):
1,536 template-generated rows split 1,024 train / 256 validation / 256 test, plus a 50-row
hand-written test set.

- **Domains:** healthcare, legal, fintech, biotech.
- **Privacy:** no real PII or PHI.
- **Leakage control:** splitting is by core request, so test requests never appear in training in
  any phrasing.
- **`refuse-or-escalate`:** built from benign escalation triggers, not harmful instructions.

## Training procedure

| Setting | Value |
| --- | --- |
| Iterations | 400 (batch size 2, ~0.8 epoch) |
| Learning rate | 1e-4, Adam |
| Max sequence length | 512 |
| Gradient checkpointing | on |
| Seed | 42 |
| Hardware | Apple M5, 16GB unified memory, shared with other local workloads |
| Wall-clock | 110 s |

## Evaluation

Every model gets the same system prompt, greedy decoding (temperature 0, max 32 new tokens) and
`enable_thinking=False`. Answers pass through the repository's `container/eval.py` scaffold
stripping. A generation that doesn't contain one of the four route names counts as wrong
("unparsed").

The rows compared:

- **base:** Qwen3-0.6B zero-shot.
- **base-fewshot:** Qwen3-0.6B with 8 labelled examples in the prompt (2 per route, drawn from train
  only).
- **tuned:** Qwen3-0.6B with the LoRA adapter, unfused.
- **fused:** this model.
- **4b, 4b-fewshot:** `mlx-community/Qwen3-4B-4bit`, not fine-tuned, zero-shot and with the same 8
  examples, as a bigger-model reference.

There are two test sets:

- **template:** 256 held-out rows from the generator.
- **handwritten:** 50 requests written by hand in a different style, with typos and lowercase, mixed
  lengths, and general business requests outside the four training domains.

| Model | Test set | n | Accuracy | Macro F1 | Unparsed | p50 ms | p90 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | template | 256 | 0.098 | 0.163 | 228 | 216 | 330 |
| base-fewshot | template | 256 | 0.469 | 0.422 | 21 | 281 | 398 |
| tuned | template | 256 | 0.992 | 0.992 | 0 | 239 | 362 |
| **fused** | template | 256 | **0.996** | **0.996** | 0 | **196** | 289 |
| 4b | template | 256 | 0.887 | 0.888 | 0 | 383 | 738 |
| 4b-fewshot | template | 256 | 0.894 | 0.893 | 1 | 585 | 1130 |
| base | handwritten | 50 | 0.060 | 0.100 | 47 | 196 | 357 |
| base-fewshot | handwritten | 50 | 0.400 | 0.336 | 5 | 317 | 745 |
| tuned | handwritten | 50 | 0.880 | 0.883 | 0 | 350 | 669 |
| **fused** | handwritten | 50 | **0.880** | **0.883** | 0 | **137** | 280 |
| 4b | handwritten | 50 | 0.900 | 0.897 | 0 | 382 | 628 |
| 4b-fewshot | handwritten | 50 | 0.840 | 0.893 | 7 | 724 | 884 |

Per-class F1 for the fused model:

| Route | Template | Handwritten |
| --- | --- | --- |
| `small-model` | 1.000 | 0.870 |
| `frontier-model` | 0.992 | 0.917 |
| `retrieval` | 1.000 | 0.917 |
| `refuse-or-escalate` | 0.992 | 0.828 |

### What the numbers say

- **The template test set is easy.** 0.996 there mostly shows that the model learned the template
  family. The hand-written set is the more honest signal: **0.88**, with 6 errors in 50.
- **The fused 0.6B model roughly matches an untuned 4B model on hand-written requests** (0.88 vs
  0.90) at about a third of the p50 latency. The 4B model is far ahead of the untuned 0.6B in either
  prompting mode.
- **The two models fail in different directions.** Five of the fused model's 6 hand-written errors
  route a harmless request to `refuse-or-escalate`, such as a billing-or-scheduling email triage, or
  asking who the on-call pharmacist is. That over-escalation costs a human hand-off. The 4B model
  instead sent 4 of 12 escalation cases to `retrieval`: a coworker's salary, another patient's
  results, releasing an out-of-spec batch, a list of flagged customers. For a router in a regulated
  setting, that miss is the more dangerous error.
- **Few-shot prompting helps the 0.6B base (0.06 to 0.40) but not the 4B (0.90 to 0.84).** With
  examples in the prompt, the 4B often just performed the request (it wrote the bullets, translated
  the text) instead of routing it.
- **Fusing cuts latency** with no change in hand-written accuracy: p50 dropped from 239 ms to 196 ms on
  the template set, measured back to back, because the separate LoRA matmuls are gone.
- **Latency figures are approximate.** They are single-request mlx-lm Python API calls on a shared
  16GB M5 that was running other ML jobs at the same time, so tail latency (p90/p99) is noisy and
  some rows ran under heavier load than others. Compare p50 values within one run, and re-measure on
  your own hardware.

Full metric JSON is in `local/metrics/metrics_<model>_<testset>.json` in the source repository. It
includes per-class precision and recall, confusion matrices, the platform's own `compute_metrics`
panel, and every hand-written prediction.

## Limitations

- **Synthetic training data from one template family.** Expect lower accuracy on real traffic than
  even the hand-written number suggests. Evaluate on a labelled sample of your own requests first.
- **Over-escalation.** The model leans toward `refuse-or-escalate` on short, unfamiliar requests.
  That fails safe, but it will push extra traffic to a human queue.
- **Small hand-written set.** At n=50, one error moves accuracy by 2 points.
- **English only; four domains.** General business requests were only spot-checked in the
  hand-written set.
- **The system prompt is part of the contract.** Change the route definitions and the model needs
  retraining.
- **Not a safety classifier.** `refuse-or-escalate` is a routing hint for a human-in-the-loop path.
  It must not be the only control on regulated or unsafe requests.

## How to reproduce

From `local/` in the [source repository](https://github.com/solvren-ai/class1-router)
(Apple Silicon, uv):

```bash
uv sync
uv run python run.py all --local      # data -> LoRA train -> fuse -> eval (all models x both test sets) -> table
```
