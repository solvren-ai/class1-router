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
  - synthetic
---

<!-- DRAFT — not published. Review before any upload to Hugging Face. -->

# Class 1 Decider router (Qwen3-0.6B LoRA, MLX)

A small request router. It reads one user request and replies with one of four routes, so a
lightweight layer in front of LLM calls can decide which model or path handles it:

| Route | Meaning |
| --- | --- |
| `small-model` | simple, well-scoped, low-risk text work (reformat, short summary, extract, translate, classify) |
| `frontier-model` | complex multi-step reasoning, long-form drafting, nuanced analysis |
| `retrieval` | needs specific internal documents, records, or current external sources |
| `refuse-or-escalate` | high-stakes individualized advice, another person's private records, or anything that needs a qualified human |

This is a showcase of Solvren AI's "Class 1 Decider" pattern. It is a demo trained on synthetic data,
not a production safety system.

## Model details

- **Base model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0), loaded as the
  MLX conversion `mlx-community/Qwen3-0.6B-bf16`.
- **Method:** LoRA, rank 8, last 8 transformer layers, 1.44M trainable parameters (0.24% of 596M),
  loss masked to the assistant label (`--mask-prompt`).
- **Framework:** [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.32.2 on Apple Silicon.
- **Artifact:** LoRA adapter (`adapters.safetensors`, 5.5MB) plus `adapter_config.json`. Fuse it into
  the base model with `mlx_lm fuse` for a standalone checkpoint.
- **License:** Apache-2.0, inherited from the base model. The training code and dataset generator are
  MIT-0, as is the rest of the repository.

## Training data

1,536 synthetic examples generated offline from templates by `local/gen_router_data.py`, with no
model calls, no AWS and no real PII or PHI. Every name, record id and company is a fictional
placeholder.

- **Domains:** healthcare, legal, fintech, biotech.
- **Construction:** 4 domains x 4 routes x 12 core requests, with 8 phrasing variants each (greeting,
  urgency marker, team prefix and so on) and fictional slot values (drug, contract type, assay).
- **Splits:** 1,024 train / 256 valid / 256 test, balanced across the routes. Splitting is by **core
  request**: a request and all of its variants land in one split, so the test set holds requests the
  model never saw in any phrasing.
- **`refuse-or-escalate`** is built from benign escalation triggers (urgent symptoms, individualized
  medical, legal or financial decisions, another person's records, skipping a required human
  approval), not from examples of harmful instructions.
- **Format:** the repository's canonical chat JSONL (`{"messages": [system, user, assistant]}`).
  The system turn carries the route definitions, and the assistant turn is the bare label.

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

Validation loss went from 5.80 before training to 0.083 at iteration 100. Training loss was about
0.000 by iteration 240.

## Evaluation

Held-out test set: 256 requests (64 per route), greedy decoding (temperature 0, max 32 new tokens),
`enable_thinking=False` in the Qwen3 chat template, and the same prompt for both models. Answers are
scored after the repository's `container/eval.py` scaffold stripping, and a generation that does
not contain one of the four route names counts as wrong.

| Metric | Base Qwen3-0.6B | Fine-tuned |
| --- | --- | --- |
| Accuracy | 0.098 | **0.992** |
| Macro F1 | 0.163 | **0.992** |
| Unparseable outputs | 228 / 256 | 0 / 256 |
| Latency p50 (ms / request) | 209.9 | 246.5 |
| Latency p90 (ms / request) | 334.4 | 360.2 |
| Latency p99 (ms / request) | 631.7 | 513.7 |

Per-class F1:

| Route | Base | Fine-tuned |
| --- | --- | --- |
| `small-model` | 0.031 | 0.992 |
| `frontier-model` | 0.030 | 0.985 |
| `retrieval` | 0.222 | 1.000 |
| `refuse-or-escalate` | 0.370 | 0.992 |

The fine-tuned model made two errors, and both went to `frontier-model`: one `small-model` request and
one `refuse-or-escalate` request.

The base model mostly ignores the label set and invents its own routes (`simple`, `legal`,
`route: critique-and-propose-...`). That failure to follow the output format explains most of its
low score.

Latency is single-request generation on the M5 through the mlx-lm Python API, with the adapter
**unfused**. The extra LoRA matmuls explain why the tuned p50 is higher than the base. Fusing the
adapter should bring it back to roughly base-model cost, but that has not been measured.

The full metric JSON, including the repository's own `compute_metrics` panel, is in
`local/metrics/metrics_base.json`, `local/metrics/metrics_tuned.json` and
`local/metrics/train_time.json`.

## Limitations

- **Synthetic and templated.** The test set holds out core requests, but they come from the same
  template family and the same writer as the training set. Real traffic is messier, so expect
  noticeably lower accuracy on real requests. Before relying on this router, evaluate it on a
  labelled sample of your own traffic.
- **English only, four domains, four routes.** Requests outside healthcare, legal, fintech and
  biotech have not been evaluated.
- **The system prompt is part of the contract.** The model was trained with the route definitions in
  the system turn. Change them and it needs retraining.
- **Not a safety classifier.** `refuse-or-escalate` is a routing hint for a human-in-the-loop path.
  It must not be the only control on regulated or unsafe requests.

## How to reproduce

From `local/` in the repository (Apple Silicon, uv):

```bash
uv sync
uv run python gen_router_data.py
uv run python mac_router.py prep
uv run python mac_router.py train
uv run python mac_router.py eval
```

## Use

```python
from mlx_lm import load, generate

model, tok = load("mlx-community/Qwen3-0.6B-bf16", adapter_path="adapters")
msgs = [
    {"role": "system", "content": SYSTEM},  # the exact routing prompt from gen_router_data.py
    {"role": "user", "content": "Summarize this lab report in one sentence."},
]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
print(generate(model, tok, prompt=prompt, max_tokens=8))  # -> small-model
```
