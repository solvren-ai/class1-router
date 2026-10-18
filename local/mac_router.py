# SPDX-License-Identifier: MIT-0
"""Local Apple Silicon (MLX) train + eval path for the SLM platform.

The platform trains on SageMaker (LLaMA-Factory container) and evaluates with
vLLM; neither runs on a Mac. This adds a small local path that keeps the same
data contract and scoring code:

  * data   — the platform's canonical {"messages": [...]} JSONL rows.
  * eval   — reuses container/eval.py (extract_answer, _normalize,
             compute_metrics) so local numbers line up with platform numbers.
  * train  — mlx-lm LoRA on the Mac's GPU instead of SageMaker.

    uv run python mac_router.py prep      # mlx-ready copy of the dataset
    uv run python mac_router.py train     # LoRA fine-tune (adapters/)
    uv run python mac_router.py eval      # base + fine-tuned -> metrics JSON
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
DATA = HERE / "data" / "router"
MLX_DATA = HERE / "data" / "mlx"
ADAPTERS = HERE / "adapters"
METRICS = HERE / "metrics"
MODEL = "mlx-community/Qwen3-0.6B-bf16"
LABELS = ["small-model", "frontier-model", "retrieval", "refuse-or-escalate"]


def _platform_eval():
    """Load container/eval.py as a module (its import-time deps are stdlib only)."""
    spec = importlib.util.spec_from_file_location(
        "slm_container_eval", REPO / "container" / "eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prep() -> None:
    """mlx-lm wants bare {"messages": [...]} rows; drop our extra `domain` key."""
    MLX_DATA.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        rows = [
            json.loads(line)
            for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        out = MLX_DATA / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        print(f"{out}: {len(rows)} rows")


def train(iters: int, batch_size: int, num_layers: int, max_seq_len: int, lr: float) -> None:
    ADAPTERS.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--train",
        "--data", str(MLX_DATA),
        "--fine-tune-type", "lora",
        "--num-layers", str(num_layers),
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--max-seq-length", str(max_seq_len),
        "--learning-rate", str(lr),
        "--mask-prompt",          # score only the route label, not the policy prompt
        "--grad-checkpoint",      # 16GB shared RAM, other local models running
        "--adapter-path", str(ADAPTERS),
        "--steps-per-eval", "100",
        "--val-batches", "8",
        "--seed", "42",
    ]
    print(" ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=HERE)
    secs = time.perf_counter() - t0
    METRICS.mkdir(exist_ok=True)
    (METRICS / "train_time.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "train_seconds": round(secs, 1),
                "iters": iters,
                "batch_size": batch_size,
                "num_layers": num_layers,
                "max_seq_length": max_seq_len,
                "learning_rate": lr,
                "grad_checkpoint": True,
                "mask_prompt": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train] {secs/60:.1f} min")


def _route_of(text: str, ev) -> str:
    """Map a raw generation to one of the four routes, or 'unparsed'.

    Reuses the container's scaffolding stripper so a reasoning model that wraps
    its answer in <think> isn't unfairly scored. Longest label first so
    'small-model' can't shadow a longer label containing it.
    """
    s = ev._normalize(ev.extract_answer(text))
    hits = [lab for lab in sorted(LABELS, key=len, reverse=True) if ev._normalize(lab) in s]
    return hits[0] if hits else "unparsed"


def _macro_f1(golds: list[str], preds: list[str]) -> tuple[dict, float]:
    per_class = {}
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {
            "support": tp + fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        f1s.append(f1)
    return per_class, round(sum(f1s) / len(f1s), 4)


def evaluate(adapter: Path | None, max_tokens: int, limit: int | None) -> dict:
    from mlx_lm import generate, load

    ev = _platform_eval()
    rows = [
        json.loads(line)
        for line in (DATA / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        rows = rows[:limit]

    model, tokenizer = load(MODEL, adapter_path=str(adapter) if adapter else None)
    preds_raw, golds, lat_ms = [], [], []
    for i, row in enumerate(rows):
        msgs = row["messages"]
        golds.append(msgs[-1]["content"])
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        t0 = time.perf_counter()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        lat_ms.append((time.perf_counter() - t0) * 1000)
        preds_raw.append(out)
        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    preds = [_route_of(p, ev) for p in preds_raw]
    per_class, macro_f1 = _macro_f1(golds, preds)
    correct = sum(g == p for g, p in zip(golds, preds))
    lat = sorted(lat_ms)
    return {
        "model": MODEL,
        "adapter": str(adapter.relative_to(HERE)) if adapter else None,
        "count": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "unparsed": sum(p == "unparsed" for p in preds),
        "confusion": {
            g: dict(Counter(p for gg, p in zip(golds, preds) if gg == g)) for g in LABELS
        },
        "latency_ms": {
            "mean": round(sum(lat) / len(lat), 1),
            "p50": round(ev._percentile(lat, 0.5), 1),
            "p90": round(ev._percentile(lat, 0.9), 1),
            "p99": round(ev._percentile(lat, 0.99), 1),
        },
        # The platform's own metric panel, computed by container/eval.py on the
        # same predictions, so local runs are comparable to SageMaker runs.
        "platform_metrics": ev.compute_metrics(preds_raw, golds),
        "decoding": {"temperature": 0.0, "max_new_tokens": max_tokens, "seed": 42},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="eval only the first N test rows")
    ap.add_argument("--which", choices=["base", "tuned", "both"], default="both")
    a = ap.parse_args()

    if a.cmd == "prep":
        prep()
    elif a.cmd == "train":
        train(a.iters, a.batch_size, a.num_layers, a.max_seq_length, a.learning_rate)
    else:
        METRICS.mkdir(exist_ok=True)
        targets = {"base": None, "tuned": ADAPTERS}
        for name in ["base", "tuned"] if a.which == "both" else [a.which]:
            print(f"[eval] {name}")
            m = evaluate(targets[name], a.max_tokens, a.limit)
            out = METRICS / f"metrics_{name}.json"
            out.write_text(json.dumps(m, indent=2), encoding="utf-8")
            print(f"  acc={m['accuracy']} macro_f1={m['macro_f1']} "
                  f"p50={m['latency_ms']['p50']}ms unparsed={m['unparsed']} -> {out}")


if __name__ == "__main__":
    main()
