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
    uv run python mac_router.py fuse      # adapter fused into merged/
    uv run python mac_router.py eval      # every target x test set -> metrics/
    uv run python mac_router.py table     # markdown summary of metrics/
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
FUSED = HERE / "merged"
METRICS = HERE / "metrics"
MODEL = "mlx-community/Qwen3-0.6B-bf16"
BIG = "mlx-community/Qwen3-4B-4bit"  # un-tuned "bigger model" reference
# name -> (model, adapter, few_shot)
TARGETS = {
    "base": (MODEL, None, False),
    "base-fewshot": (MODEL, None, True),
    "tuned": (MODEL, ADAPTERS, False),
    "fused": (FUSED, None, False),
    "4b": (BIG, None, False),
    "4b-fewshot": (BIG, None, True),
}
TEST_FILES = {"template": "test.jsonl", "handwritten": "handwritten_test.jsonl"}
SHOTS_PER_ROUTE = 2
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


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _few_shot_turns() -> list[dict]:
    """SHOTS_PER_ROUTE labelled examples per route, drawn from TRAIN only, as
    prior user/assistant turns. Fixed seed so every model sees the same shots."""
    import random

    rng = random.Random(42)
    rows = _read(DATA / "train.jsonl")
    rng.shuffle(rows)
    shots = []
    for lab in LABELS:
        shots += [r for r in rows if r["messages"][-1]["content"] == lab][:SHOTS_PER_ROUTE]
    rng.shuffle(shots)
    turns = []
    for r in shots:
        turns += [r["messages"][1], r["messages"][2]]
    return turns


def fuse() -> None:
    cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", MODEL,
           "--adapter-path", str(ADAPTERS), "--save-path", str(FUSED)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=HERE)


def evaluate(target: str, test: str, max_tokens: int, limit: int | None) -> dict:
    from mlx_lm import generate, load

    model_id, adapter, few_shot = TARGETS[target]
    ev = _platform_eval()
    rows = _read(DATA / TEST_FILES[test])[:limit]
    shots = _few_shot_turns() if few_shot else []

    model, tokenizer = load(str(model_id), adapter_path=str(adapter) if adapter else None)
    # One warm-up call so the first row doesn't carry compile/cache cost.
    generate(model, tokenizer, prompt="hi", max_tokens=1, verbose=False)
    preds_raw, golds, lat_ms = [], [], []
    for i, row in enumerate(rows):
        msgs = row["messages"]
        golds.append(msgs[-1]["content"])
        prompt = tokenizer.apply_chat_template(
            [msgs[0], *shots, *msgs[1:-1]],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        t0 = time.perf_counter()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        lat_ms.append((time.perf_counter() - t0) * 1000)
        preds_raw.append(out)
        if (i + 1) % 64 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    preds = [_route_of(p, ev) for p in preds_raw]
    per_class, macro_f1 = _macro_f1(golds, preds)
    correct = sum(g == p for g, p in zip(golds, preds))
    lat = sorted(lat_ms)
    rel = lambda p: str(Path(p).relative_to(HERE)) if Path(p).is_absolute() else str(p)  # noqa: E731
    return {
        "target": target,
        "test_set": test,
        "model": rel(model_id),
        "adapter": rel(adapter) if adapter else None,
        "few_shot_examples": len(shots) // 2,
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
        "predictions": [
            {"user": r["messages"][-2]["content"], "gold": g, "pred": p, "raw": raw}
            for r, g, p, raw in zip(rows, golds, preds, preds_raw)
        ] if test == "handwritten" else None,
    }


def table() -> None:
    """Markdown summary of every metrics/metrics_<target>_<test>.json present."""
    print("| Model | Test set | n | Accuracy | Macro F1 | Unparsed | p50 ms | p90 ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for test in TEST_FILES:
        for target in TARGETS:
            f = METRICS / f"metrics_{target}_{test}.json"
            if f.exists():
                m = json.loads(f.read_text())
                L = m["latency_ms"]
                print(f"| {target} | {test} | {m['count']} | {m['accuracy']:.3f} | "
                      f"{m['macro_f1']:.3f} | {m['unparsed']} | {L['p50']:.0f} | {L['p90']:.0f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["prep", "train", "fuse", "eval", "table"])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="eval only the first N test rows")
    ap.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    ap.add_argument("--tests", nargs="+", choices=list(TEST_FILES), default=list(TEST_FILES))
    a = ap.parse_args()

    if a.cmd == "prep":
        prep()
    elif a.cmd == "train":
        train(a.iters, a.batch_size, a.num_layers, a.max_seq_length, a.learning_rate)
    elif a.cmd == "fuse":
        fuse()
    elif a.cmd == "table":
        table()
    else:
        METRICS.mkdir(exist_ok=True)
        for target in a.targets:
            for test in a.tests:
                print(f"[eval] {target} on {test}", flush=True)
                m = evaluate(target, test, a.max_tokens, a.limit)
                out = METRICS / f"metrics_{target}_{test}.json"
                out.write_text(json.dumps(m, indent=2), encoding="utf-8")
                print(f"  acc={m['accuracy']} macro_f1={m['macro_f1']} "
                      f"p50={m['latency_ms']['p50']}ms unparsed={m['unparsed']}", flush=True)

if __name__ == "__main__":
    main()
