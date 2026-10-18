# SPDX-License-Identifier: MIT-0
"""One entry point for the Class 1 Decider router: local (MLX) or cloud (SageMaker).

    uv run python run.py data                     # dataset (same in both modes)
    uv run python run.py train --local            # mlx-lm LoRA on this Mac
    uv run python run.py train                    # SageMaker job, LLaMA-Factory image
    uv run python run.py fuse --local             # merge adapter -> merged/
    uv run python run.py eval --local             # MLX eval matrix -> metrics/
    uv run python run.py eval                     # SageMaker vLLM eval + Bedrock baselines
    uv run python run.py table                    # summary of local (+ fetched cloud) metrics
    uv run python run.py publish                  # push model + dataset to HF (private)
    uv run python run.py all [--local]            # data -> train -> (fuse) -> eval -> table
    uv run python run.py train --dry-run          # print the AWS requests, send nothing

Wraps, rather than re-implements: the local steps are gen_router_data.py and
mac_router.py; the cloud steps are backend/app/orchestrate.py (training + eval
jobs), backend/app/storage.py (split persistence) and backend/app/baseline.py
(Bedrock baselines). Cloud mode needs `uv sync --extra cloud` and a deployed
stack (see the README "Commands" section).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
CLOUD_STATE = HERE / "cloud"  # job names from the last cloud train/eval (gitignored)
DEFAULT_CLOUD_MODEL = "qwen3-0.6b"  # backend/app/catalog.py id
DEFAULT_BASELINES = ["haiku-4-5", "nova-micro"]  # backend/app/baseline.py keys
HF_ORG = "SolvrenAI"
HF_MODEL_REPO = f"{HF_ORG}/class1-router-qwen3-0.6b-mlx"
HF_DATA_REPO = f"{HF_ORG}/class1-router-synthetic"

STEPS = ["data", "train", "fuse", "eval", "table", "publish", "all"]


# ----------------------------------------------------------------- arguments ---

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--local", action="store_true", help="run on this Mac with MLX")
    shared.add_argument("--dry-run", action="store_true",
                        help="cloud only: print every AWS request, send nothing")
    shared.add_argument("--model", default=None,
                        help="local: MLX/HF repo id; cloud: catalog id (default qwen3-0.6b)")
    shared.add_argument("--region", default="us-east-1")
    shared.add_argument("--instance-type", default="ml.g5.xlarge")
    # Smoke-run knobs (defaults reproduce the published run).
    shared.add_argument("--iters", type=int, default=400, help="local LoRA iterations")
    shared.add_argument("--limit", type=int, default=None, help="eval only the first N rows")
    shared.add_argument("--out", default=None,
                        help="local: write adapters/merged/metrics here instead of local/")
    shared.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES,
                        help="cloud eval: Bedrock baseline keys (backend/app/baseline.py)")
    shared.add_argument("--env-file", default=str(REPO.parent / ".env"),
                        help="publish: file holding HF_TOKEN (read into memory only)")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in STEPS:
        sub.add_parser(name, parents=[shared])
    a = ap.parse_args(argv)

    if a.dry_run and a.local:
        ap.error("--dry-run applies to the cloud path only; drop --local")
    if a.step == "fuse" and not a.local:
        ap.error("fuse is local-only (--local); the cloud job merges in-container via export.yaml")
    if a.step == "publish" and a.dry_run:
        ap.error("publish talks to Hugging Face, not AWS; --dry-run does not apply")
    return a


def plan(a: argparse.Namespace) -> list[str]:
    """The concrete steps a command runs, e.g. ['data', 'train:local', ...]."""
    mode = "local" if a.local else "cloud"
    if a.step == "all":
        steps = ["data", f"train:{mode}"] + (["fuse:local"] if a.local else []) + [f"eval:{mode}", "table"]
        return steps  # publish is deliberately NOT part of `all`
    if a.step in ("train", "eval"):
        return [f"{a.step}:{mode}"]
    if a.step == "fuse":
        return ["fuse:local"]
    return [a.step]


# ------------------------------------------------------------- local steps ---

def _local_env(a: argparse.Namespace) -> None:
    """Point mac_router at --model/--out. Must run before mac_router is imported."""
    if a.out:
        os.environ["CLASS1_OUT"] = str(Path(a.out).resolve())
    if a.model and a.local:
        os.environ["CLASS1_MODEL"] = a.model


def step_data(a: argparse.Namespace) -> None:
    import gen_router_data
    import mac_router

    gen_router_data.main()
    mac_router.prep()


def step_train_local(a: argparse.Namespace) -> None:
    import mac_router

    mac_router.train(a.iters, batch_size=2, num_layers=8, max_seq_len=512, lr=1e-4)


def step_fuse_local(a: argparse.Namespace) -> None:
    import mac_router

    mac_router.fuse()


def step_eval_local(a: argparse.Namespace) -> None:
    import mac_router

    mac_router.run_eval(list(mac_router.TARGETS), list(mac_router.TEST_FILES), 32, a.limit)


def step_table(a: argparse.Namespace) -> None:
    import mac_router

    mac_router.table()
    if not a.local and (CLOUD_STATE / "eval.json").exists():
        _cloud_table(a)


# ------------------------------------------------------------- cloud steps ---

def _backend(a: argparse.Namespace):
    """Import the platform backend with the CLI's region applied."""
    os.environ["SLM_AWS_REGION"] = a.region
    os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")
    if str(REPO / "backend") not in sys.path:
        sys.path.insert(0, str(REPO / "backend"))
    from app import baseline, catalog, orchestrate, storage

    return orchestrate, storage, catalog, baseline


def _persist_split(a: argparse.Namespace) -> str:
    """Our train/valid/test as a backend split (train / val / held-out eval)."""
    _, storage, _, _ = _backend(a)
    import mac_router

    rows = {n: mac_router._read(mac_router.MLX_DATA / f"{n}.jsonl") for n in ("train", "valid", "test")}
    split_id, _ = storage.persist_split(
        rows["train"], rows["test"], {"source": "class1-router run.py"}, val=rows["valid"]
    )
    return split_id


def _save_state(name: str, data: dict) -> None:
    CLOUD_STATE.mkdir(exist_ok=True)
    (CLOUD_STATE / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_state(name: str) -> dict:
    f = CLOUD_STATE / f"{name}.json"
    if not f.exists():
        sys.exit(f"no {f.name}: run `run.py train` (cloud) first")
    return json.loads(f.read_text())


def step_train_cloud(a: argparse.Namespace) -> None:
    orchestrate, _, catalog, _ = _backend(a)
    model_id = a.model or DEFAULT_CLOUD_MODEL
    if catalog.get_model(model_id) is None:
        sys.exit(f"unknown catalog model {model_id!r} (see backend/app/catalog.py)")
    split_id = _persist_split(a)
    hp = catalog.Hyperparams(num_train_epochs=1.0, cutoff_len=512, learning_rate=1e-4)
    job = orchestrate.launch_training_job(
        model_id, split_id, hp, a.instance_type, stamp=time.strftime("%Y%m%d-%H%M%S"),
    )
    print(json.dumps(job, indent=2, default=str))
    if not a.dry_run:
        _save_state("train", {"jobName": job["jobName"], "splitId": split_id, "model": model_id})


def step_eval_cloud(a: argparse.Namespace) -> None:
    orchestrate, _, catalog, baseline = _backend(a)
    state = _load_state("train") if not a.dry_run else {
        "jobName": "slm-dry-run-train", "splitId": _persist_split(a), "model": a.model or DEFAULT_CLOUD_MODEL}
    decoding = catalog.DecodingParams(max_new_tokens=32)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # Block until the training job finishes (immediate in --dry-run).
    sm = orchestrate.get_session(region_name=a.region).client("sagemaker")
    print(f"[eval] waiting for {state['jobName']} ...", flush=True)
    sm.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=state["jobName"])

    tuned = orchestrate.launch_eval_job(state["jobName"], state["splitId"], decoding, stamp,
                                        instance_type=a.instance_type)
    base = orchestrate.launch_base_eval_job(state["model"], state["splitId"], decoding, stamp,
                                            instance_type=a.instance_type)
    print(json.dumps({"tuned": tuned, "base": base}, indent=2, default=str))
    baselines = {}
    for key in a.baselines:  # Bedrock calls run here, row by row
        m = baseline.run_sonnet_baseline(state["splitId"], max_new_tokens=32, baseline_key=key)
        baselines[key] = {k: m.get(k) for k in ("count", "normalized_match", "exact_match")}
        print(f"[baseline] {key}: {baselines[key]}" + (" (canned dry-run responses)" if a.dry_run else ""))
    if not a.dry_run:
        _save_state("eval", {"splitId": state["splitId"], "tuned": tuned["jobName"],
                             "base": base["jobName"], "baselines": list(baselines)})


def _cloud_table(a: argparse.Namespace) -> None:
    orchestrate, _, _, baseline = _backend(a)
    st = _load_state("eval")
    # normalized_match == accuracy for one-word route labels; both eval.py and
    # baseline.py emit it, so the rows are comparable.
    print("\n| Cloud model | Accuracy (normalized match) | Exact match |\n| --- | --- | --- |")
    rows = [(n, orchestrate.fetch_metrics(st[n])) for n in ("base", "tuned")]
    rows += [(k, baseline.load_baseline(st["splitId"], k)) for k in st["baselines"]]
    for name, m in rows:
        if m is None:
            print(f"| {name} | (not finished) | |")
        else:
            print(f"| {name} | {m.get('normalized_match')} | {m.get('exact_match')} |")


# ----------------------------------------------------------------- dry run ---

_CANNED = {
    "GetCallerIdentity": {"Account": "000000000000", "Arn": "arn:aws:iam::000000000000:user/dry-run",
                          "UserId": "DRYRUN"},
    "CreateTrainingJob": {"TrainingJobArn": "arn:aws:sagemaker:dry-run:000000000000:training-job/dry-run"},
    "DescribeTrainingJob": {"TrainingJobStatus": "Completed",
                            "ModelArtifacts": {"S3ModelArtifacts": "s3://dry-run/output/model.tar.gz"},
                            "AlgorithmSpecification": {"TrainingImage": "dry-run-image"}},
    "Converse": {"output": {"message": {"role": "assistant", "content": [{"text": "small-model"}]}},
                 "usage": {"inputTokens": 0, "outputTokens": 0}},
}
_SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY", "API_KEY")


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("<redacted>" if any(w in str(k).upper() for w in _SECRET_WORDS) and isinstance(v, str)
                    else "<body omitted>" if k == "Body" else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


@contextmanager
def dry_run(enabled: bool):
    """Intercept every botocore API call: print the request, return a canned
    response. A second guard blocks every urllib3 connection (botocore, requests,
    SDK telemetry), so nothing leaves the machine. Throwaway credentials stop the
    credential chain before any SSO/login refresh; local state goes to a temp dir."""
    if not enabled:
        yield
        return
    import tempfile

    import botocore.client
    import urllib3.connectionpool

    seen: dict[str, int] = {}
    blocked: list[str] = []
    orig_call = botocore.client.BaseClient._make_api_call
    orig_urlopen = urllib3.connectionpool.HTTPConnectionPool.urlopen

    def fake_call(self, operation_name, api_params):
        op = f"{self.meta.service_model.service_name}.{operation_name}"
        seen[op] = seen.get(op, 0) + 1
        if seen[op] == 1 or operation_name.startswith("Create"):  # every job; first of the rest
            print(f"[dry-run] {op}\n{json.dumps(_redact(api_params), indent=2, default=str)}", flush=True)
        return _CANNED.get(operation_name, {})

    def no_network(pool, method, url, *args, **kwargs):
        blocked.append(f"{method} {pool.host}")
        raise RuntimeError(f"dry-run: network blocked ({method} {pool.host})")

    env = {"SLM_DATA_DIR": tempfile.mkdtemp(prefix="class1-dryrun-"), "SLM_AWS_PROFILE": "",
           "AWS_ACCESS_KEY_ID": "DRYRUN", "AWS_SECRET_ACCESS_KEY": "DRYRUN",
           "AWS_SESSION_TOKEN": None, "AWS_PROFILE": None, "AWS_DEFAULT_PROFILE": None}  # None = unset
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    botocore.client.BaseClient._make_api_call = fake_call
    urllib3.connectionpool.HTTPConnectionPool.urlopen = no_network
    try:
        yield
    finally:
        botocore.client.BaseClient._make_api_call = orig_call
        urllib3.connectionpool.HTTPConnectionPool.urlopen = orig_urlopen
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        print("[dry-run] AWS requests intercepted, none sent: "
              + (", ".join(f"{op} x{n}" for op, n in seen.items()) or "none"), flush=True)
        print(f"[dry-run] other network attempts blocked: {len(blocked)}"
              + (f" ({', '.join(sorted(set(blocked)))})" if blocked else ""), flush=True)


# ----------------------------------------------------------------- publish ---

def _load_hf_token(env_file: str) -> None:
    """HF_TOKEN from the environment, else from env_file — into os.environ only."""
    if os.environ.get("HF_TOKEN"):
        return
    p = Path(env_file)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")
                return
    sys.exit(f"HF_TOKEN not set and not found in {env_file}")


def step_publish(a: argparse.Namespace) -> None:
    import shutil
    import tempfile

    import mac_router
    from huggingface_hub import HfApi

    _load_hf_token(a.env_file)
    api = HfApi()
    if HF_ORG not in [o.get("name") for o in api.whoami().get("orgs", [])]:
        sys.exit(f"token's user is not a member of {HF_ORG}")

    with tempfile.TemporaryDirectory() as tmp:
        model_dir, data_dir = Path(tmp, "model"), Path(tmp, "data")
        shutil.copytree(mac_router.FUSED, model_dir, ignore=shutil.ignore_patterns("README.md"))
        (model_dir / "adapter").mkdir()
        shutil.copy(mac_router.ADAPTERS / "adapters.safetensors", model_dir / "adapter")
        cfg = json.loads((mac_router.ADAPTERS / "adapter_config.json").read_text())
        cfg.update(adapter_path="adapter", data=HF_DATA_REPO)  # drop local absolute paths
        (model_dir / "adapter" / "adapter_config.json").write_text(json.dumps(cfg, indent=4))
        shutil.copy(HERE / "MODEL_CARD.md", model_dir / "README.md")
        data_dir.mkdir()
        for n in ("train", "valid", "test", "handwritten_test"):
            shutil.copy(mac_router.DATA / f"{n}.jsonl", data_dir)
        shutil.copy(HERE / "DATASET_CARD.md", data_dir / "README.md")

        for repo_id, rtype, folder in [(HF_DATA_REPO, "dataset", data_dir), (HF_MODEL_REPO, "model", model_dir)]:
            api.create_repo(repo_id, repo_type=rtype, private=True, exist_ok=True)
            if not api.repo_info(repo_id, repo_type=rtype).private:
                sys.exit(f"{repo_id} exists and is PUBLIC; refusing to upload")
            api.upload_folder(repo_id=repo_id, repo_type=rtype, folder_path=str(folder),
                              commit_message="Update from run.py publish")
            print(f"published (private): https://huggingface.co/{'datasets/' if rtype == 'dataset' else ''}{repo_id}")


# ------------------------------------------------------------------- main ---

DISPATCH = {
    "data": step_data,
    "train:local": step_train_local,
    "train:cloud": step_train_cloud,
    "fuse:local": step_fuse_local,
    "eval:local": step_eval_local,
    "eval:cloud": step_eval_cloud,
    "table": step_table,
    "publish": step_publish,
}


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    _local_env(a)
    sys.path.insert(0, str(HERE))
    with dry_run(a.dry_run):
        for name in plan(a):
            print(f"== {name}", flush=True)
            DISPATCH[name](a)


if __name__ == "__main__":
    main()
