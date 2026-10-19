# SPDX-License-Identifier: MIT-0
"""Self-check for run.py argument parsing and local/cloud dispatch (no I/O).

    uv run python test_run.py
"""

import contextlib
import io

import run


def steps(*argv: str) -> list[str]:
    return run.plan(run.parse_args(list(argv)))


def rejected(*argv: str) -> bool:
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            run.parse_args(list(argv))
        except SystemExit:
            return True
    return False


assert steps("all", "--local") == ["data", "train:local", "fuse:local", "eval:local", "table"]
assert steps("all") == ["data", "train:cloud", "eval:cloud", "table"]  # no fuse in the cloud
assert steps("train") == ["train:cloud"] and steps("train", "--local") == ["train:local"]
assert steps("eval", "--dry-run") == ["eval:cloud"] and steps("eval", "--local") == ["eval:local"]
assert steps("fuse", "--local") == ["fuse:local"]
assert steps("data") == ["data"] and steps("table") == ["table"] and steps("publish") == ["publish"]
assert "publish" not in steps("all") + steps("all", "--local")  # never publish implicitly
assert rejected("fuse")                          # fuse is local-only
assert rejected("train", "--local", "--dry-run")  # dry-run is cloud-only
assert rejected("publish", "--dry-run")
assert set(n for argv in (["all"], ["all", "--local"], ["publish"]) for n in steps(*argv)) <= set(run.DISPATCH)

a = run.parse_args(["train"])
assert (a.region, a.instance_type, a.model, a.local, a.dry_run) == ("us-east-1", "ml.g5.xlarge", None, False, False)
assert run._redact({"Environment": {"HF_TOKEN": "x", "SLM_EVAL_MAX_NEW_TOKENS": "512"}, "Key": "a/b", "Body": b".."}) == \
    {"Environment": {"HF_TOKEN": "<redacted>", "SLM_EVAL_MAX_NEW_TOKENS": "512"}, "Key": "a/b", "Body": "<body omitted>"}
assert rejected("publish", "--out", "/tmp/x")      # never publish smoke artifacts

# local writes into local/ refuse to replace the published (committed) metrics without --force
def guarded(*argv: str) -> bool:
    a = run.parse_args(list(argv))
    try:
        run._guard_published(a, run.plan(a))
    except SystemExit:
        return True
    return False


assert guarded("eval", "--local") and guarded("all", "--local")
assert not guarded("eval", "--local", "--force") and not guarded("all", "--local", "--out", "/tmp/x")
assert not guarded("train") and not guarded("data")  # cloud and data never touch adapters/merged/metrics
print("test_run ok")
