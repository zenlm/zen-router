import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_dataset_picks_argmax_reward(tmp_path: Path) -> None:
    evals = tmp_path / "evals.jsonl"
    rows = [
        {"prompt": "write a fib function", "model": "zen-coder-24b", "quality": 0.9, "cost": 0.0, "latency_ms": 800, "task": "code"},
        {"prompt": "write a fib function", "model": "hanzo-max", "quality": 0.95, "cost": 3.0, "latency_ms": 2000, "task": "code"},
    ]
    evals.write_text("\n".join(json.dumps(r) for r in rows))
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")
    out = tmp_path / "out.jsonl"

    subprocess.run(
        [sys.executable, str(ROOT / "training/build_dataset.py"),
         "--evals", str(evals), "--ledger", str(ledger), "--out", str(out)],
        check=True,
    )
    got = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(got) == 1
    # 0.9 - 0.15*0.8 = 0.78 beats 0.95 - 0.35*3 - 0.15*2 = -0.4
    assert got[0]["route"] == "zen-coder-24b"
    assert got[0]["task"] == "code"
