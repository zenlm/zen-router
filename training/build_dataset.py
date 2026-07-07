"""Build zen-router SFT data from eval profiles and the usage ledger.

Inputs
  --evals   eval rows: {"prompt", "model", "quality", "cost", "latency_ms",
                        "task"?}  (one JSON object per line)
  --ledger  routed-request telemetry with the same fields, exported from the
            billing/usage ledger

Output: {"prompt", "task", "route", "reward"} per line — `route` is the
argmax-reward model for that prompt; rewards use the serving objective
quality - lambda*cost - mu*latency.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LAMBDA_COST = 0.35
MU_LATENCY = 0.15


def reward(row: dict) -> float:
    return (
        float(row.get("quality", 0.0))
        - LAMBDA_COST * float(row.get("cost", 0.0))
        - MU_LATENCY * float(row.get("latency_ms", 0.0)) / 1000.0
    )


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in load(args.evals) + load(args.ledger):
        if row.get("prompt") and row.get("model"):
            by_prompt[row["prompt"]].append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as f:
        for prompt, rows in by_prompt.items():
            best = max(rows, key=reward)
            f.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "task": best.get("task", "general"),
                        "route": best["model"],
                        "reward": round(reward(best), 6),
                    }
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} examples -> {args.out}")


if __name__ == "__main__":
    main()
