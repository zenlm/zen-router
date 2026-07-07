"""Stage-2 reward tuning (GRPO) on the routing objective.

Reward per sampled route: quality - lambda*cost - mu*latency, settled from
the usage ledger for that (prompt, model) pair. Uses the zenlm grpo trainer;
this file only adapts the routing task to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    g = cfg["grpo"]

    try:
        from grpo import RoutingRewardTrainer  # zenlm/grpo
    except ImportError as e:
        raise SystemExit(
            "zenlm/grpo not installed — `uv pip install -e ../grpo` from the zen workspace"
        ) from e

    trainer = RoutingRewardTrainer(
        model_path=cfg["output_dir"],
        lambda_cost=g["lambda_cost"],
        mu_latency=g["mu_latency"],
        group_size=g["group_size"],
        lr=float(g["lr"]),
    )
    trainer.train("data/routing-sft.jsonl")
    trainer.save(cfg["output_dir"])


if __name__ == "__main__":
    main()
