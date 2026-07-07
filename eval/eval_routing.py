"""Routing eval: accuracy vs oracle route and mean regret.

Data rows: {"prompt", "task", "route", "reward"} (build_dataset output held
out from training). Reports task accuracy, route accuracy, and top-3 route
recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()

    from training.sft import ZenRouter  # noqa: PLC0415

    cfg = json.loads((args.model / "router_config.json").read_text())
    tasks, catalog = cfg["tasks"], cfg["catalog"]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = ZenRouter(str(args.model), len(tasks), len(catalog), 256)
    model.load_state_dict(torch.load(args.model / "zen-router.pt", map_location="cpu"))
    model.eval()

    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    task_hits = route_hits = top3_hits = 0
    for r in rows:
        enc = tok(r["prompt"], return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            t_logits, r_logits, _ = model(**enc)
        if tasks[t_logits.argmax().item()] == r["task"]:
            task_hits += 1
        ranked = r_logits[0].argsort(descending=True).tolist()
        if catalog[ranked[0]] == r["route"]:
            route_hits += 1
        if r["route"] in [catalog[i] for i in ranked[:3]]:
            top3_hits += 1

    n = len(rows)
    print(f"n={n} task_acc={task_hits/n:.3f} route_acc={route_hits/n:.3f} route_top3={top3_hits/n:.3f}")


if __name__ == "__main__":
    main()
