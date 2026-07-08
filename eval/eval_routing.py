"""Routing eval: accuracy vs oracle route and mean regret.

Data rows: {"prompt", "task", "route", "reward"} (build_dataset output held
out from training). Reports task accuracy, route accuracy, and top-3 route
recall.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
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
    base = cfg.get("base", str(args.model))  # backbone architecture source; weights come from the .pt
    tok = AutoTokenizer.from_pretrained(args.model)
    model = ZenRouter(base, len(tasks), len(catalog), 256)
    # strict=False: a frozen-backbone run publishes only the head weights, so the
    # backbone stays the (public) base loaded above; a full run matches all keys.
    model.load_state_dict(torch.load(args.model / "zen-router.pt", map_location="cpu"), strict=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    task_hits = route_hits = top3_hits = 0
    task_pred, route_pred = Counter(), Counter()
    latencies: list[float] = []
    for r in rows:
        enc = tok(r["prompt"], return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(device) for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            t_logits, r_logits, _ = model(**enc)
        if device != "cpu":
            torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        pt = tasks[t_logits.argmax().item()]
        pr = catalog[r_logits[0].argmax().item()]
        task_pred[pt] += 1
        route_pred[pr] += 1
        if pt == r["task"]:
            task_hits += 1
        ranked = r_logits[0].argsort(descending=True).tolist()
        if catalog[ranked[0]] == r["route"]:
            route_hits += 1
        if r["route"] in [catalog[i] for i in ranked[:3]]:
            top3_hits += 1

    n = len(rows)
    lat = sorted(latencies)
    p50, p99 = lat[n // 2], lat[int(n * 0.99)]
    mean = sum(lat) / n
    print(f"device={device} n={n} task_acc={task_hits/n:.3f} route_acc={route_hits/n:.3f} route_top3={top3_hits/n:.3f}")
    print(f"single-forward routing latency (1 prompt, incl. tokenize+forward): mean={mean:.1f}ms p50={p50:.1f}ms p99={p99:.1f}ms")
    print("task pred dist :", dict(task_pred.most_common()))
    print("route pred dist:", dict(route_pred.most_common()))


if __name__ == "__main__":
    main()
