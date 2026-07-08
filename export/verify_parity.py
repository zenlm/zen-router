"""Parity check: torch pipeline vs quantized GGUF + numpy heads.

Runs the SAME heads two ways on holdout prompts and reports agreement:

  torch path : ZenRouter (fp32 backbone) .embed -> heads (the training pipeline)
  gguf path  : llama-server Q4_K_M --pooling last --embd-normalize -1 -> same heads (numpy)

Reports task-argmax agreement %, route-argmax agreement %, route top-3 overlap,
and mean cosine similarity of the pooled hidden states. High route agreement is
the load-bearing number: it says the quantized embedding, fed to the trained
heads, reproduces the routing decision.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from export.route_gguf import GGUFRouter, spawn_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("out/zen-router"))
    ap.add_argument("--heads", type=Path, default=Path("export"))
    ap.add_argument("--data", type=Path, default=Path("data/routing-eval.jsonl"))
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--server", default=None)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from training.sft import ZenRouter  # noqa: PLC0415

    cfg = json.loads((args.model / "router_config.json").read_text())
    tasks, catalog, base = cfg["tasks"], cfg["catalog"], cfg["base"]
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(args.model)
    tmodel = ZenRouter(base, len(tasks), len(catalog), 256)
    tmodel.load_state_dict(torch.load(args.model / "zen-router.pt", map_location="cpu"), strict=False)
    tmodel.eval()

    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    proc = None
    if not args.server:
        proc = spawn_server(args.gguf, args.port)
        server = f"http://127.0.0.1:{args.port}"
    else:
        server = args.server
    try:
        router = GGUFRouter(args.heads, server)
        t_ok = r_ok = top3_ok = 0
        cos = []
        for r in rows:
            enc = tok(r["prompt"], return_tensors="pt", truncation=True, max_length=2048)
            with torch.no_grad():
                x_t = tmodel.embed(**enc)[0].numpy()
                tl_t, rl_t, _ = tmodel.heads(torch.from_numpy(x_t))
            t_task = tasks[int(tl_t.argmax())]
            t_route = catalog[int(rl_t.argmax())]

            x_g = router.embed(r["prompt"])
            tl_g, rl_g, _ = router.heads(x_g)
            g_task = tasks[int(tl_g.argmax())]
            g_route = catalog[int(rl_g.argmax())]
            g_top3 = set(rl_g.argsort()[::-1][:3].tolist())

            t_ok += t_task == g_task
            r_ok += t_route == g_route
            top3_ok += int(rl_t.argmax()) in g_top3
            cos.append(float(np.dot(x_t, x_g) / (np.linalg.norm(x_t) * np.linalg.norm(x_g))))

        n = len(rows)
        print(f"n={n}  base={base}  gguf={args.gguf.name}")
        print(f"task-argmax agreement : {t_ok}/{n} = {t_ok/n:.1%}")
        print(f"route-argmax agreement: {r_ok}/{n} = {r_ok/n:.1%}")
        print(f"torch-route in gguf-top3: {top3_ok}/{n} = {top3_ok/n:.1%}")
        print(f"pooled cosine sim     : mean={np.mean(cos):.4f} min={np.min(cos):.4f}")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
