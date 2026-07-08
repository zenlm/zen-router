"""Export the trained heads-only checkpoint to safetensors for quantized serving.

The deployment recipe splits the router in two:

  * backbone  -> runs quantized in llama.cpp EMBEDDING mode. The last-token
    pooled hidden state (`--pooling last --embd-normalize -1`, i.e. the raw
    post-final-norm hidden state, NOT L2-normalized) reproduces the torch
    `ZenRouter.embed` output to within quantization error.
  * heads     -> three tiny linear maps (task/route/feature) applied by the
    caller in numpy. A 1024x28 matmul is microseconds; no torch at serve time.

This script reads `out/zen-router/zen-router.pt` (head weights only, from the
frozen-backbone run) and writes `heads.safetensors` + `router_config.json` next
to it (or to --out). Nothing here touches the backbone: it stays the public
`zenlm/zen-nano-0.6b` GGUF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

HEAD_KEYS = (
    "task_head.weight", "task_head.bias",
    "route_head.weight", "route_head.bias",
    "feature_head.weight", "feature_head.bias",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("out/zen-router"),
                    help="dir with zen-router.pt + router_config.json")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: export/)")
    args = ap.parse_args()

    out = args.out or Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)

    sd = torch.load(args.model / "zen-router.pt", map_location="cpu")
    missing = [k for k in HEAD_KEYS if k not in sd]
    if missing:
        raise SystemExit(f"checkpoint missing head keys {missing}; not a heads-only run?")
    if any(k.startswith("backbone.") for k in sd):
        raise SystemExit("checkpoint carries backbone weights; expected a --freeze-backbone run")

    heads = {k: sd[k].contiguous().float() for k in HEAD_KEYS}
    save_file(heads, str(out / "heads.safetensors"))

    cfg = json.loads((args.model / "router_config.json").read_text())
    hidden = heads["route_head.weight"].shape[1]
    cfg["hidden_size"] = hidden
    cfg["feature_dim"] = heads["feature_head.weight"].shape[0]
    cfg["pooling"] = "last"
    cfg["embd_normalize"] = -1  # raw hidden state; heads carry their own scale
    (out / "router_config.json").write_text(json.dumps(cfg, indent=2))

    total = sum(v.numel() for v in heads.values())
    print(f"wrote {out/'heads.safetensors'}  ({total} params, "
          f"{(out/'heads.safetensors').stat().st_size} bytes)")
    print(f"wrote {out/'router_config.json'}  "
          f"(hidden={hidden}, tasks={len(cfg['tasks'])}, routes={len(cfg['catalog'])})")


if __name__ == "__main__":
    main()
