"""End-to-end quantized routing: prompt in -> model id out.

Backbone runs quantized in llama.cpp (llama-server, EMBEDDING mode, resident);
the three linear heads run in numpy in this process. This is the serving path
the README's Deployment section measures.

  # one shot (auto-spawns a llama-server on the GGUF, tears it down after):
  python export/route_gguf.py --gguf /path/zen-nano-0.6b-Q4_K_M.gguf \
      --prompt "write a python function to reverse a linked list"

  # against an already-running server:
  python export/route_gguf.py --server http://127.0.0.1:8899 --prompt "..."

  # latency benchmark over N holdout prompts (server must stay resident):
  python export/route_gguf.py --gguf .../Q4_K_M.gguf --bench data/routing-eval.jsonl --n 200

The heads were trained on the raw post-final-norm last-token hidden state, so the
server MUST run `--pooling last --embd-normalize -1` (raw, not L2-normalized);
the auto-spawn path sets this. Feeding a normalized vector silently breaks the
bias term of the linear heads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


class GGUFRouter:
    def __init__(self, heads_dir: Path, server: str):
        cfg = json.loads((heads_dir / "router_config.json").read_text())
        self.tasks: list[str] = cfg["tasks"]
        self.catalog: list[str] = cfg["catalog"]
        h = load_file(str(heads_dir / "heads.safetensors"))
        self.Wt, self.bt = h["task_head.weight"], h["task_head.bias"]
        self.Wr, self.br = h["route_head.weight"], h["route_head.bias"]
        self.Wf, self.bf = h["feature_head.weight"], h["feature_head.bias"]
        self.server = server.rstrip("/")

    def embed(self, prompt: str) -> np.ndarray:
        req = urllib.request.Request(
            f"{self.server}/embedding",
            data=json.dumps({"content": prompt}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            d = json.load(r)
        e = d[0]["embedding"] if isinstance(d, list) else d["embedding"]
        e = e[0] if isinstance(e[0], list) else e  # pooled: single vector
        return np.asarray(e, dtype=np.float32)

    def heads(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.Wt @ x + self.bt, self.Wr @ x + self.br, self.Wf @ x + self.bf

    def route(self, prompt: str) -> dict:
        x = self.embed(prompt)
        tl, rl, _ = self.heads(x)
        top3 = rl.argsort()[::-1][:3]
        return {
            "task": self.tasks[int(tl.argmax())],
            "model": self.catalog[int(rl.argmax())],
            "top3": [self.catalog[i] for i in top3],
        }


def wait_ready(url: str, timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    raise TimeoutError(f"llama-server not ready at {url}")


def spawn_server(gguf: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        # -b/-ub must cover the longest prompt: a pooled ('last') embedding needs
        # the whole sequence in one ubatch, else the slot dies mid-request.
        ["llama-server", "-m", str(gguf), "--embeddings", "--pooling", "last",
         "--embd-normalize", "-1", "-c", "4096", "-b", "4096", "-ub", "4096",
         "-ngl", "99", "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wait_ready(f"http://127.0.0.1:{port}")
    return proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=Path, default=Path(__file__).resolve().parent,
                    help="dir with heads.safetensors + router_config.json")
    ap.add_argument("--gguf", type=Path, default=None, help="GGUF to auto-spawn a server on")
    ap.add_argument("--server", default=None, help="URL of a running llama-server (embedding mode)")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--bench", type=Path, default=None, help="jsonl of {'prompt':...} to time")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    proc = None
    try:
        if args.server:
            server = args.server
        elif args.gguf:
            proc = spawn_server(args.gguf, args.port)
            server = f"http://127.0.0.1:{args.port}"
        else:
            raise SystemExit("pass --server URL or --gguf PATH")

        router = GGUFRouter(args.heads, server)

        if args.prompt:
            print(json.dumps(router.route(args.prompt)))

        if args.bench:
            rows = [json.loads(l) for l in args.bench.read_text().splitlines() if l.strip()]
            rows = rows[: args.n]
            router.route(rows[0]["prompt"])  # warm the slot
            lat = []
            for r in rows:
                t0 = time.perf_counter()
                router.route(r["prompt"])
                lat.append((time.perf_counter() - t0) * 1000.0)
            lat.sort()
            n = len(lat)
            mean = sum(lat) / n
            print(f"quantized routing latency (Q4_K_M, resident llama-server, "
                  f"prompt->model id, n={n}): "
                  f"mean={mean:.1f}ms p50={lat[n//2]:.1f}ms "
                  f"p99={lat[min(int(n*0.99), n-1)]:.1f}ms "
                  f"min={lat[0]:.1f}ms max={lat[-1]:.1f}ms")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
