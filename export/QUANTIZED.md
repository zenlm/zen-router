# Quantized deployment — Zen Router

The router is a **quantized backbone + tiny numpy heads**. The backbone runs in
`llama.cpp` embedding mode (the public `zenlm/zen-nano-0.6b` GGUF); the three
trained linear heads (`heads.safetensors`, 1.2 MB) are applied by the caller in
numpy. Nothing torch is needed at serve time.

```
prompt ──▶ llama.cpp (zen-nano Q4_K_M, --pooling last --embd-normalize -1)
             └─▶ raw last-token hidden state x  (1024-d, NOT L2-normalized)
                   └─▶ numpy:  route = argmax(W_route · x + b_route)
                               task  = argmax(W_task  · x + b_task)
                               feat  = W_feat · x + b_feat   (for the online bandit)
```

## Why raw (un-normalized) embeddings are mandatory

The heads were trained on `ZenRouter.embed` = the **raw** post-final-norm
last-token hidden state (norm ≈ 100). The head bias `b` is calibrated to that
scale. If you feed an L2-normalized vector (norm = 1), `W·x` shrinks ~100× while
`b` does not, so the bias swamps the signal and every prompt collapses to one
label. Always run llama.cpp with `--embd-normalize -1`.

`llama.cpp` emits the same hidden state HF's `AutoModel` returns as
`last_hidden_state` (post-RMSNorm). Verified on the f16 GGUF: torch norm 101.4 /
llama norm 102.7, matching signs and magnitudes element-wise.

## Files

| file | what |
|------|------|
| `heads.safetensors` | 6 tensors: `{task,route,feature}_head.{weight,bias}` (299,300 params, 1.2 MB) |
| `router_config.json` | `base`, `tasks` (8), `catalog` (28), `hidden_size`, `pooling`, `embd_normalize` |
| `export_heads.py` | regenerate the two files above from `out/zen-router/zen-router.pt` |
| `route_gguf.py` | end-to-end routing CLI (prompt → model id) + latency benchmark |
| `verify_parity.py` | torch pipeline vs GGUF+heads agreement check |

## Serve

`llama.cpp` must be Metal-enabled (`ggml_metal_device_init: GPU name … Apple M4
Max`). A pooled (`last`) embedding needs the whole sequence in **one** ubatch, so
`-b`/`-ub` must cover the longest prompt:

```sh
llama-server -m zen-nano-0.6b-Q4_K_M.gguf \
    --embeddings --pooling last --embd-normalize -1 \
    -c 4096 -b 4096 -ub 4096 -ngl 99 --port 8899

python export/route_gguf.py --server http://127.0.0.1:8899 \
    --prompt "write a python function to reverse a linked list"
# {"task": "cheap_chat", "model": "claude-opus-4-5", "top3": [...]}
```

`route_gguf.py --gguf <path>` auto-spawns/tears down a server if you don't have
one running.

## Measured — Apple M4 Max, 128 GB, macOS 26.5

Backbone `zen-nano-0.6b` **Q4_K_M** (397 MB), `llama.cpp` build 9430 (Metal).
Heads in numpy. Reference torch path: fp32 backbone on MPS.

### Parity (torch pipeline vs GGUF + same heads), 50-prompt holdout classes

Agreement of the routing decision between the fp32-torch pipeline and the
quantized GGUF+heads pipeline, per quant:

| quant | size | task argmax | **route argmax** | route top-3 | pooled cosine |
|-------|-----:|:-----------:|:----------------:|:-----------:|:-------------:|
| Q4_K_M | 397 MB | 89% | **95%** | **100%** | 0.969 |
| Q8_0   | 639 MB | 95% | **97%** | **100%** | 0.988 |

(n=100 holdout prompts, seed 0.) The quantized backbone reproduces the torch
routing decision 95% of the time at Q4 and never drops the torch pick out of the
top-3. Higher precision (Q8_0) recovers 2 more points and a tighter cosine, at
1.6× the footprint. **Q4_K_M is the default; Q8_0 for fidelity-sensitive use.**

### Latency — quantized routing, prompt → model id (Q4_K_M, resident server)

Full held-out prompt mix (n=200; token lengths min 3 / p50 159 / mean 405 / max
1373):

```
mean 132.2 ms   p50 96.9 ms   p99 350.9 ms   min 12.5 ms   max 358.1 ms
```

Stratified by prompt length (unique prompts, cold — each routing request is new):

| prompt tokens | n | p50 |
|---------------|--:|----:|
| 1–32   | 59 |  50 ms |
| 33–128 | 31 |  78 ms |
| 129–512| 37 | 107 ms |
| 513+   | 73 | 251 ms |

Latency is Metal-dispatch + HTTP fixed cost (~40–50 ms floor on this 0.6B model)
plus linear prompt-processing; the p50 (~97 ms) tracks the earlier `llama-bench`
`pp1024 ≈ 112 ms` figure. For reference the **fp32 torch/MPS** path measured p50
57 ms on the same holdout — for a model this small, in-process MPS matmuls beat a
Q4 llama.cpp **HTTP round-trip**, so the quantized path's win here is
**footprint** (397 MB vs 2.4 GB) and **portability** (CPU / edge / Vulkan, no
torch), not raw M4-Max latency. On a caching hit (repeated prompt) the same
mid-size request drops to ~5–21 ms.

Reproduce:

```sh
python export/export_heads.py
llama-server -m .../zen-nano-0.6b-Q4_K_M.gguf --embeddings --pooling last \
    --embd-normalize -1 -c 4096 -b 4096 -ub 4096 -ngl 99 --port 8899 &
python -m export.verify_parity --gguf .../zen-nano-0.6b-Q4_K_M.gguf \
    --server http://127.0.0.1:8899 --n 100
python export/route_gguf.py --server http://127.0.0.1:8899 \
    --bench data/routing-eval.jsonl --n 200
```
