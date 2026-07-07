---
language:
- en
- zh
license: apache-2.0
tags:
- zen
- zen-lm
- routing
- edge
- lightweight
pipeline_tag: text-classification
library_name: transformers
---

# Zen Router 0.6B

**Zen Router** is a tiny trainable routing model: it reads a prompt and picks
the best model to answer it. One forward pass emits the task class, a routing
distribution over the model catalog, and a feature embedding — small enough to
run in front of every request, on every device.

## Model Details

- **Model Type**: Encoder-pooled router (causal LM backbone, routing head)
- **Architecture**: 0.6B dense transformer (Zen Nano lineage)
- **Parameters**: 0.6 billion
- **License**: Apache 2.0
- **Context Length**: 32K tokens
- **Latency target**: <50 ms CPU / <10 ms Metal at Q4
- **Developed by**: Zen AI Team (Hanzo AI)

## What it predicts

One forward pass over the prompt (pooled last hidden state) produces three
heads:

1. **Task** — code, reasoning, math, creative, vision, long-context,
   cheap-chat, general (the `hanzo-router` task taxonomy).
2. **Route** — a distribution over the model catalog, trained against
   realized quality/cost/latency, decoded under the caller's SLO
   (cost ceiling, latency budget, quality floor).
3. **Features** — a compact embedding consumed by per-user adaptation
   (contextual bandit) so routing personalizes online without retraining.

A generative mode is also trained: constrained decode of a single route tag
(`<route model="..." level="fast|balanced|max"/>`) for gateway-side use where
only text-in/text-out is available.

## Training

Two stages (see `training/`):

1. **SFT** — `(prompt → task, best-model)` pairs built from eval profiles and
   gateway telemetry: every routed request settles realized cost, latency,
   and quality back into the ledger, which becomes labels.
2. **Reward tuning** — GRPO on the routing reward
   `quality − λ·cost − μ·latency`, the same objective the serving-side SLO
   uses.

```sh
make data     # build dataset from eval JSONL + usage ledger export
make train    # stage-1 SFT
make grpo     # stage-2 reward tuning
make eval     # routing accuracy / regret vs oracle
make quantize # GGUF Q4_K_M + MLX
```

## Serving & integration

- Served by **hanzo-engine** (OpenAI-compatible, local or cloud) — dense 0.6B
  quantized runs on laptops, phones (Metal/Vulkan), and CPUs.
- **Rust**: implements the `hanzo-router` `Classifier` seam (task head) and
  the `enso` `Featurizer` seam (feature head); `enso`'s per-user LinUCB
  consumes the features, so cold-start falls back to rules and improves
  online.
- **Gateway**: cloud-api resolves the route tag to a provider route
  (primary + fallbacks) and meters it through the billing gate; rate-limit
  and spend snapshots from the usage plane bias decoding away from providers
  near their window limits.

See `docs/integration.md`.

## Formats

PyTorch (safetensors), GGUF (Q2_K–F16), MLX.
