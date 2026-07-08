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

## Measured

A real SFT run on this repo's own tooling, on real local AI-assistant usage.
Everything below came from a command that actually ran; nothing is projected.

**Hardware / stack.** Apple M4 Max, 128 GB unified memory, macOS 26.5. PyTorch
2.12.1 + Transformers 5.13.0, training and eval on the Metal (MPS) backend,
fp32. Backbone: on-disk `zen-nano-0.6b` (Qwen3, 596M params, hidden 1024, 28
layers).

**Dataset (provenance).** Built by `training/extract_local_sessions.py` from
this machine's actual assistant logs:
`~/.claude/projects/**/*.jsonl` (Claude Code) and `~/.codex/sessions/**/*.jsonl`
(Codex). Each user prompt is paired with the model that actually served it;
`quality` is a fixed **proxy = 1.0** (we cannot observe the counterfactual
quality of models that did not run), `cost` = served tokens x the catalog's
`cost_per_1k`, `latency_ms` = 0 (not recorded in the logs), and `task` comes
from a keyword heuristic that is a direct port of `hanzo-router`'s
`Heuristic::classify`. `training/build_dataset.py` then collapses to one
max-reward row per unique prompt.

- 5,975 raw eval rows -> **5,455 unique-prompt routing rows** -> 80/20 split =
  **4,364 train / 1,091 holdout**.

**Class balance (the load-bearing caveat).** This machine runs Claude Code on
Opus almost exclusively, so the labels are severely imbalanced:

- Route label: `claude-opus-4-8` **94.6%**, `claude-haiku-4-5-20251001` 4.9%,
  `claude-fable-5` 0.2%, `gpt-5.5` 0.2%, `claude-sonnet-4-6` 0.1%. The other 19
  catalog models have **zero** training rows.
- Task label: code 59.6%, general 24.0%, cheap_chat 8.5%, math 5.3%,
  reasoning 2.6%, long_context 0.07%, and vision/creative 0%.

**Catalog & route head.** `training/catalog.yaml` is the routable universe: 3
local zen models + 21 cloud models (zen cloud tiers + Anthropic / OpenAI /
Gemini / Grok / DeepSeek / Kimi), priced from the gateway's live config
(`hanzoai/ai/conf/models.yaml`, refreshed from pricing.hanzo.ai). Because the
route head sizes from the catalog (`classes_from: catalog`), the SFT run printed
`route head: 24 classes` (up from 7 in the original scaffold) with no code
change -- adding a routable model is one catalog row.

**Training.** 1 epoch, batch 16, seq 512, AdamW lr 2e-5, ~17 min wall on MPS;
two-head cross-entropy loss fell 7.23 -> ~0.7-1.1.

**Accuracy (1,091-prompt holdout).**

| metric | value | honest reading |
|--------|------:|----------------|
| task_acc   | **0.824** | the meaningful signal: it learned the task taxonomy (predicts code/general/cheap_chat/math; never predicts the near-absent reasoning/creative/vision/long_context) |
| route_acc  | **0.961** | barely above the 94.2% majority-class baseline; the head only ever emits 2 of 24 models (`claude-opus-4-8` x1029, `claude-haiku-4-5-20251001` x62) because the other 22 have no data |
| route_top3 | **0.999** | trivially high -- Opus is in the top-3 for essentially every prompt |

Route accuracy is **not** evidence of good multi-model routing here -- it is
evidence that the model memorized the dominant label. The route head is
structurally correct (24 logits, gateway-priced catalog) and the mechanism is
proven; teaching it to route across providers needs balanced data (a gateway
telemetry export where many models actually serve), not more local Opus logs.

**Single-forward routing latency.** One prompt, tokenize + forward, MPS fp32:
**mean 137.3 ms, p50 130.3 ms, p99 384.7 ms**. This is unquantized eager
Transformers; the `<50 ms CPU / <10 ms Metal` target is at Q4 (see
`make quantize`, not run here). For reference, `llama.cpp` prompt-processing of
the same `zen-nano-0.6b` at Q4_K_M on this box runs 9,092 tok/s (a 1k-token
forward ~= 112 ms).

**Mechanism vs. model.** The Rust routing mechanism this model plugs into
(`hanzo-router` + `enso`) decides in **1.6 us** (rules) to **16 us** (learned
featurize + bilinear select) -- 4-5 orders of magnitude below one 0.6B forward,
so the encoder-mode router forward dominates end-to-end cost, as designed.

Reproduce: `python training/extract_local_sessions.py` -> `make data` (+ 80/20
split) -> `python training/sft.py --config training/config.local.yaml` ->
`python -m eval.eval_routing --model out/zen-router --data data/routing-eval.jsonl`.

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
