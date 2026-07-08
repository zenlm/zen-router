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

Two real SFT runs on this repo's tooling, on real AI-assistant usage. Every
number below came from a command that actually ran; nothing is projected. The
**published weights are from Run B** (the frozen-backbone, private-safe run).

**Hardware / stack.** Apple M4 Max, 128 GB unified memory, macOS 26.5. PyTorch
2.12.1 + Transformers 5.13.0, MPS backend, fp32. Backbone: `zen-nano-0.6b`
(Qwen3, 596M params, hidden 1024, 28 layers).

**How the data is built.** `extract_local_sessions.py` and
`extract_agentic_dataset.py` (shared logic in `session_common.py`) pair each
real user prompt with the model that actually served it. `quality` is a fixed
**proxy = 1.0** (the counterfactual quality of models that did not run is
unobservable), `cost` = served tokens x the catalog's `cost_per_1k`,
`latency_ms` = 0 (not in the logs), `task` = a keyword heuristic ported from
`hanzo-router`'s `Heuristic::classify`. `build_dataset.py` collapses to one
max-reward row per unique prompt. Real model ids are mapped 1:1 to catalog ids
(dated Anthropic ids normalized, e.g. `claude-opus-4-5-20251101` ->
`claude-opus-4-5`); no collapsing into generic tiers.

**Catalog & route head.** `training/catalog.yaml` is the routable universe: **28
models** = 3 local zen + zen cloud tiers + Anthropic / OpenAI / Gemini / Grok /
DeepSeek / Kimi / MiniMax, priced from the gateway's live config
(`hanzoai/ai/conf/models.yaml`, refreshed from pricing.hanzo.ai; models not yet
in it -- e.g. `claude-fable-5`, `gpt-5.5`, `gemini-3-*`, `grok-4` -- carry a
`# verify` provider-list price). The route head sizes from the catalog
(`classes_from: catalog`), so the runs printed `route head: 24`/`28 classes`
(up from 7 in the scaffold) with no code change -- **adding a routable model is
one catalog row.**

### Run A -- local logs, full fine-tune

- **Data**: `~/.claude/projects` + `~/.codex/sessions`. 5,975 raw ->
  **5,455 unique-prompt rows** -> random 80/20 = 4,364 train / 1,091 holdout.
- **Class balance** (the load-bearing caveat): `claude-opus-4-8` **94.6%**,
  `claude-haiku-4-5` 4.9%, others <0.3%; 19 catalog models have zero rows.
  Tasks: code 59.6%, general 24%, cheap_chat 8.5%, math 5.3%, reasoning 2.6%,
  long_context 0.07%, vision/creative 0%.
- **Train**: full fine-tune (backbone + heads), 1 epoch, batch 16, seq 512,
  ~17 min; loss 7.23 -> ~0.7. Artifact: 2.4 GB full weights.
- **Holdout (n=1,091)**: task_acc **0.824**, route_acc **0.961**, top3 **0.999**
  -- but route_acc barely clears the 94.2% majority baseline and the head emits
  only **2 of 24** models. High route_acc here is label memorization, not
  routing skill.

### Run B -- local + private agentic corpus, FROZEN backbone (published)

- **Data**: Run A's local rows (5,993) **plus** the private
  `hanzoai/zen-agentic-dataset-private`. That corpus is 9.9B tokens / 2.3M
  samples, but is mostly synthetic identity SFT + git history; the routing
  signal lives in **embedded Claude Code transcripts** inside assistant turns.
  Streaming the 35 train chunks + valid split yielded only **4,420** extractable
  routing rows (transcripts sit in chunks aa-ad + valid; the rest have no
  served-model label). Combined and deduped: **9,713 unique rows**, stratified
  80/20 by route = **7,772 train / 1,941 holdout**.
- **Class balance**: still Opus-dominated but more diverse --
  `claude-opus-4-8` 53%, `claude-opus-4-5` 35%, `claude-haiku-4-5` 6%,
  `claude-sonnet-4-5` 5%, then `claude-opus-4-1`/`claude-fable-5`/`gpt-5.5`/
  `claude-sonnet-4-6`. **8 models carry data** (vs 5 in Run A).
- **Privacy-safe training**: `--freeze-backbone` freezes the backbone
  (`requires_grad=False`), precomputes pooled embeddings once (cached), and fits
  **only the three linear heads** (12 epochs, seconds; loss 3.04 -> 0.97). The
  published `zen-router.pt` is **1.2 MB of head weights only** -- the backbone
  stays the public `zenlm/zen-nano-0.6b`, so the release cannot memorize or
  reconstruct any private corpus text.
- **Holdout (n=1,941)**: task_acc **0.723**, route_acc **0.791**, top3 **0.990**.

| metric | Run A (full, local) | Run B (frozen, combined) | reading |
|--------|:------:|:------:|---------|
| route models with data | 5 | **8** | corpus adds opus-4-5, sonnet-4-5, opus-4-1 |
| route head size | 24 | **28** | auto-sized from catalog |
| task_acc   | **0.824** | 0.723 | freezing the backbone costs task expressivity |
| route_acc  | 0.961 | **0.791** | Run B beats its **53.3%** majority baseline by **+25.8 pts** and emits **5** models (opus-4-8 x928, opus-4-5 x856, haiku x125, sonnet-4-5 x31, fable x1) -- **genuine multi-model discrimination**, unlike Run A's memorized single label |
| route_top3 | 0.999 | 0.990 | |
| artifact | 2.4 GB | **1.2 MB** | heads only |

Run B's route_acc is a *real* signal: the model learned to tell
`claude-opus-4-8` from `claude-opus-4-5` and route cheap prompts to haiku,
scoring 26 points over the majority baseline. Run A's higher route_acc is not.

**Honest bottom line.** Both corpora are overwhelmingly Claude-Opus-labeled --
this machine and the private corpus both ran Opus for almost everything, and
19-20 of the 28 catalog models never appear. The router head is structurally
complete (28 gateway-priced logits) and now does real discrimination among the
models it has seen, but true cross-provider routing needs **balanced
counterfactual labels** -- the same prompt served by many models with measured
quality/cost/latency. That is exactly what the RouterBench pipeline in
`benchmarks/` provides, and is the stated path forward.

**Single-forward routing latency (Run B, n=1,941).** One prompt, tokenize +
forward, MPS fp32: **mean 95.7 ms, p50 57.0 ms, p99 303.6 ms**. Unquantized
eager Transformers; the `<50 ms CPU / <10 ms Metal` target is at Q4 (see
`make quantize`, not run here). For reference, `llama.cpp` prompt-processing of
`zen-nano-0.6b` at Q4_K_M on this box runs 9,092 tok/s (a 1k-token forward
~= 112 ms).

**Mechanism vs. model.** The Rust routing mechanism this model plugs into
(`hanzo-router` + `enso`) decides in **1.6 us** (rules) to **16 us** (learned
featurize + bilinear select) -- 4-5 orders of magnitude below one 0.6B forward,
so the encoder-mode router forward dominates end-to-end cost, as designed.

Reproduce Run B: `python training/extract_local_sessions.py --out data/evals-local.jsonl`
+ `python training/extract_agentic_dataset.py` -> combine -> `build_dataset.py`
-> stratified split -> `python training/sft.py --config training/config.scaled.yaml
--freeze-backbone` -> `python -m eval.eval_routing --model out/zen-router --data
data/routing-eval.jsonl`.

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

## Quantized deployment

The published weights are **heads only** (1.2 MB), so serving splits in two: the
public `zen-nano-0.6b` backbone runs **quantized in `llama.cpp` embedding mode**
and the three linear heads run in the caller in **numpy** (a 1024×28 matmul is
microseconds). No torch at serve time. Export + recipe live in `export/`
(`export_heads.py`, `route_gguf.py`, `verify_parity.py`, `QUANTIZED.md`).

```
prompt ─▶ llama.cpp (zen-nano Q4_K_M, --pooling last --embd-normalize -1)
            └─ raw last-token hidden state x (1024-d, NOT normalized)
                 └─ numpy: route/task = argmax(W·x + b);  feat = W_feat·x + b_feat
```

**Correctness.** `llama.cpp --pooling last --embd-normalize -1` emits the same
raw post-final-norm hidden state HF returns as `last_hidden_state` (torch f16
norm 101.4 / llama 102.7, element-wise match). Normalization must be **off** — the
head bias is calibrated to the ≈100 norm; an L2-normalized (norm 1) vector lets
the bias swamp the signal and collapses routing to one label.

**Parity** (torch fp32/MPS pipeline vs GGUF+heads, same heads, n=100 holdout,
Apple M4 Max, `llama.cpp` build 9430 Metal):

| quant | size | task argmax | **route argmax** | route top-3 | pooled cosine |
|-------|-----:|:-----------:|:----------------:|:-----------:|:-------------:|
| Q4_K_M | 397 MB | 89% | **95%** | **100%** | 0.969 |
| Q8_0   | 639 MB | 95% | **97%** | **100%** | 0.988 |

The quantized backbone reproduces the torch routing pick 95% of the time at Q4
(97% at Q8) and never drops it out of the top-3. Q4_K_M is the default; Q8_0 for
fidelity-sensitive use.

**Latency** (Q4_K_M, resident `llama-server`, prompt → model id, n=200 holdout
mix; token lengths min 3 / p50 159 / mean 405 / max 1373):

```
mean 132 ms   p50 97 ms   p99 351 ms   min 12.5 ms
```

Stratified p50 by prompt length: 1–32 tok **50 ms** · 129–512 tok **107 ms** ·
513+ tok **251 ms** — a ~40–50 ms Metal-dispatch + HTTP floor plus linear
prompt-processing (p50 tracks the earlier `llama-bench pp1024 ≈ 112 ms`). For
reference the **fp32 torch/MPS** path measured p50 **57 ms** on the same holdout:
for a 0.6B model, in-process MPS matmuls beat a Q4 `llama.cpp` **HTTP
round-trip**, so the quantized path's win here is **footprint** (397 MB vs
2.4 GB) and **portability** (CPU / edge / Vulkan, no torch), not raw M4-Max
latency. See `export/QUANTIZED.md` for the full recipe and reproduction.

## Formats

PyTorch (safetensors), GGUF (Q2_K–F16), MLX.
