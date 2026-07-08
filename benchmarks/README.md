# benchmarks — replicating published LLM-routing research

A reproducible pipeline that measures zen-router against the **public benchmarks
and methodology used in the LLM-routing literature** — RouterBench, FrugalGPT,
and the LMSYS/Arena preference datasets — with closed, open, and per-model
baselines computed from the *same* data.

```
make bench-data       # download a suite, convert to profile rows
make bench-replicate  # sweep the cost-quality frontier + baselines
```

## What's here

| file | role |
|---|---|
| `suites.yaml` | declarative registry of eval suites (HF dataset id, kind, schema, model list, task map). Every id verified to resolve on 2026-07-07. |
| `build_profiles.py` | downloads a `profile` suite and converts per-example, per-model records into router rows `{prompt, model, quality, cost, latency_ms, task}`. |
| `prices.yaml` | fallback per-model token prices (only used when a suite has no native cost; RouterBench ships native cost, so unused for it). |
| `replicate.py` | RouterBench-methodology evaluation: cost-quality frontier, AIQ area metric, and baselines (every individual model, random, oracle). Pure numpy; emits JSON + markdown + CSV. |

## The suites

Each suite is tagged by `kind`, which decides what it's good for:

- **`profile`** — per-example, per-model correctness **and** cost already in the
  dataset. Directly convertible to router train/eval rows.
  - `routerbench` (`withmartian/routerbench`) — **the substrate.** 36 497 prompts
    drawn from MMLU, GSM8K, MBPP, HellaSwag, ARC, Winogrande, MT-Bench and more,
    each answered by **11 LLMs** with a `[0,1]` correctness score and a USD cost
    per response. This is exactly the RouterBench / FrugalGPT measurement setting.
- **`preference`** — pairwise human votes between two model answers (the RouteLLM
  training substrate). Needs a Bradley-Terry / win-rate reduction, not raw
  conversion.
  - `lmsys/chatbot_arena_conversations`, `lmarena-ai/arena-human-preference-100k`.
- **`raw`** — benchmark questions with a gold answer but **no per-model outputs**.
  To become a profile they must first be answered by each candidate model and
  graded. Declared here so the methodology (which models, which metric) lives in
  one place; conversion is out of scope until generations exist.
  - `TIGER-Lab/MMLU-Pro`, `Idavidrein/gpqa`, `openai/gsm8k`, `HuggingFaceH4/MATH-500`,
    `openai/openai_humaneval`, `google-research-datasets/mbpp`,
    `livecodebench/code_generation_lite`, `lmarena-ai/arena-hard-auto-v0.1`,
    `google/IFEval`.

All 12 dataset ids returned HTTP 200 from `https://huggingface.co/api/datasets/<id>`
on 2026-07-07; none were dropped.

## Methodology (RouterBench)

For a candidate set of models, RouterBench (Hu et al. 2024, arXiv:2403.12031)
measures a *router* by the cost-quality trade-off it achieves:

1. Each prompt `i` has, per model `j`, a quality `Q[i,j]` and a cost `C[i,j]`.
2. A routing policy assigns a score `s[i,j]`. Routing to
   `argmax_j (s[i,j] - λ·C[i,j])` and sweeping the willingness-to-pay `λ` traces a
   realized `(mean cost, mean quality)` curve; its non-dominated upper convex hull
   is the achievable frontier.
3. **AIQ** (Average Improvement in Quality) = area under that frontier normalized
   by the cost span = average achievable quality across the willingness-to-pay
   range. Higher is better.

Baselines, all computed **from the same data** so our curve sits against
published-result-style references:

- every **individual model** (a point) — including the best (closed: GPT-4) and
  the cheapest;
- the **interpolation frontier** = convex hull of those points — the "just pick
  one model / statically mix" curve a router must beat;
- the **random router** (expected point = grand means);
- the **oracle** (per-prompt max quality, tie-break min cost) — a point and, via
  `s = true quality`, an upper-bound frontier.

Policies (all run **without trained weights** except `checkpoint`):
`oracle` · `prior` (untrained heuristic: score = each model's global mean quality)
· `random` · `individual:<model>` · `checkpoint` (trained route-head logits).

## Full run — all 36,497 prompts (RESULTS)

The whole RouterBench 0-shot suite, converted and evaluated end-to-end. These are
the headline numbers; the 200-sample smoke slice below is only a pipeline sanity
check.

```
make bench-data BENCH_SUITE=routerbench BENCH_SAMPLE=0          # 36,497 ex x 11 = 401,467 rows
make bench-replicate BENCH_POLICY=checkpoint \
     BENCH_DATA=data/profiles-routerbench.jsonl                 # trained heads + all baselines
make bench-replicate BENCH_POLICY=prior                        # untrained heuristic AIQ
```

- **Build**: 36,497 examples × 11 models → **401,467 rows** (36,481 unique prompts);
  `load_matrix` keeps the **36,481** prompts with full 11-model coverage (0 dropped).
- **Hardware / wall-clock**: Apple **M4 Max**, MPS, backbone fp16. The checkpoint
  policy batch-embeds all 36,481 prompts (backbone forward, pooled last token,
  batch 64, embeddings cached to `out/bench/*.emb.npz`) then applies the route
  head — **27.8 min wall-clock** end-to-end (`1667.99s real`), full dataset, no
  subsampling. Baseline policies (numpy over the matrix) are seconds.

### RouterBench-style cost-quality replication — full suite

- data: `data/profiles-routerbench.jsonl`  ·  prompts: **36,481**  ·  models: **11**
- best individual model: **gpt-4-1106-preview** (quality 0.781 @ cost 0.003292)
- cheapest model: **mistralai/mistral-7b-chat** (quality 0.306 @ cost 0.000046)

| policy / baseline | quality | cost (USD/ex) | AIQ |
|---|---:|---:|---:|
| oracle (per-prompt best, upper bound) | 0.912 | 0.000242 | **0.8701** |
| prior-heuristic (untrained, global-mean quality) | 0.781 | 0.003292 | 0.7427 |
| interpolation / non-routing hull (statically pick one) | — | — | 0.7054 |
| **zen-router checkpoint** (frozen-backbone heads, tier-mapped) | 0.636 | 0.002413 | **0.6248** |
| random router | 0.521 | 0.000831 | — |
| individual: gpt-4-1106-preview (best) | 0.781 | 0.003292 | — |
| individual: zero-one-ai/Yi-34B-Chat | 0.647 | 0.000186 | — |
| individual: claude-v2 | 0.636 | 0.002418 | — |
| individual: claude-v1 | 0.630 | 0.002145 | — |
| individual: gpt-3.5-turbo-1106 | 0.619 | 0.000243 | — |
| individual: claude-instant-v1 | 0.598 | 0.000233 | — |
| individual: mistralai/mixtral-8x7b-chat | 0.547 | 0.000135 | — |
| individual: WizardLM/WizardLM-13B-V1.2 | 0.431 | 0.000073 | — |
| individual: meta/llama-2-70b-chat | 0.329 | 0.000203 | — |
| individual: mistralai/mistral-7b-chat (cheapest) | 0.306 | 0.000046 | — |
| individual: meta/code-llama-instruct-34b-chat | 0.202 | 0.000172 | — |

The **oracle** reaches **0.912 quality at 0.000242 USD/example** — higher quality
than the best single model (GPT-4, 0.781 @ 0.003292) at ~14× lower cost — by
routing each prompt to the cheapest model that gets it right. The untrained
**prior-heuristic** (AIQ 0.7427) sits between the static interpolation hull
(0.7054) and the oracle upper bound (0.8701), exactly where an untrained baseline
should.

### The trained checkpoint on RouterBench — a NEGATIVE transfer result, reported plainly

**The trained zen-router checkpoint scores AIQ 0.6248 — BELOW the non-routing
interpolation hull (0.7054), the untrained prior heuristic (0.7427), and the
oracle (0.8701).** On this eval it does not beat "just statically pick one model."
Its cost-quality frontier collapses onto a single high-tier point (0.636 @
0.002413, ≈ the `claude-v2`→`claude-opus-4-5` column) because the route head
concentrates mass on the frontier-Claude tier for almost every prompt.

This is expected, and the cause is documented, not massaged:

- **Zero in-catalog overlap.** None of RouterBench's 11 (2023-era) models are in
  the checkpoint's 28-class 2026 catalog. Scores are read through a **family/tier
  bridge** (`benchmarks/checkpoint_map.yaml`), so this measures whether the head's
  learned **task/tier discrimination transfers** onto RouterBench's universe — it
  is **not** native in-catalog routing.
- **Zero training rows for these targets.** The checkpoint's training corpus was
  overwhelmingly Claude-Opus-labeled (see the main model card): 8 of 28 catalog
  models carry any data, and the mapped-to classes for the *cheap* RouterBench
  models (`zen5-flash`, `zen-agent-4b`, `deepseek-v3.2`, …) saw ~0 rows. The head
  never learned to prefer a cheap model for an easy prompt, so under a
  cost-penalized sweep it cannot trace a frontier — it just keeps voting frontier
  tier, which is exactly the failure the AUC shows.

**Family/tier bridge used** (RouterBench model → catalog class whose route-head
logit is read; a clean 11→11 bijection):

| RouterBench model | → catalog class | tier |
|---|---|---|
| gpt-4-1106-preview | gpt-5.5 | closed frontier |
| claude-v2 | claude-opus-4-5 | closed frontier |
| claude-v1 | claude-opus-4-1 | closed frontier (older) |
| gpt-3.5-turbo-1106 | gpt-5.4-mini | closed cheap |
| claude-instant-v1 | claude-haiku-4-5 | closed cheap |
| meta/llama-2-70b-chat | kimi-k2 | open large |
| mistralai/mixtral-8x7b-chat | deepseek-v3.2 | open MoE, cheap-mid |
| zero-one-ai/Yi-34B-Chat | minimax-m2 | open mid |
| meta/code-llama-instruct-34b-chat | zen5-coder | open code specialist |
| mistralai/mistral-7b-chat | zen5-flash | open small / cheapest |
| WizardLM/WizardLM-13B-V1.2 | zen-agent-4b | small open |

**Bottom line.** The checkpoint's *own* holdout showed real discrimination among
the models it was trained on (Run B route_acc 0.791, +25.8 pts over majority — see
the model card), but that skill **does not transfer** to routing a disjoint,
never-seen 2023 model set through a tier bridge. Beating the RouterBench frontier
requires training on RouterBench's own per-model counterfactual labels (the
balanced quality/cost signal this pipeline exists to provide), not a family/tier
projection of a Claude-Opus-skewed catalog. The harness, mapping, and numbers are
all reproducible with the commands above.

## Smoke run (200 samples — pipeline sanity, NOT results)

A 200-example CI-sized slice of RouterBench 0-shot proves the pipeline runs
end-to-end. **This is a smoke test on a tiny random slice, not a benchmark
result** — use the full run above.

```
make bench-data BENCH_SAMPLE=200      # -> 200 examples x 11 models = 2200 rows
make bench-replicate BENCH_POLICY=oracle
```

On that slice: oracle 0.909 @ 0.000248 (AIQ 0.8606), prior-heuristic AIQ 0.7295,
interpolation hull 0.6891 — the same ordering as the full run, on 0.5% of the data.

## What "replication" means here — honestly

- **Same data, same metric.** We use the published datasets unchanged and the
  RouterBench cost-quality-frontier + AIQ methodology. The `routerbench` suite's
  quality labels and per-response costs are the dataset authors' own.
- **Labels / prices differ where noted.** For `raw` suites there are no per-model
  outputs yet, so quality would come from *our* generations and cost from
  `prices.yaml` (each rate marked `# verify`), not from the original papers. Those
  suites are declared but not converted until generations exist.
- **Model sets differ by era.** RouterBench's 11 models are 2023-era; none are in
  the current `training/catalog.yaml`, so each keeps its own id as a routing
  target (`model_map: {}`). Comparing our router to *current* SotA models requires
  re-running the profile step with those models' generations — the pipeline is the
  same, only the model list changes.
- **No latency signal in RouterBench** — emitted rows carry `latency_ms: null`.
- This is a **measurement harness**, not a leaderboard submission. Numbers here
  are reproducible from the cited public datasets with the commands above.

## Citations

- Hu et al., *RouterBench: A Benchmark for Multi-LLM Routing Systems*, 2024 — arXiv:2403.12031
- Chen et al., *FrugalGPT*, 2023 — arXiv:2305.05176
- Ong et al., *RouteLLM*, 2024 — arXiv:2406.18665
- Zheng et al., *Judging LLM-as-a-Judge / Chatbot Arena*, 2023 — arXiv:2306.05685
- Wang et al., *MMLU-Pro*, 2024 — arXiv:2406.01574 · Rein et al., *GPQA*, 2023 — arXiv:2311.12022
- Cobbe et al., *GSM8K*, 2021 — arXiv:2110.14168 · Lightman et al., *MATH / PRM800K*, 2023
- Chen et al., *HumanEval*, 2021 — arXiv:2107.03374 · Austin et al., *MBPP*, 2021 — arXiv:2108.07732
- Jain et al., *LiveCodeBench*, 2024 — arXiv:2403.07974 · Zhou et al., *IFEval*, 2023 — arXiv:2311.07911
