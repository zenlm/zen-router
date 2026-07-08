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

## Smoke run (200 samples — NOT results)

A 200-example CI-sized slice of RouterBench 0-shot, to prove the pipeline runs
end-to-end. **This is a smoke test on a tiny random slice, not a benchmark
result.** Full numbers require the whole suite (`make bench-data BENCH_SAMPLE=`).

```
make bench-data BENCH_SAMPLE=200      # -> 200 examples x 11 models = 2200 rows
make bench-replicate BENCH_POLICY=oracle
```

### RouterBench-style cost-quality replication — `oracle`

- data: `data/profiles-routerbench.jsonl`  ·  prompts: **200**  ·  models: **11**
- best individual model: **gpt-4-1106-preview** (quality 0.779 @ cost 0.003143)
- cheapest model: **mistralai/mistral-7b-chat** (quality 0.309 @ cost 0.000044)

| policy / baseline | quality | cost (USD/ex) | AIQ |
|---|---:|---:|---:|
| **oracle** (frontier, best point) | 0.909 | 0.000248 | 0.8606 |
| oracle (per-prompt best) | 0.909 | 0.000248 | 0.8606 |
| random router | 0.512 | 0.000794 | — |
| interpolation (non-routing hull) | — | — | 0.6891 |
| individual: gpt-4-1106-preview | 0.779 | 0.003143 | — |
| individual: claude-v2 | 0.646 | 0.002295 | — |
| individual: claude-v1 | 0.637 | 0.002044 | — |
| individual: zero-one-ai/Yi-34B-Chat | 0.618 | 0.000181 | — |
| individual: gpt-3.5-turbo-1106 | 0.604 | 0.000236 | — |
| individual: claude-instant-v1 | 0.599 | 0.000223 | — |
| individual: mistralai/mixtral-8x7b-chat | 0.547 | 0.000134 | — |
| individual: WizardLM/WizardLM-13B-V1.2 | 0.417 | 0.000071 | — |
| individual: mistralai/mistral-7b-chat | 0.309 | 0.000044 | — |
| individual: meta/llama-2-70b-chat | 0.301 | 0.000198 | — |
| individual: meta/code-llama-instruct-34b-chat | 0.175 | 0.000170 | — |

The oracle reaches **0.909 quality at 0.000248 USD/example** — higher quality than
the best single model (GPT-4, 0.779 @ 0.003143) at ~13× lower cost — because it
routes each prompt to the cheapest model that gets it right. The prior-heuristic
policy scores AIQ 0.7295, between the static interpolation hull (0.6891) and the
oracle upper bound (0.8606), exactly where an untrained baseline should sit. A
trained checkpoint (`--policy checkpoint --checkpoint out/zen-router`) is measured
on the same axes.

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
