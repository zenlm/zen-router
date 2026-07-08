"""Convert a public routing-research suite into zen-router profile rows.

A "profile row" is the eval-row format the rest of the repo already speaks
(training/build_dataset.py --evals):

    {"prompt", "model", "quality", "cost", "latency_ms", "task"}

One source example (a prompt answered by N models) expands to N rows, one per
model, carrying that model's correctness score (quality), its cost, and the
mapped task. This is the RouterBench / FrugalGPT substrate: per-example,
per-model quality and cost, from which a cost-quality routing curve is measured
(see benchmarks/replicate.py).

Default suite: routerbench (the one `kind: profile` suite). `raw` and
`preference` suites have no per-model scores yet and are refused with a clear
message -- they need generation / a Bradley-Terry reduction first.

Usage:
    uv run --with datasets,huggingface_hub,pandas \
        python benchmarks/build_profiles.py --suite routerbench --sample 200 \
        --out data/profiles-routerbench.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def map_task(eval_name: str, task_map: dict[str, str]) -> str:
    """Longest-prefix match of an eval_name to a router task; else 'general'."""
    name = (eval_name or "").lower()
    for prefix in sorted((k for k in task_map if k), key=len, reverse=True):
        if name.startswith(prefix.lower()):
            return task_map[prefix]
    return task_map.get("", "general")


def _clean(value) -> float | None:
    """Coerce a score/cost cell to float, dropping NaN/None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def rows_from_records(
    records: Iterable[dict],
    suite: dict,
    price_map: dict[str, float] | None = None,
) -> Iterator[dict]:
    """Pure conversion: wide per-model records -> flat profile rows.

    Each record is one prompt with, per model M: a bare "<M>" score column, an
    optional "<M><cost_suffix>" cost column, and (unused) response column.
    Cost is native when present; otherwise priced as tokens x prices.yaml, and
    if neither is available the row's cost is 0.0 (documented, not silent-NaN).
    """
    sch = suite["schema"]
    prompt_field = sch["prompt"]
    task_field = sch.get("task")
    cost_suffix = sch.get("cost_suffix", "|total_cost")
    task_map = suite.get("task_map", {})
    model_map = suite.get("model_map", {})
    prices = price_map or {}

    for rec in records:
        prompt = rec.get(prompt_field)
        if prompt is None:
            continue
        prompt = str(prompt)
        task = map_task(rec.get(task_field, ""), task_map) if task_field else suite.get("default_task", "general")
        for src_model in suite["models"]:
            quality = _clean(rec.get(src_model))
            if quality is None:
                continue  # model didn't answer this example
            native_cost = _clean(rec.get(src_model + cost_suffix))
            if native_cost is not None:
                cost = native_cost
            elif src_model in prices:
                # No token counts in these suites -> price is the fallback rate
                # per response; documented in prices.yaml as approximate.
                cost = float(prices[src_model])
            else:
                cost = 0.0
            yield {
                "prompt": prompt,
                "model": model_map.get(src_model, src_model),
                "quality": quality,
                "cost": cost,
                "latency_ms": None,  # RouterBench has no latency signal
                "task": task,
            }


def load_profile_records(suite: dict, variant: str, sample: int | None) -> list[dict]:
    """Download a `profile` suite and return wide per-model records (df rows)."""
    import pandas as pd  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    fname = suite["files"][variant]
    path = hf_hub_download(suite["hf_id"], fname, repo_type="dataset")
    df = pd.read_pickle(path)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=0).reset_index(drop=True)
    return df.to_dict("records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="routerbench")
    ap.add_argument("--variant", default="0shot", help="profile file variant (routerbench: 0shot|5shot)")
    ap.add_argument("--sample", type=int, default=None, help="random N examples (CI-sized runs)")
    ap.add_argument("--suites", type=Path, default=ROOT / "benchmarks/suites.yaml")
    ap.add_argument("--prices", type=Path, default=ROOT / "benchmarks/prices.yaml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    registry = load_yaml(args.suites)["suites"]
    if args.suite not in registry:
        raise SystemExit(f"unknown suite '{args.suite}'. known: {', '.join(registry)}")
    suite = registry[args.suite]
    if suite["kind"] != "profile":
        raise SystemExit(
            f"suite '{args.suite}' is kind={suite['kind']}: no per-model scores to convert. "
            "Only `profile` suites (e.g. routerbench) build directly; `raw` needs generation, "
            "`preference` needs a Bradley-Terry reduction."
        )

    price_map = load_yaml(args.prices).get("prices", {}) if args.prices.exists() else {}
    # sample <= 0 means "use the whole suite" (0 is the documented full-run value);
    # only a positive N subsamples. Guards df.sample(n=0) emptying the frame.
    sample = args.sample if args.sample and args.sample > 0 else None
    records = load_profile_records(suite, args.variant, sample)
    rows = list(rows_from_records(records, suite, price_map))

    out = args.out or (ROOT / f"data/profiles-{args.suite}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_prompts = len({r["prompt"] for r in rows})
    print(f"{args.suite}: {len(records)} examples x {len(suite['models'])} models "
          f"-> {len(rows)} rows ({n_prompts} prompts) -> {out}")


if __name__ == "__main__":
    main()
