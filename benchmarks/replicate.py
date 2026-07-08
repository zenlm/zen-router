"""RouterBench-methodology cost-quality evaluation for zen-router.

Reads profile rows (benchmarks/build_profiles.py output):
    {"prompt", "model", "quality", "cost", "latency_ms", "task"}
and reproduces the RouterBench measurement (Hu et al. 2024, arXiv:2403.12031):

  * Build the per-prompt quality/cost matrix over the candidate models.
  * A routing *policy* assigns each prompt a per-model score s[i,j]; sweeping a
    willingness-to-pay lambda and routing to argmax_j (s[i,j] - lambda*cost[i,j])
    traces a realized (mean cost, mean quality) curve. Its non-dominated upper
    convex hull is the achievable cost-quality frontier.
  * AIQ (Average Improvement in Quality): area under that frontier, normalized by
    the cost span -> average achievable quality across the willingness-to-pay
    range. Higher = better cost-quality trade-off.

Baselines computed FROM THE SAME DATA, so our curve sits against published-style
references:
  * every individual model (a single point) -- incl. best and cheapest,
  * the linear-interpolation frontier = convex hull of those points (the curve a
    router must beat; RouterBench's "non-routing" reference),
  * random router (expected point = grand means),
  * oracle (per-prompt max-quality, tie-break min-cost) as a point AND as an
    upper-bound frontier (score = true quality).

Policies (all run WITHOUT trained weights except `checkpoint`):
  oracle             -- headline frontier = oracle (true-quality) frontier
  prior              -- untrained heuristic: score = each model's global mean
                        quality (prompt-independent) -> traces the individual hull
  random             -- headline = random point
  individual:<model> -- headline = that single model's point
  checkpoint         -- score = trained route-head logits (needs --checkpoint)

Pure numpy; no plotting. Emits JSON + a markdown table + a CSV of curve points.

Usage:
    uv run python benchmarks/replicate.py --data data/profiles-routerbench.jsonl \
        --policy oracle --out-prefix out/bench/routerbench
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_matrix(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """profile rows -> (models, Q, C) with Q,C shape (n_prompts, n_models).

    Keeps only prompts covered by ALL models (a clean rectangular matrix), which
    is the RouterBench setting. Returns model order and the two matrices.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    models = sorted({r["model"] for r in rows})
    midx = {m: j for j, m in enumerate(models)}
    by_prompt: dict[str, dict[str, tuple[float, float]]] = {}
    for r in rows:
        by_prompt.setdefault(r["prompt"], {})[r["model"]] = (
            float(r["quality"]),
            float(r["cost"]),
        )
    full = [p for p, mm in by_prompt.items() if len(mm) == len(models)]
    dropped = len(by_prompt) - len(full)
    if dropped:
        print(f"note: dropped {dropped} prompts without full model coverage "
              f"(kept {len(full)}/{len(by_prompt)})")
    n = len(full)
    if n == 0:
        raise SystemExit("no prompts with full model coverage; nothing to evaluate")
    Q = np.zeros((n, len(models)))
    C = np.zeros((n, len(models)))
    for i, p in enumerate(full):
        for m, (q, c) in by_prompt[p].items():
            Q[i, midx[m]] = q
            C[i, midx[m]] = c
    return models, Q, C


def route(score: np.ndarray, Q: np.ndarray, C: np.ndarray, lam: float) -> tuple[float, float]:
    """Route each prompt to argmax(score - lam*cost); return realized (cost, q)."""
    choice = np.argmax(score - lam * C, axis=1)
    rows = np.arange(Q.shape[0])
    return float(C[rows, choice].mean()), float(Q[rows, choice].mean())


def frontier(score: np.ndarray, Q: np.ndarray, C: np.ndarray, n_lambda: int = 64) -> list[tuple[float, float]]:
    """Sweep lambda, realize (cost, quality) points, return non-dominated hull."""
    # lambda 0 -> pure quality; large -> pure cost. Span the cost scale of data.
    cmax = float(C.max()) or 1.0
    lams = np.concatenate(([0.0], np.logspace(-3, 3, n_lambda - 1) / cmax))
    pts = {route(score, Q, C, float(l)) for l in lams}
    return upper_hull(sorted(pts))


def upper_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pareto upper-left hull: max quality per cost, then convex, cost-ascending."""
    # Non-dominated: no other point has >= quality at <= cost.
    pts = sorted(set(points))
    nd: list[tuple[float, float]] = []
    best_q = -np.inf
    for c, q in pts:  # ascending cost; keep points that raise the quality ceiling
        if q > best_q:
            nd.append((c, q))
            best_q = q
    # Upper convex hull over the non-dominated set.
    hull: list[tuple[float, float]] = []
    for c, q in nd:
        while len(hull) >= 2:
            (c0, q0), (c1, q1) = hull[-2], hull[-1]
            # drop c1 if it's below the line c0->(c,q) (not on upper hull)
            if (q1 - q0) * (c - c0) <= (q - q0) * (c1 - c0):
                hull.pop()
            else:
                break
        hull.append((c, q))
    return hull


def aiq(hull: list[tuple[float, float]]) -> float:
    """Average quality across the cost span = area / cost-range (0..1-ish)."""
    if len(hull) < 2:
        return float(hull[0][1]) if hull else 0.0
    c = np.array([p[0] for p in hull])
    q = np.array([p[1] for p in hull])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 rename
    area = float(trapezoid(q, c))
    span = float(c[-1] - c[0])
    return area / span if span > 0 else float(q.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--policy", default="oracle",
                    help="oracle|prior|random|individual:<model>|checkpoint")
    ap.add_argument("--checkpoint", type=Path, default=None, help="trained router dir (policy=checkpoint)")
    ap.add_argument("--out-prefix", type=Path, default=ROOT / "out/bench/routerbench")
    args = ap.parse_args()

    models, Q, C = load_matrix(args.data)
    n = Q.shape[0]

    # --- baselines (points) -------------------------------------------------
    individual = [
        {"model": m, "cost": float(C[:, j].mean()), "quality": float(Q[:, j].mean())}
        for j, m in enumerate(models)
    ]
    best_model = max(individual, key=lambda d: d["quality"])
    cheapest = min(individual, key=lambda d: d["cost"])
    random_point = {"cost": float(C.mean()), "quality": float(Q.mean())}
    orc_choice = np.argmax(Q - 1e-9 * C, axis=1)  # max quality, tie-break min cost
    rows = np.arange(n)
    oracle_point = {"cost": float(C[rows, orc_choice].mean()), "quality": float(Q[rows, orc_choice].mean())}

    # --- reference frontiers ------------------------------------------------
    interp_hull = upper_hull([(d["cost"], d["quality"]) for d in individual])  # non-routing reference
    oracle_hull = frontier(Q, Q, C)  # upper bound (score = true quality)

    # --- selected policy frontier / point -----------------------------------
    policy = args.policy
    if policy == "oracle":
        headline = {"kind": "frontier", "name": "oracle", "hull": oracle_hull, "aiq": aiq(oracle_hull)}
    elif policy == "prior":
        prior = np.tile(Q.mean(axis=0), (n, 1))  # global mean quality per model
        hull = frontier(prior, Q, C)
        headline = {"kind": "frontier", "name": "prior-heuristic", "hull": hull, "aiq": aiq(hull)}
    elif policy == "random":
        headline = {"kind": "point", "name": "random", **random_point}
    elif policy.startswith("individual:"):
        name = policy.split(":", 1)[1]
        if name not in models:
            raise SystemExit(f"unknown model '{name}'. known: {', '.join(models)}")
        d = individual[models.index(name)]
        headline = {"kind": "point", "name": f"individual:{name}", **{"cost": d["cost"], "quality": d["quality"]}}
    elif policy == "checkpoint":
        score = checkpoint_scores(args.checkpoint, args.data, models)
        hull = frontier(score, Q, C)
        headline = {"kind": "frontier", "name": "zen-router", "hull": hull, "aiq": aiq(hull)}
    else:
        raise SystemExit(f"unknown policy '{policy}'")

    report = {
        "data": str(args.data),
        "n_prompts": n,
        "models": models,
        "policy": policy,
        "headline": headline,
        "baselines": {
            "individual": individual,
            "best_model": best_model,
            "cheapest_model": cheapest,
            "random": random_point,
            "oracle_point": oracle_point,
            "interpolation_frontier": {"hull": interp_hull, "aiq": aiq(interp_hull)},
            "oracle_frontier": {"hull": oracle_hull, "aiq": aiq(oracle_hull)},
        },
    }

    out_json = Path(f"{args.out_prefix}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    # curve points CSV (headline frontier if any, else oracle frontier)
    curve = headline.get("hull") or oracle_hull
    with Path(f"{args.out_prefix}.curve.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cost", "quality"])
        w.writerows(curve)

    md = render_markdown(report)
    Path(f"{args.out_prefix}.md").write_text(md)
    print(md)


def checkpoint_scores(ckpt: Path | None, data: Path, models: list[str]) -> np.ndarray:
    """Per-(prompt,model) routing score from a trained route head's logits."""
    if ckpt is None:
        raise SystemExit("policy=checkpoint requires --checkpoint <dir>")
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    import sys  # noqa: PLC0415
    sys.path.insert(0, str(ROOT))
    from training.sft import ZenRouter  # noqa: PLC0415

    cfg = json.loads((ckpt / "router_config.json").read_text())
    catalog = cfg["catalog"]
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = ZenRouter(cfg.get("base", str(ckpt)), len(cfg["tasks"]), len(catalog), 256)
    model.load_state_dict(torch.load(ckpt / "zen-router.pt", map_location="cpu"))
    model.eval()
    prompts = [json.loads(l)["prompt"] for l in data.read_text().splitlines() if l.strip()]
    prompts = list(dict.fromkeys(prompts))  # unique, order-preserving
    # map data models -> catalog columns; models absent from catalog score 0
    col = [catalog.index(m) if m in catalog else None for m in models]
    scores = np.zeros((len(prompts), len(models)))
    with torch.no_grad():
        for i, p in enumerate(prompts):
            enc = tok(p, return_tensors="pt", truncation=True, max_length=2048)
            _, r_logits, _ = model(**enc)
            logits = r_logits[0].numpy()
            for j, c in enumerate(col):
                scores[i, j] = logits[c] if c is not None else logits.min()
    return scores


def render_markdown(rep: dict) -> str:
    b = rep["baselines"]
    lines = [
        f"### RouterBench-style cost-quality replication — `{rep['policy']}`",
        "",
        f"- data: `{rep['data']}`  ·  prompts: **{rep['n_prompts']}**  ·  models: **{len(rep['models'])}**",
        f"- best individual model: **{b['best_model']['model']}** "
        f"(quality {b['best_model']['quality']:.3f} @ cost {b['best_model']['cost']:.6f})",
        f"- cheapest model: **{b['cheapest_model']['model']}** "
        f"(quality {b['cheapest_model']['quality']:.3f} @ cost {b['cheapest_model']['cost']:.6f})",
        "",
        "| policy / baseline | quality | cost (USD/ex) | AIQ |",
        "|---|---:|---:|---:|",
    ]
    h = rep["headline"]
    if h["kind"] == "frontier":
        hi = h["hull"][-1]
        lines.append(f"| **{h['name']}** (frontier, best point) | {hi[1]:.3f} | {hi[0]:.6f} | {h['aiq']:.4f} |")
    else:
        lines.append(f"| **{h['name']}** | {h['quality']:.3f} | {h['cost']:.6f} | — |")
    lines.append(f"| oracle (per-prompt best) | {b['oracle_point']['quality']:.3f} | {b['oracle_point']['cost']:.6f} | {b['oracle_frontier']['aiq']:.4f} |")
    lines.append(f"| random router | {b['random']['quality']:.3f} | {b['random']['cost']:.6f} | — |")
    lines.append(f"| interpolation (non-routing hull) | — | — | {b['interpolation_frontier']['aiq']:.4f} |")
    for d in sorted(b["individual"], key=lambda x: -x["quality"]):
        lines.append(f"| individual: {d['model']} | {d['quality']:.3f} | {d['cost']:.6f} | — |")
    lines.append("")
    lines.append("AIQ = area under the cost-quality frontier / cost span "
                 "(average achievable quality across willingness-to-pay).")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
