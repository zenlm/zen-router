# zen-router integration

One tiny model, three consumption points.

## 1. Rust local/hybrid router (engine)

`hanzo-router` exposes the `Classifier` seam (`classify.rs`: request → task)
and `enso` exposes the `Featurizer` seam (request → feature vector for the
per-user LinUCB bandit). zen-router serves both from one forward pass:

- task head → `impl Classifier for ZenRouterClassifier` (replaces the
  keyword heuristic; falls back to it when the model isn't loaded)
- feature head → `impl Featurizer` (replaces hashed n-grams; `enso`'s
  bilinear `xᵀWp` and per-user bandit consume richer features without any
  other change)

Runtime: the node keeps zen-router resident in hanzo-engine (0.6B Q4 ≈
0.4 GB) and calls it before every route decision. Cold start without the
model = existing rules; nothing breaks.

## 2. Cloud gateway (cloud-api)

Generative mode: cloud-api asks engine for a single constrained route tag
`<route model="..." level="..."/>`, then resolves it through the ModelRoute
table (primary + fallbacks, pricing) and the billing gate. Rate-limit and
spend snapshots from the usage plane are applied as decode-time masks:
providers near their window limits are excluded before the argmax.

## 3. Online learning loop

Every routed request settles realized quality/cost/latency into the usage
ledger. `make data` re-exports the ledger into training rows; `enso`'s
`observe(reward)` handles the per-user online part between retrains. The
reward is the same objective everywhere:
`quality − λ·cost − μ·latency` under the caller's SLO.

## Catalog discipline

`training/catalog.yaml` ids must equal the serving registry `ModelCard.id`s
1:1 — route logits map by index. Adding a model = add a catalog row, retrain
(or mask the new logit until data exists).
