# Routing configuration -- zero-code cost saving

zen-router is the learned brain; `hanzo-router` is the mechanism. You do not
need the learned model to get cost-aware routing: `hanzo-router` loads a
declarative `router_policy.yaml` and routes rules-only, immediately. Drop the
file in, point hanzo-node at it, and requests route zen-local-first with a cloud
fallback under a cost ceiling -- no code changes.

## Drop-in `router_policy.yaml`

```yaml
# Per-task preference: the first usable model wins. Local zen models are listed
# first, so anything that fits the box's RAM is served on-device (cost 0); the
# walk falls through to cloud only when no listed local model fits or is usable.
prefer:
  cheap_chat:   [zen-nano-0.6b, zen5-flash, claude-haiku-4-5-20251001]
  general:      [zen-agent-4b, zen5, claude-haiku-4-5-20251001]
  code:         [zen-coder-24b, zen5-coder, gpt-5.1-codex-max, claude-sonnet-4-6]
  reasoning:    [zen-agent-4b, zen5-max, claude-opus-4-8, gpt-5.5]
  math:         [zen5-max, deepseek-reasoner, o3, claude-opus-4-8]
  creative:     [claude-fable-5, claude-sonnet-4-6, zen5]
  vision:       [zen-omni, zen-vl, gemini-3-pro]
  long_context: [gemini-3-flash, zen5-max, claude-opus-4-8]

# Fraction of *available* memory a local model may occupy to count as "fits".
# Omit to use the engine default (0.85 unified / 0.70 discrete).
memory_fraction: 0.85

# Hard ceiling on cloud cost_per_1k. Cloud models above this are never selected,
# so the router can only escalate to affordable providers.
cost_ceiling: 0.02
```

Load and use it (`hanzo-router`):

```rust
let policy = hanzo_router::load_policy(&std::fs::read_to_string("router_policy.yaml")?)?;
let task = hanzo_router::Heuristic.classify(&req);
let ctx = hanzo_router::Context { task, registry: &registry, mem, running: &running,
    vision_required: req.has_media, min_context: req.approx_tokens };
let decision = policy.select(&ctx);   // Reuse | LoadLocal | Cloud | NoFit
```

Every field is optional. An empty file (`{}`) is a valid policy: it routes by the
registry's own task tags with no preferences, no ceiling, engine-default memory.

## How the decision uses the file

1. **Reuse** -- if a preferred, usable local model is already loaded, serve on it
   (zero load cost).
2. **LoadLocal** -- else the first preferred local model whose resident footprint
   fits `memory_fraction * available` memory is loaded and served.
3. **Cloud** -- else the first preferred cloud model whose `cost_per_1k <=
   cost_ceiling` is used.
4. **NoFit** -- nothing usable; the caller errors.

A higher-preference local model that does not fit is skipped, falling through to
the next preference (which may be cloud). "Run it locally if the RAM is there,
else the next-best wherever it is."

## Route to ANY model = one catalog row

The route target space is the catalog (`training/catalog.yaml`); the route head
sizes itself from it (`classes_from: catalog` in `training/config.yaml`), so the
route-head logit count equals the number of catalog rows. Adding a new routable
model -- a frontier proprietary model or a new local zen tier -- is:

1. **one catalog row** -- `{id, backend, provider, tasks, cost_per_1k}`, and
2. **the serving side already dispatching it** -- the Hanzo gateway `ModelRoute`
   for a cloud provider, or `hanzo-engine` for a local model.

That is the whole change. The new row is immediately usable by the rules policy
(reference its id in a `prefer` list). The learned route head gains a
corresponding logit on the next training run; until data for it exists, that
logit is masked/untrained, so the learned policy simply never emits it while the
rules path routes to it right away. Mechanism and brain stay decomplected: the
catalog is the single source of truth both read.
