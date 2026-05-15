---
icon: lucide/git-branch
---

# Adaptation

Single-route adaptation converts one raw payload into one canonical `Route`.
Provider-output adaptation converts raw provider output into `PredictedRoute`
envelopes around canonical routes, preserving rank and source metadata when
there is corpus context. Benchmark collection is separate: it assigns those
predictions to benchmark targets when you need the target-keyed `routes.json.gz`
shape used for scoring.

## Vocabulary

| Name | Raw or canonical? | Target-keyed? | Function |
| --- | --- | --- | --- |
| raw route-like payload | Raw provider output | No | `adapt_route(raw_route, adapter)` |
| raw prediction payload | Raw provider output | No | `adapt_prediction(raw_route, adapter)` |
| `provider_output` | Raw provider output | No | `adapt_provider_output(provider_output, adapter)` |
| `target_keyed_provider_output` | Raw provider output | Yes | `adapt_target_keyed_provider_output(raw, benchmark, adapter)` |
| `predictions` | Canonical `PredictedRoute` objects | No | returned by adaptation functions |
| `target_keyed_routes` | Canonical `Route` objects | Yes | `collect_benchmark_predictions(predictions, benchmark)` |

## Adapt One Route

```python title="Convert one raw route-like payload to one Route"
from retrocast import adapt_route
from retrocast.adapters import DMSAdapter

raw_data = {
    "smiles": "CCO",
    "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
}

adapter = DMSAdapter()
route = adapt_route(raw_data, adapter)

if route:
    print(f"Length: {route.length}")
    print(f"Leaves: {[m.smiles for m in route.leaves]}")
    print(f"Hash: {route.structural_signature}")
```

`adapt_route(...)` accepts one raw route-like payload and returns one
canonical `Route`, or `None` when the adapter cannot adapt it.

Use `adapt_prediction(...)` when you explicitly want the prediction envelope for
one payload:

```python title="Convert one raw prediction payload to one PredictedRoute"
from retrocast import adapt_prediction
from retrocast.adapters import DMSAdapter

adapter = DMSAdapter()
prediction = adapt_prediction(raw_data, adapter, rank=1)

if prediction:
    print(f"Rank: {prediction.rank}")
    print(f"Length: {prediction.route.length}")
```

The canonical chemistry is available at `prediction.route`; rank, score,
confidence, and source provenance live on the envelope.

## Adapt Provider Output

```python title="Convert raw provider output to predictions"
from retrocast import adapt_provider_output
from retrocast.adapters import AizynthAdapter

adapter = AizynthAdapter()

# raw_provider_output can be one planner dump, service response, script output,
# or any shape the selected adapter knows how to split into route entries.
predictions = adapt_provider_output(raw_provider_output, adapter)

print(f"Total predictions: {len(predictions)}")
```

## Collect For A Benchmark

Collection takes predictions and assigns their canonical routes to benchmark
targets. The result keeps both shapes: `predicted_routes_by_target` for
prediction metadata, and `routes_by_target` for scoring.

```python title="Adapt then collect"
from retrocast import (
    adapt_provider_output,
    collect_benchmark_predictions,
    load_benchmark,
)
from retrocast.adapters import UrsaLlmAdapter

adapter = UrsaLlmAdapter()

# Target-free adaptation: choose this when the provider output carries targets.
predictions = adapt_provider_output(raw_provider_output, adapter)

benchmark = load_benchmark("benchmark.json.gz")

# Benchmark collection: list[PredictedRoute] -> dict[target_id, list[Route]].
collected = collect_benchmark_predictions(predictions, benchmark)
routes_by_target = collected.routes_by_target
```

For already target-keyed raw output, use the benchmark-aware path instead:

```python title="Adapt target-keyed output then collect"
from retrocast import adapt_target_keyed_provider_output, collect_benchmark_predictions, load_benchmark
from retrocast.adapters import AizynthAdapter

benchmark = load_benchmark("benchmark.json.gz")
adapter = AizynthAdapter()

predictions = adapt_target_keyed_provider_output(raw_mapping, benchmark, adapter)

# Benchmark collection: list[PredictedRoute] -> dict[target_id, list[Route]].
collected = collect_benchmark_predictions(predictions, benchmark)
routes_by_target = collected.routes_by_target
```

`adapt_target_keyed_provider_output(...)` is not required for ordinary
standardization. Use it only when the raw provider output is already keyed by
target ID or target SMILES and you want RetroCast to validate those keys against
a benchmark during adaptation.

`adapt_routes(...)` and `adapt_single_route(...)` are deprecated target-local
compatibility helpers in v0.6 and will be removed in v0.7. Use `adapt_route(...)`
for one raw route, `adapt_provider_output(...)` for ordinary standardization, or
`adapt_target_keyed_provider_output(...)` when the raw provider output is keyed
by target.

!!! note "Route ordering"

    `Route.rank` was removed from the canonical route schema.
    Adaptation records source order on `PredictedRoute.rank`; scoring still
    writes explicit ranks onto `ScoredRoute`.

## Available Adapters

See the [adapter developer guide](../../developers/adapters.md#common-architecture-patterns).

`aizynth`, `askcos`, `dms`, `dreamretro`, `molbuilder`, `multistepttl`,
`paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`,
`syntheseus`, `ursa-llm`
