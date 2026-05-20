---
icon: lucide/git-branch
---

# Adaptation

Adaptation is the part of RetroCast that standardizes planner output. It has two layers: adapters know how to parse one model's native format, and workflow helpers decide whether you are adapting one payload, a whole provider output, or collecting predictions onto a benchmark.

## Choose A Workflow

| Goal | Function or command | Input shape | Output shape |
| --- | --- | --- | --- |
| Standardize one route-like payload for your own code | `adapt_route(raw_route, adapter)` | One raw route-like payload | `Route` or `None` |
| Standardize one payload and keep rank, score, or source metadata | `adapt_prediction(raw_route, adapter)` | One raw route-like payload | `PredictedRoute` or `None` |
| Standardize a provider artifact containing one or many routes | `adapt_provider_output(raw_provider_output, adapter)` | Raw provider output | `list[PredictedRoute]` |
| Standardize provider output already keyed by benchmark target | `adapt_target_keyed_provider_output(raw_mapping, benchmark, adapter)` | Raw provider output keyed by target ID or target SMILES | `list[PredictedRoute]` |
| Produce benchmark-keyed routes for scoring | `collect_benchmark_predictions(predictions, benchmark)` | `list[PredictedRoute]` plus `BenchmarkSet` | `CollectedBenchmarkRoutes` |
| Run the full file-based benchmark pipeline | `retrocast ingest` | Raw files in `2-raw/` plus project config | `routes.json.gz` |

Use `adapt_route(...)` when you only care about the canonical chemistry tree. Use `adapt_provider_output(...)` when you are standardizing a model's prediction artifact and want RetroCast to preserve ranking and source provenance. Use `adapt_target_keyed_provider_output(...)` when the raw artifact already groups predictions by benchmark target and you want key validation during adaptation. Use `collect_benchmark_predictions(...)` only when you need to produce the benchmark-keyed `routes.json.gz` shape for scoring.

## Terms

`Route` is the canonical chemistry object. It contains the target molecule, reaction steps, reactants, route metadata, and structural hashes. It should not own provider-list concepts such as rank.

`PredictedRoute` is a prediction envelope around a `Route`. It keeps provider-level data such as rank, score, confidence, source row index, source record ID, and source key. The chemistry is still available at `prediction.route`.

Single-route adaptation is the workflow for one raw route-like payload. It calls the selected adapter's `cast(...)` method and returns one canonical `Route` through `adapt_route(...)`, or one `PredictedRoute` through `adapt_prediction(...)`.

Provider-output adaptation is the workflow for a larger provider artifact, such as a planner dump, service response, JSONL export, or list of predictions. RetroCast calls the adapter's `iter_raw_entries(...)` method to split that artifact into route-like entries, calls `cast(...)` on each entry, then wraps each successful `Route` in a `PredictedRoute`.

Target-keyed provider-output adaptation is the benchmark-aware version of provider-output adaptation. It is useful when the raw artifact is a mapping keyed by benchmark target ID or target SMILES. RetroCast uses the benchmark to resolve and validate those keys before adapting each target's payload, but the result is still a flat `list[PredictedRoute]`; collection is still the step that creates benchmark-keyed routes for scoring.

Benchmark collection is separate from adaptation. It takes canonical predictions plus a `BenchmarkSet`, matches predictions to benchmark targets, deduplicates per target, and returns the target-keyed shape used by scoring.

## Adapt One Route

```python title="Convert one raw route-like payload to one Route"
from retrocast import adapt_route
from retrocast.adapters import DirectMultiStepAdapter

raw_data = {
    "smiles": "CCO",
    "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
}

adapter = DirectMultiStepAdapter()
route = adapt_route(raw_data, adapter)

if route:
    print(f"Length: {route.length}")
    print(f"Leaves: {[m.smiles for m in route.leaves]}")
    print(f"Hash: {route.structural_signature}")
```

`adapt_route(...)` accepts one raw route-like payload and returns one canonical `Route`, or `None` when the adapter cannot adapt it.

Use `adapt_prediction(...)` when you explicitly want the prediction envelope for one payload:

```python title="Convert one raw prediction payload to one PredictedRoute"
from retrocast import adapt_prediction
from retrocast.adapters import DirectMultiStepAdapter

adapter = DirectMultiStepAdapter()
prediction = adapt_prediction(raw_data, adapter, rank=1)

if prediction:
    print(f"Rank: {prediction.rank}")
    print(f"Length: {prediction.route.length}")
```

The canonical chemistry is available at `prediction.route`; rank, score, confidence, and source provenance live on the envelope.

## Adapt Provider Output

Provider-output adaptation is a built-in RetroCast workflow. Users call `adapt_provider_output(...)`; adapter authors provide `iter_raw_entries(...)` and `cast(...)`; RetroCast handles iteration, failure accounting, and wrapping successful routes in `PredictedRoute`.

```python title="Convert raw provider output to predictions"
from retrocast import adapt_provider_output
from retrocast.adapters import AiZynthFinderAdapter

adapter = AiZynthFinderAdapter()

# raw_provider_output can be one planner dump, service response, script output,
# or any shape the selected adapter knows how to split into route entries.
predictions = adapt_provider_output(raw_provider_output, adapter)

print(f"Total predictions: {len(predictions)}")
```

## Collect For A Benchmark

Collection takes predictions and assigns their canonical routes to benchmark targets. The result keeps both shapes: `predicted_routes_by_target` for prediction metadata, and `routes_by_target` for scoring.

```python title="Adapt then collect"
from retrocast import (
    adapt_provider_output,
    collect_benchmark_predictions,
    load_benchmark,
)
from retrocast.adapters import UrsaAdapter

adapter = UrsaAdapter()

# Target-free adaptation: choose this when the provider output carries targets.
predictions = adapt_provider_output(raw_provider_output, adapter)

benchmark = load_benchmark("benchmark.json.gz")

# Benchmark collection: list[PredictedRoute] -> dict[target_id, list[Route]].
collected = collect_benchmark_predictions(predictions, benchmark)
routes_by_target = collected.routes_by_target
```

For already target-keyed raw output, use the benchmark-aware adaptation path before collection:

```python title="Adapt target-keyed output then collect"
from retrocast import adapt_target_keyed_provider_output, collect_benchmark_predictions, load_benchmark
from retrocast.adapters import AiZynthFinderAdapter

benchmark = load_benchmark("benchmark.json.gz")
adapter = AiZynthFinderAdapter()

predictions = adapt_target_keyed_provider_output(raw_mapping, benchmark, adapter)

# Benchmark collection: list[PredictedRoute] -> dict[target_id, list[Route]].
collected = collect_benchmark_predictions(predictions, benchmark)
routes_by_target = collected.routes_by_target
```

This is not exactly the same as `adapt_provider_output(...)` followed by `collect_benchmark_predictions(...)`. Both paths eventually produce predictions that can be collected onto a benchmark, but they assume different raw artifact shapes.

Use `adapt_provider_output(...)` when your planner writes a streamable corpus, such as a JSONL file, a list of records, or any format where each entry carries enough target information for the adapter to parse it. This is the more flexible shape when you want to append, shard, or process predictions in chunks.

Use `adapt_target_keyed_provider_output(...)` when your planner writes a mapping like `{target_id: raw_predictions_for_target}` or `{target_smiles: raw_predictions_for_target}`. This can simplify model runners because target association is explicit at the top level, and RetroCast can reject unknown or ambiguous keys early.

`adapt_routes(...)` and `adapt_single_route(...)` are deprecated target-local compatibility helpers in v0.6 and will be removed in v0.7. Use `adapt_route(...)` for one raw route, `adapt_provider_output(...)` for ordinary standardization, or `adapt_target_keyed_provider_output(...)` when the raw provider output is keyed by target.

!!! note "Route ordering"

    `Route.rank` was removed from the canonical route schema.
    Adaptation records source order on `PredictedRoute.rank`; scoring still
    writes explicit ranks onto `ScoredRoute`.

## Available Adapters

See the [adapter developer guide](../../developers/adapters.md#common-architecture-patterns).

`aizynthfinder`, `askcos`, `directmultistep`, `dreamretroer`, `molbuilder`, `multistepttl`, `paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`, `syntheseus`, `ursa`
