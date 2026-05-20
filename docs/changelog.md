---
icon: lucide/history
---

# Changelog

## v0.6 Adapter Workflow Split

v0.6 makes the adapter API more explicit. Earlier versions centered the public workflow on `ingest`, which did two jobs at once:

1. standardize raw planner output into canonical `Route` objects
2. collect those routes onto benchmark targets and write `routes.json.gz`

That shape worked for benchmark runs, but it made RetroCast feel less like a general route-standardization library. In v0.6, standardization and benchmark collection are separate library workflows. `ingest` still exists as the project-mode convenience command that runs both.

### What Changed

| Before v0.6 | v0.6 replacement | Why |
| --- | --- | --- |
| `adapt_single_route(raw, target, adapter_name)` | `adapt_route(raw_route, adapter)` | The one-route API no longer requires target context when the raw route carries its own target. |
| `adapt_routes(raw, target, adapter_name)` | `adapt_provider_output(raw_provider_output, adapter)` | Standardization can now handle one provider output without benchmark target context. |
| `retrocast ingest` as the main way to adapt benchmark predictions | `retrocast adapt` then `retrocast collect` for ad-hoc use | Exposes standardization and benchmark collection as separate steps. |
| `retrocast ingest` for project-mode benchmark runs | still `retrocast ingest` | Project mode keeps the one-command convenience wrapper. |
| `Route.rank` | `PredictedRoute.rank` during provider-output adaptation, list order in scoring inputs | Keeps canonical `Route` free of benchmark/list-position metadata while preserving provider rank. |

Provider-output adaptation APIs now return `PredictedRoute`, an envelope around canonical `Route` chemistry. It carries provider-level metadata such as rank, score, confidence, and source row provenance. `adapt_route(...)` remains the single-payload chemistry API and returns `Route | None`; use `adapt_prediction(...)` when you explicitly want a one-off prediction envelope. Scoring artifacts remain benchmark-keyed `dict[target_id, list[Route]]` so existing evaluation semantics stay stable.

Prediction manifest `content_hash` values are now order-sensitive within each target because route list order is the ranking signal. Re-exported manifests can therefore differ from older manifests even when the route structures are otherwise identical.

`adapt_single_route(...)` and `adapt_routes(...)` emit `RetroCastFutureWarning` in v0.6 and are scheduled for removal in v0.7.

### 1-5 Minute Migration

For one raw route-like payload:

```python
from retrocast import adapt_route
from retrocast.adapters import DirectMultiStepAdapter

adapter = DirectMultiStepAdapter()
route = adapt_route(raw_route, adapter)
```

For one raw prediction payload where you want an envelope:

```python
from retrocast import adapt_prediction
from retrocast.adapters import DirectMultiStepAdapter

adapter = DirectMultiStepAdapter()
prediction = adapt_prediction(raw_route, adapter, rank=1)
```

For one raw provider output containing one or many routes:

```python
from retrocast import adapt_provider_output
from retrocast.adapters import AiZynthFinderAdapter

adapter = AiZynthFinderAdapter()
predictions = adapt_provider_output(raw_provider_output, adapter)
```

For raw output already keyed by target ID or target SMILES:

```python
from retrocast import adapt_target_keyed_provider_output, load_benchmark
from retrocast.adapters import AiZynthFinderAdapter

benchmark = load_benchmark("benchmark.json.gz")
adapter = AiZynthFinderAdapter()
predictions = adapt_target_keyed_provider_output(raw_mapping, benchmark, adapter)
```

To produce benchmark-keyed routes for scoring:

```python
from retrocast import collect_benchmark_predictions

collected = collect_benchmark_predictions(predictions, benchmark)
routes_by_target = collected.routes_by_target
```

### CLI Migration

Ad-hoc CLI users can now run the two steps explicitly:

```bash
retrocast adapt \
  --input raw_predictions.json.gz \
  --adapter aizynthfinder \
  --input-kind provider-output \
  --output route-corpus.jsonl.gz

retrocast collect \
  --input route-corpus.jsonl.gz \
  --benchmark benchmark.json.gz \
  --output routes.json.gz
```

Project-mode users can keep using `retrocast ingest`. In v0.6 it remains the one-command wrapper that adapts raw output and then collects routes onto the selected benchmark.
