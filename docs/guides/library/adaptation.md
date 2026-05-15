---
icon: lucide/git-branch
---

# Adaptation

Adaptation converts raw provider output into canonical `Route` objects.
Benchmark collection is separate: it assigns canonical routes to benchmark
targets when you need the target-keyed `routes.json.gz` shape.

## Vocabulary

| Name | Raw or canonical? | Target-keyed? | Function |
| --- | --- | --- | --- |
| raw route-like payload | Raw provider output | No | `adapt_route(raw_route, adapter)` |
| `provider_output` | Raw provider output | No | `adapt_provider_output(provider_output, adapter)` |
| `target_keyed_provider_output` | Raw provider output | Yes | `adapt_target_keyed_provider_output(raw, benchmark, adapter)` |
| `routes` | Canonical `Route` objects | No | returned by adaptation functions |
| `target_keyed_routes` | Canonical `Route` objects | Yes | `collect_benchmark_predictions(routes, benchmark)` |

## Adapt One Route

```python title="Convert one raw route-like payload to one Route"
from retrocast import adapt_route, get_adapter

raw_data = {
    "smiles": "CCO",
    "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}],
}

adapter = get_adapter("dms")
route = adapt_route(raw_data, adapter)

if route:
    print(f"Length: {route.length}")
    print(f"Leaves: {[m.smiles for m in route.leaves]}")
    print(f"Hash: {route.structural_signature}")
```

`adapt_route(...)` accepts one raw route-like payload and returns one canonical
`Route`, or `None` when the adapter cannot adapt it.

## Adapt Provider Output

```python title="Convert raw provider output to canonical routes"
from retrocast import adapt_provider_output, deduplicate_routes, get_adapter

adapter = get_adapter("aizynth")

# raw_provider_output can be one planner dump, service response, script output,
# or any shape the selected adapter knows how to split into route entries.
routes = adapt_provider_output(raw_provider_output, adapter)
unique_routes = deduplicate_routes(routes)

print(f"Total unique routes: {len(unique_routes)}")
```

## Collect For A Benchmark

Collection takes canonical routes and assigns them to benchmark targets.

```python title="Adapt then collect"
from retrocast import (
    adapt_provider_output,
    adapt_target_keyed_provider_output,
    collect_benchmark_predictions,
    get_adapter,
    load_benchmark,
)

adapter = get_adapter("ursa-llm")

# Target-free adaptation: raw provider output -> list[Route].
routes = adapt_provider_output(raw_provider_output, adapter)

benchmark = load_benchmark("benchmark.json.gz")

# Optional benchmark-aware adaptation for already target-keyed raw output.
routes = adapt_target_keyed_provider_output(raw_mapping, benchmark, adapter)

# Benchmark collection: list[Route] -> dict[target_id, list[Route]].
collected = collect_benchmark_predictions(routes, benchmark)
routes_by_target = collected.routes_by_target
```

`adapt_target_keyed_provider_output(...)` is not required for ordinary
standardization. Use it only when the raw provider output is already keyed by
target ID or target SMILES and you want RetroCast to validate those keys against
a benchmark during adaptation.

`adapt_routes(...)` and `adapt_single_route(...)` are deprecated target-local
compatibility helpers in v0.6 and will be removed in v0.7. Use
`adapt_provider_output(...)` for ordinary standardization, or
`adapt_target_keyed_provider_output(...)` when the raw provider output is keyed
by target.

!!! note "Route ordering"

    `Route.rank` was removed from the canonical route schema. Use list order for
    raw/adapted routes, or `enumerate(routes, start=1)` when code needs an
    explicit rank value. Scoring writes explicit ranks onto `ScoredRoute`.

## Available Adapters

See the [adapter developer guide](../../developers/adapters.md#common-architecture-patterns).

`aizynth`, `askcos`, `dms`, `dreamretro`, `molbuilder`, `multistepttl`,
`paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`,
`syntheseus`, `ursa-llm`
