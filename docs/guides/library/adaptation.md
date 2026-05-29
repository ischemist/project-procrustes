---
icon: lucide/git-branch
---

# Adaptation

Adaptation is the part of RetroCast that turns planner-specific output into schema-2 objects.

Adapters know raw formats. Workflow helpers decide whether failures are skipped or preserved.

## Choose A Workflow

| Goal | Function | Input shape | Output shape |
| --- | --- | --- | --- | --- |
| Inspect one raw route record | `adapt_route(raw_route, adapter)` | one raw route record | `Route | None` |
| Inspect only successful routes from a raw artifact | `adapt_routes(raw_payload, adapter)` | raw artifact | `list[Route]` |
| Preserve ranked prediction slots, including failures | `adapt_candidates(raw_payload, adapter)` | raw artifact | `list[Candidate]` |
| Map candidates onto a task | `collect_candidates(candidates, task)` | `list[Candidate]` plus `Task` | `dict[target_id, list[Candidate]]` |
| Run adapt and collect together | `ingest_candidates(raw_payload, adapter, task)` | raw artifact plus `Task` | collected candidates |

Use route-only adaptation when you only want to inspect chemistry. Use candidate-preserving adaptation when you are evaluating a planner and failed prediction slots must remain visible.

## Terms

`raw_payload` is the whole raw artifact passed to `iter_raw_routes(...)`.

`RawRouteEntry` is the envelope yielded by the adapter. It carries one raw route record plus provenance such as source order and target hints.

`RawRouteEntry.payload` is the raw route record passed to `cast(...)`.

`Route` is the successful chemistry object returned by `cast(...)`.

`Candidate` is produced by `adapt_candidates(...)`. It stores rank and either a `Route` or a `FailureRecord`.

## Adapt One Route

```python
from retrocast import adapt_route
from retrocast.adapters import DirectMultiStepAdapter

adapter = DirectMultiStepAdapter()
route = adapt_route(raw_route_record, adapter)

if route is not None:
    print(route.target.smiles)
    print(route.depth())
    print([leaf.value.smiles for leaf in route.leaves()])
```

`adapt_route(...)` returns `None` for expected adapter or chemistry failures. It is convenient for ad hoc inspection, but it is not benchmark-safe because failures disappear.

## Adapt Successful Routes

```python
from retrocast import adapt_routes, get_adapter

adapter = get_adapter("paroutes")
routes = adapt_routes(raw_payload, adapter, max_routes=50)
```

`max_routes` counts successful routes because a `Route` is, by definition, a valid adapted route. Malformed raw route records are skipped and do not consume the limit. Use `adapt_candidates(..., max_candidates=N)` when you need to bound the first N raw prediction slots or preserve failures for evaluation.

## Preserve Candidate Slots

```python
from retrocast import adapt_candidates, get_adapter

adapter = get_adapter("paroutes")

candidates = adapt_candidates(raw_payload, adapter, max_candidates=50)
```

`max_candidates` means the first N raw prediction slots. A failed slot becomes:

```python
Candidate(rank=rank, failure=FailureRecord(...))
```

This is the right path for Solv-0 and MRR accounting.

## Collect For A Task

```python
from retrocast import collect_candidates
from retrocast.io import load_benchmark

task = load_benchmark("benchmark.json.gz")
predictions = collect_candidates(candidates, task)
```

Collection maps candidates to target ids. Successful candidates are placed by route target identity. Failed candidates are placed by `FailureRecord.target_id` or `FailureRecord.target_inchikey`.

## Ingest In One Step

```python
from retrocast import get_adapter, ingest_candidates
from retrocast.io import load_benchmark

task = load_benchmark("benchmark.json.gz")
adapter = get_adapter("aizynthfinder")

predictions = ingest_candidates(raw_payload, adapter, task, max_candidates=50)
```

`ingest_candidates(...)` is just `adapt_candidates(...) + collect_candidates(...)`.

## Adapt Modes

```python
routes = adapt_routes(raw_payload, adapter, mode="strict")
routes = adapt_routes(raw_payload, adapter, mode="prune")
```

`strict` rejects raw route records with invalid chemistry or impossible route structure.

`prune` lets adapters return the longest valid prefix route when an invalid branch can be removed unambiguously. Not every adapter supports useful pruning.

## Available Adapters

Use `get_adapter(...)` or import adapter classes directly.

```python
from retrocast import get_adapter

adapter = get_adapter("paroutes")
```

See [Writing a Custom Adapter](../../developers/adapters.md) for adapter contracts and raw-shape patterns.
