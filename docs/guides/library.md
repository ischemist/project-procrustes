---
icon: lucide/cable
---

# Python Library Guide

RetroCast is designed as a modular Python library. While the CLI handles file-based workflows, the Python API allows you to integrate RetroCast's standardization, scoring, and analysis logic directly into your research pipelines.

!!! tip "When to use the Python API"

    - Jupyter notebooks for interactive analysis
    - Custom evaluation loops
    - Integration with existing research pipelines
    - Programmatic access to metrics without file I/O

## Installation

=== "uv (recommended)"

    ```bash
    uv add retrocast
    ```
    
    For visualization support:
    
    ```bash
    uv add retrocast[viz]
    ```


=== "pip"

    ```bash
    pip install retrocast
    ```

    For visualization support:
    
    ```bash
    pip install retrocast[viz]
    ```
## Standardization And Benchmark Collection

Before v0.6, RetroCast's public mental model centered on `ingest`: take raw
predictions, adapt them, and write benchmark-keyed `routes.json.gz`. That is
still useful, but it hides two different operations.

In v0.6, those operations are exposed as separate library workflows:

| Workflow | Input | Output | Requires benchmark targets? |
| --- | --- | --- | --- |
| Adaptation | Raw provider output | Canonical `Route` objects | No |
| Benchmark collection | Canonical `Route` objects plus a benchmark | Target-keyed route mapping | Yes |

Use adaptation when you want the general library promise: "give me whatever the
planner emitted and I will give you standard `Route` objects." Use benchmark
collection only when you need the `routes.json.gz` shape for benchmark scoring
or aggregate target-level statistics.

RetroCast uses two naming axes for adapter workflows:

| Name | Raw or canonical? | Target-keyed? | Function |
| --- | --- | --- | --- |
| raw route-like payload | Raw provider output | No | `adapt_route(raw_route, adapter)` |
| `provider_output` | Raw provider output | No | `adapt_provider_output(provider_output, adapter)` |
| `target_keyed_provider_output` | Raw provider output | Yes | `adapt_target_keyed_provider_output(raw, benchmark, adapter)` |
| `routes` | Canonical `Route` objects | No | returned by adaptation functions |
| `target_keyed_routes` | Canonical `Route` objects | Yes | `collect_benchmark_predictions(routes, benchmark)` |

### Adapt One Route

```python title="Convert one raw route-like payload to one Route" hl_lines="1 10"
from retrocast import adapt_route, get_adapter

raw_data = {
    "smiles": "CCO",
    "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}], # (1)!
}

adapter = get_adapter("dms")
route = adapt_route(raw_data, adapter) # (2)!

if route:
    print(f"Length: {route.length}")
    print(f"Leaves: {[m.smiles for m in route.leaves]}")
    print(f"Hash: {route.content_hash}") # (4)!
```

1. Raw route-like payload format varies by model - this is a DMS example
2. `adapt_route(...)` returns the first successfully adapted `Route`, or `None`
3. Adapters infer target identity from the route payload when it includes it
4. Content hash used for deduplication

### Adapt Provider Output

```python title="Convert raw provider output to canonical routes" hl_lines="1 8"
from retrocast import adapt_provider_output, deduplicate_routes, get_adapter

adapter = get_adapter("aizynth")

# raw_provider_output can be one planner dump, service response, script output,
# or any shape the selected adapter knows how to split into route entries.
routes = adapt_provider_output(raw_provider_output, adapter) # (1)!
unique_routes = deduplicate_routes(routes) # (2)!

print(f"Total unique routes: {len(unique_routes)}")
```

1. `adapt_provider_output` returns successful canonical `Route` objects
2. Deduplication uses cryptographic hashing of route topology

!!! info "Available adapters"

    See [full adapter list](../developers/adapters.md#common-architecture-patterns):
    
    `aizynth`, `askcos`, `dms`, `dreamretro`, `molbuilder`, `multistepttl`, `paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`, `syntheseus`, `ursa-llm`

### Collect For A Benchmark

Collection is a separate step. It takes canonical routes and assigns them to
benchmark targets, producing the target-keyed shape used by scoring.

```python title="Adapt then collect" hl_lines="1 13"
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

`adapt_routes(...)` is a lower-level target-local compatibility helper for
callers that already have a single target context in memory.
`adapt_single_route(...)` is deprecated in v0.6 and will be removed in v0.7; use
`adapt_route(...)` for target-free single-route adaptation.

!!! note "Route ordering"

    `Route.rank` was removed from the canonical route schema. Use list order for
    raw/adapted routes, or `enumerate(routes, start=1)` when code needs an
    explicit rank value. Scoring writes explicit ranks onto `ScoredRoute`.

## Evaluation Workflow

Run the full scoring pipeline in memory without creating intermediate files.

### Tracking runtime performance

RetroCast provides a context manager to track wall and cpu time per target. These metrics are automatically aggregated into the final report.

```python title="Measure inference time"
from retrocast.utils import ExecutionTimer

timer = ExecutionTimer()

for target in benchmark.targets.values():
    # context manager captures time for this specific block
    with timer.measure(target.id):
        raw_output = model.predict(target.smiles)
    
    # ... process/store results ...

# convert to a portable stats object
exec_stats = timer.to_model()
```

### Score Predictions

```python title="Evaluate routes against stock" hl_lines="4 5 8 13"
from retrocast.api import score_predictions, load_benchmark, load_stock_file

# 1. Load resources
benchmark = load_benchmark("data/1-benchmarks/definitions/mkt-cnv-160.json.gz") # (1)!
stock = load_stock_file("data/1-benchmarks/stocks/buyables-stock.txt") # (2)!

# 2. Prepare predictions (dict: target_id -> list[Route])
predictions = {"target-001": [route1, route2], "target-002": [route3]} # (3)!

# 3. Run scoring
results = score_predictions(
    benchmark=benchmark,
    predictions=predictions, # (4)!
    stock=stock,
    model_name="Experimental-Model-V1",
    execution_stats=exec_stats # (5)!
)

# 4. Access granular results
for target_id, evaluation in results.results.items():
    print(f"\nTarget: {target_id}")
    print(f"  Is solvable: {evaluation.is_solvable}") # (6)!
    print(f"  Top-1 solved: {evaluation.top_1_is_solved}")
    print(f"  GT rank: {evaluation.gt_rank}") # (7)!
    print(f"  Best route length: {evaluation.best_route_length}")
```

1. Load benchmark definition (targets + ground truth)
2. Load stock (one canonical SMILES per line)
3. Predictions must be a dict mapping target IDs to lists of Route objects
4. Each route is evaluated: are all leaves in stock?
5. Optional: Pass the `ExecutionStats` object here to include timing data in the final analysis.
6. `is_solvable = True` if at least one route is solved
7. Rank of ground truth route in predictions (None if not found)

### Compute Statistics

RetroCast uses [bootstrap resampling](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) to calculate confidence intervals (95% CI) for all metrics.

```python title="Generate statistical summary" hl_lines="4 7 13"
from retrocast.api import compute_model_statistics

# Compute stats from scored results
stats = compute_model_statistics(results, n_boot=10000, seed=42) # (1)!

# Access overall metrics
solvability = stats.solvability.overall # (2)!
print(f"Solvability: {solvability.value:.1%} "
      f"[{solvability.ci_lower:.1%}, {solvability.ci_upper:.1%}]")

# Access Top-K metrics
for k in [1, 5, 10]:
    topk = stats.top_k[k].overall # (3)!
    print(f"Top-{k}: {topk.value:.1%} [{topk.ci_lower:.1%}, {topk.ci_upper:.1%}]")

# Access stratified metrics (by route length)
print("\nStratified by length:")
for length, metric in stats.solvability.by_group.items(): # (4)!
    print(f"  Length {length}: {metric.value:.1%} [{metric.ci_lower:.1%}, {metric.ci_upper:.1%}]")
```

1. Bootstrap resampling with 10,000 iterations for robust CIs
2. Overall solvability across all targets
3. Top-K accuracy: percentage of targets with ≥1 solved route in top K
4. Performance broken down by route length

??? example "Example output"

    ```
    Solvability: 45.3% [42.1%, 48.6%]
    Top-1: 23.5% [20.8%, 26.3%]
    Top-5: 38.2% [35.1%, 41.4%]
    Top-10: 42.7% [39.5%, 45.9%]
    
    Stratified by length:
      Length 2: 65.2% [58.3%, 72.1%]
      Length 3: 52.8% [47.2%, 58.4%]
      Length 4: 38.1% [32.5%, 43.8%]
      Length 5: 24.3% [19.1%, 29.6%]
    ```

## Visualization

Generate interactive Plotly figures directly from `ModelStatistics` objects.

!!! warning "Requires visualization dependencies"

    Install with: `uv add retrocast[viz]`

### Single Model Diagnostics

```python title="Plot model performance" hl_lines="4"
from retrocast.visualization import plot_diagnostics

# Generate diagnostic plot (Solvability & Top-K vs Route Length)
fig = plot_diagnostics(stats) # (1)!
fig.show()

# Save to file
fig.write_html("model_diagnostics.html")
```

1. Creates an interactive plot with stratified performance metrics

### Multi-Model Comparison

```python title="Compare multiple models" hl_lines="5-9"
from retrocast.visualization import plot_comparison

# Assume stats_a and stats_b are ModelStatistics objects from different models
fig_comp = plot_comparison(
    models_stats=[stats_a, stats_b, stats_c], # (1)!
    metric_type="Top-K",  # (2)!
    k=1
)
fig_comp.show()
```

1. List of `ModelStatistics` objects to compare
2. Metric types: `"Solvability"`, `"Top-K"`, `"GT-Rank"`

### Custom Visualization

```python title="Access raw data for custom plots"
import plotly.graph_objects as go

# Extract stratified solvability data
lengths = sorted(stats.solvability.by_group.keys())
values = [stats.solvability.by_group[l].value for l in lengths]
ci_lower = [stats.solvability.by_group[l].ci_lower for l in lengths]
ci_upper = [stats.solvability.by_group[l].ci_upper for l in lengths]

# Create custom plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=lengths,
    y=values,
    error_y=dict(
        type='data',
        symmetric=False,
        array=[u - v for u, v in zip(ci_upper, values)],
        arrayminus=[v - l for v, l in zip(values, ci_lower)]
    ),
    mode='lines+markers',
    name='Solvability'
))
fig.update_layout(
    title="Solvability by Route Length",
    xaxis_title="Route Length",
    yaxis_title="Solvability (%)"
)
fig.show()
```

## Working with Route Objects

The `Route` object is the core data structure. Here are common operations:

### Route Properties

```python title="Access route properties"
from retrocast import Route

# Assuming you have a route object
print(f"Target: {route.target.smiles}")
print(f"Length: {route.length}") # (1)!
print(f"Number of leaves: {len(route.leaves)}") # (2)!
print(f"Content hash: {route.content_hash}") # (3)!

# Check if route is linear or convergent
has_convergent_step = any(
    step.is_convergent 
    for mol in route.target.get_all_molecules() 
    if mol.synthesis_step
    for step in [mol.synthesis_step]
)
print(f"Has convergent step: {has_convergent_step}")
```

1. Longest path from target to any leaf
2. All starting materials (leaf molecules)
3. SHA256 hash of route topology for deduplication

### Serialization

```python title="Save and load routes"
import json
import gzip

# Save routes to JSON
routes_dict = {
    "target-001": [r.model_dump() for r in route_list]
}

with gzip.open("routes.json.gz", "wt") as f:
    json.dump(routes_dict, f, indent=2)

# Load routes from JSON
with gzip.open("routes.json.gz", "rt") as f:
    data = json.load(f)
    loaded_routes = [Route.model_validate(r) for r in data["target-001"]]
```

### Route Comparison

```python title="Compare routes"
from retrocast.metrics import compute_route_similarity

# Check if two routes are topologically identical
route1_sig = route1.signature
route2_sig = route2.signature
are_identical = route1_sig == route2_sig

# Get shared starting materials
leaves1 = {m.inchikey for m in route1.leaves}
leaves2 = {m.inchikey for m in route2.leaves}
shared_leaves = leaves1 & leaves2
print(f"Shared starting materials: {len(shared_leaves)}")
```

---

## Complete Example

Putting it all together in a Jupyter notebook workflow:

```python title="End-to-end evaluation pipeline"
from retrocast import adapt_routes, deduplicate_routes, TargetInput
from retrocast.utils import ExecutionTimer
from retrocast.api import score_predictions, compute_model_statistics
from retrocast.api import load_benchmark, load_stock_file
from retrocast.visualization import plot_diagnostics


# 1. Load benchmark and stock
benchmark = load_benchmark("data/1-benchmarks/definitions/mkt-cnv-160.json.gz")
stock = load_stock_file("data/1-benchmarks/stocks/buyables-stock.txt")

# 2. Inference Loop with Timing
predictions = {}
timer = ExecutionTimer() # (2)!

for target in benchmark.targets.values():
    target_input = TargetInput(id=target.id, smiles=target.smiles)
    
    # Measure inference time
    with timer.measure(target.id): # (3)!
        raw_output = get_model_predictions(target.smiles)
    
    # Adapt and store
    routes = adapt_routes(raw_output, target_input, adapter_name="aizynth")
    unique_routes = deduplicate_routes(routes)
    predictions[target.id] = list(unique_routes)[:10]

# 3. Score predictions (including runtime stats)
results = score_predictions(
    benchmark=benchmark,
    predictions=predictions,
    stock=stock,
    model_name="MyModel-v1.0",
    execution_stats=timer.to_model() # (4)!
)

# 4. Compute statistics
stats = compute_model_statistics(results, n_boot=10000, seed=42)

# 5. Print summary
print(f"Overall Solvability: {stats.solvability.overall.value:.1%}")
print(f"Mean Wall Time: {stats.mean_wall_time:.2f}s") # (5)!

# 6. Generate visualizations
fig = plot_diagnostics(stats)
fig.write_html("diagnostics.html")
```

2. Initialize the timer before the loop
3. Wrap the expensive model call in the context manager
4. Pass the collected stats to the scorer
5. Access aggregated timing metrics in the final statistics object

---

## API Reference

### Core Functions

| Function | Purpose | Returns |
|:---------|:--------|:--------|
| `adapt_route(raw, adapter)` | Adapt the first successful route from raw provider output | `Route \| None` |
| `adapt_provider_output(raw, adapter)` | Adapt one raw provider output | `list[Route]` |
| `adapt_target_keyed_provider_output(raw, benchmark, adapter)` | Adapt raw output keyed by target id or smiles | `list[Route]` |
| `adapt_routes(raw, target, adapter)` | Lower-level target-local compatibility helper | `list[Route]` |
| `adapt_single_route(raw, target, adapter)` | Deprecated in v0.6; use `adapt_route(raw, adapter)` | `Route \| None` |
| `collect_benchmark_predictions(routes, benchmark)` | Collect a route corpus onto a benchmark | `CollectedBenchmarkRoutes` |
| `deduplicate_routes(routes)` | Remove duplicate routes | `list[Route]` |
| `score_predictions(benchmark, predictions, stock)` | Evaluate routes | `ScoredResults` |
| `compute_model_statistics(results, n_boot)` | Bootstrap statistics | `ModelStatistics` |
| `load_benchmark(path)` | Load benchmark definition | `Benchmark` |
| `load_stock_file(path)` | Load stock molecules | `set[str]` |

### Visualization Functions

| Function | Purpose | Returns |
|:---------|:--------|:--------|
| `plot_diagnostics(stats)` | Single model performance | `plotly.Figure` |
| `plot_comparison(models_stats, metric_type, k)` | Multi-model comparison | `plotly.Figure` |

### Available Adapters

To see registered adapters programmatically:

```python
from retrocast.adapters import ADAPTER_MAP

print("Available adapters:")
for name in ADAPTER_MAP.keys():
    print(f"  - {name}")
```

**Supported:** `aizynth`, `askcos`, `dms`, `dreamretro`, `molbuilder`, `multistepttl`, `paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`, `syntheseus`, `ursa-llm`
