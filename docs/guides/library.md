# Python Library Guide

RetroCast is designed as a modular Python library. While the CLI handles file-based workflows, the Python API allows you to integrate RetroCast's standardization, scoring, and analysis logic directly into your research pipelines (e.g., Jupyter notebooks or internal evaluation loops).

## 1. Standardization (Adapters)

The most common use case is converting raw model outputs into the canonical `Route` format. This creates a unified interface for any downstream task.

### Adapting a Single Route

```python
from retrocast import adapt_single_route, TargetInput

# 1. Define the target context. ID is a unique identifier for the target molecule.
target = TargetInput(id="mol-1", smiles="CCO")

# 2. Provide raw data (e.g., a generic dictionary from a model output)
raw_data = {
    "smiles": "CCO",
    "children": [{"smiles": "CC", "children": []}, {"smiles": "O", "children": []}]
}

# 3. Cast to Route
# The adapter automatically handles schema validation and tree construction
route = adapt_single_route(raw_data, target, adapter_name="dms")

if route:
    print(f"Depth: {route.length}")
    print(f"Leaves: {[m.smiles for m in route.leaves]}")
```

### Adapting Batch Predictions

```python
from retrocast import adapt_routes, deduplicate_routes, TargetInput

targets = [TargetInput(id=f"t{i}", smiles=s) for i, s in enumerate(smiles_list)]
all_routes = []

for target, raw_output in zip(targets, model_outputs):
    # Adapt
    routes = adapt_routes(raw_output, target, adapter_name="aizynth")
    
    # Deduplicate based on topological signature
    unique_routes = deduplicate_routes(routes)
    
    all_routes.extend(unique_routes)
```

## 2. Evaluation Workflow

You can run the full scoring pipeline in memory without creating intermediate files.

### Step A: Score Predictions

```python
from retrocast.api import score_predictions, load_benchmark, load_stock_file

# 1. Load Resources
benchmark = load_benchmark("data/1-benchmarks/definitions/paroutes-n1.json.gz")
stock = load_stock_file("data/1-benchmarks/stocks/zinc-stock.txt")

# 2. Prepare Predictions (dict: target_id -> list[Route])
# Assume 'my_routes' was created using the adaptation steps above
predictions = {"n1-0001": [route1, route2], ...}

# 3. Run Scoring
results = score_predictions(
    benchmark=benchmark,
    predictions=predictions,
    stock=stock,
    model_name="Experimental-Model-V1"
)

# Access granular results
t1_eval = results.results["n1-0001"]
print(f"Is solved: {t1_eval.is_solvable}")
print(f"Ground truth rank: {t1_eval.gt_rank}")
```

### Step B: Compute Statistics

RetroCast uses bootstrap resampling to calculate confidence intervals (95% CI) for all metrics.

```python
from retrocast.api import compute_model_statistics

# Compute stats from the scored results
stats = compute_model_statistics(results, n_boot=10000, seed=42)

# Access aggregated metrics
solvability = stats.solvability.overall
print(f"Solvability: {solvability.value:.1%} "
      f"[{solvability.ci_lower:.1%}, {solvability.ci_upper:.1%}]")

# Access stratified metrics (e.g., by route length)
for length, metric in stats.solvability.by_group.items():
    print(f"Length {length}: {metric.value:.1%}")
```

## 3. Visualization

You can generate Plotly figures directly from the `ModelStatistics` object.

```python
from retrocast.visualization import plot_diagnostics, plot_comparison

# 1. Diagnostic Plot (Solvability & Top-K vs Route Length)
fig = plot_diagnostics(stats)
fig.show()

# 2. Compare Multiple Models
# Assume stats_a and stats_b are ModelStatistics objects
fig_comp = plot_comparison(
    models_stats=[stats_a, stats_b], 
    metric_type="Top-K", 
    k=1
)
fig_comp.show()
```

## Reference: Available Adapters

To see the list of registered adapters programmatically:

```python
from retrocast import ADAPTER_MAP

print(list(ADAPTER_MAP.keys()))
# ['aizynth', 'dms', 'retrostar', 'askcos', ...]
```
