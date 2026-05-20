---
icon: lucide/book-open-text
---

# Library Reference

## Core Functions

| Function | Purpose | Returns |
| :-- | :-- | :-- |
| `adapt_route(raw, adapter)` | Adapt one raw route-like payload | `Route \| None` |
| `adapt_prediction(raw, adapter)` | Adapt one raw prediction payload | `PredictedRoute \| None` |
| `adapt_provider_output(raw, adapter)` | Adapt one raw provider output | `list[PredictedRoute]` |
| `adapt_target_keyed_provider_output(raw, benchmark, adapter)` | Adapt raw output keyed by target id or smiles | `list[PredictedRoute]` |
| `adapt_routes(raw, target, adapter)` | Deprecated in v0.6; use provider-output workflows | `list[Route]` |
| `adapt_single_route(raw, target, adapter)` | Deprecated in v0.6; use `adapt_route(raw, adapter)` | `Route \| None` |
| `collect_benchmark_predictions(predictions, benchmark)` | Collect predictions onto benchmark targets | `CollectedBenchmarkRoutes` |
| `deduplicate_routes(routes)` | Remove duplicate routes | `list[Route]` |
| `score_predictions(model_name, benchmark, predictions, stock)` | Evaluate routes | `ScoredResults` |
| `compute_model_statistics(results, n_boot)` | Bootstrap statistics | `ModelStatistics` |
| `load_benchmark(path)` | Load benchmark definition | `Benchmark` |
| `load_stock_file(path)` | Load stock molecules | `set[str]` |

## Visualization Functions

| Function                                        | Purpose                  | Returns         |
| :---------------------------------------------- | :----------------------- | :-------------- |
| `plot_diagnostics(stats)`                       | Single model performance | `plotly.Figure` |
| `plot_comparison(models_stats, metric_type, k)` | Multi-model comparison   | `plotly.Figure` |

## Adapters

```python
from retrocast.adapters import ADAPTER_MAP

for name in ADAPTER_MAP:
    print(name)
```

Supported adapters:

`aizynth`, `askcos`, `dms`, `dreamretro`, `molbuilder`, `multistepttl`, `paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`, `syntheseus`, `ursa-llm`
