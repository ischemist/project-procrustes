---
icon: lucide/chart-spline
---

# Visualization

The primary schema-2 reporting artifact is `AnalysisReport`. The CLI writes both `analysis.json.gz` and a markdown `report.md` from that object.

For most workflows, start with the generated markdown report. Use Python plotting when you need custom figures or comparisons.

## Plot From AnalysisReport

`AnalysisReport` is intentionally simple: metric names map to `MetricSummary` values. This makes custom plotting straightforward.

```python
import plotly.graph_objects as go

metric = "solv_0[buyables]_rate"
strata = sorted(report.by_stratum)
values = [report.by_stratum[stratum][metric].value for stratum in strata]
ci_low = [report.by_stratum[stratum][metric].ci_low for stratum in strata]
ci_high = [report.by_stratum[stratum][metric].ci_high for stratum in strata]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=strata,
        y=values,
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [hi - v if hi is not None else 0 for hi, v in zip(ci_high, values, strict=True)],
            "arrayminus": [v - lo if lo is not None else 0 for v, lo in zip(values, ci_low, strict=True)],
        },
        name="Solv-0[buyables]",
    )
)
fig.show()
```

## Built-In Plot Helpers

RetroCast still exposes Plotly helpers for legacy model-statistics objects:

```python
from retrocast.visualization import plot_comparison, plot_diagnostics

fig = plot_diagnostics(stats)
fig.show()

comparison = plot_comparison([stats_a, stats_b], metric_type="Top-K", k=10)
comparison.show()
```

These helpers are useful for older analysis scripts. For new schema-2 reports, prefer plotting directly from `AnalysisReport` until the visualization layer is fully rebuilt around the v2 metric names.

## Compare Reports

The CLI includes one schema-2 comparison command for Pareto-frontier plots:

```bash
retrocast compare pareto-frontier compare.yaml --no-open
```

```yaml title="compare.yaml"
benchmark: small
stock: buyables
metric: solv_0[buyables]_rate
output_dir: comparisons
sources:
  - root: data/retrocast
    models:
      - name: model-a
        hourly_cost: 1.0
      - name: model-b
        hourly_cost: 0.5
```

Use the exact metric key from `AnalysisReport.metrics`.
