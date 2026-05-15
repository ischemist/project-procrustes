---
icon: lucide/chart-spline
---

# Visualization

Visualization helpers create interactive Plotly figures from
`ModelStatistics` objects.

!!! warning "Requires visualization dependencies"

    Install with: `uv add retrocast[viz]`

## Single Model Diagnostics

```python title="Plot model performance"
from retrocast.visualization import plot_diagnostics

fig = plot_diagnostics(stats)
fig.show()
fig.write_html("model_diagnostics.html")
```

## Multi-Model Comparison

```python title="Compare multiple models"
from retrocast.visualization import plot_comparison

fig = plot_comparison(
    models_stats=[stats_a, stats_b, stats_c],
    metric_type="Top-K",
    k=1,
)
fig.show()
```

Metric types include `"Solvability"`, `"Top-K"`, and `"GT-Rank"`.

## Custom Plots

```python title="Access raw data for custom plots"
import plotly.graph_objects as go

lengths = sorted(stats.solvability.by_group.keys())
values = [stats.solvability.by_group[length].value for length in lengths]
ci_lower = [stats.solvability.by_group[length].ci_lower for length in lengths]
ci_upper = [stats.solvability.by_group[length].ci_upper for length in lengths]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=lengths,
        y=values,
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [upper - value for upper, value in zip(ci_upper, values, strict=True)],
            "arrayminus": [value - lower for value, lower in zip(values, ci_lower, strict=True)],
        },
        mode="lines+markers",
        name="Solvability",
    )
)
fig.show()
```
