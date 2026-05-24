---
icon: lucide/chart-spline
---

# Visualization

Visualization helpers create interactive Plotly figures from `ModelStatistics` objects.

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

Metric types currently include legacy visualization labels such as
`"Solvability"` and `"Top-K"`. The legacy `"Solvability"` plot label predates
the Solv-N migration and corresponds to stock termination. Prefer the tabular
report for explicit `Tier-0` and `Solv-0[STR]` labels.

## Custom Plots

```python title="Access raw data for custom plots"
import plotly.graph_objects as go

def depth_value(depth):
    return int(str(depth).removeprefix("depth "))


depths = sorted(stats.solv_0.by_group.keys(), key=depth_value)
values = [stats.solv_0.by_group[depth].value for depth in depths]
ci_lower = [stats.solv_0.by_group[depth].ci_lower for depth in depths]
ci_upper = [stats.solv_0.by_group[depth].ci_upper for depth in depths]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=depths,
        y=values,
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [upper - value for upper, value in zip(ci_upper, values, strict=True)],
            "arrayminus": [value - lower for value, lower in zip(values, ci_lower, strict=True)],
        },
        mode="lines+markers",
        name="Solv-0[STR]",
    )
)
fig.show()
```
