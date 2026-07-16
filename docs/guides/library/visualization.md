---
icon: lucide/chart-spline
---

# Visualization

The schema-2 reporting artifact is `AnalysisReport`. RetroCast keeps plotting downstream from evaluation: the core produces stable metric data, then Python, Rust, a notebook, or another application chooses how to render it.

For managed CLI workflows, start with the generated `report.md`. Use the structured report when you need custom figures or cross-model comparisons.

## Prepare Plot Data

The same metric names and strata are available from both library interfaces.

=== "Python 0.8.x"

    ```python
    metric = "solv_0[buyables]_rate"
    strata = sorted(report["by_stratum"])
    summaries = [report["by_stratum"][name][metric] for name in strata]

    values = [summary["value"] for summary in summaries]
    ci_low = [summary.get("ci_low") for summary in summaries]
    ci_high = [summary.get("ci_high") for summary in summaries]
    ```

=== "Rust 0.8.x"

    ```rust
    let metric = "solv_0[buyables]_rate";
    let plot_rows = report
        .by_stratum
        .iter()
        .filter_map(|(stratum, metrics)| {
            metrics.get(metric).map(|summary| {
                (
                    stratum.clone(),
                    summary.value,
                    summary.ci_low,
                    summary.ci_high,
                )
            })
        })
        .collect::<Vec<_>>();
    ```

=== "Python 0.7.1"

    ```python
    metric = "solv_0[buyables]_rate"
    strata = sorted(report.by_stratum)
    summaries = [report.by_stratum[name][metric] for name in strata]

    values = [summary.value for summary in summaries]
    ci_low = [summary.ci_low for summary in summaries]
    ci_high = [summary.ci_high for summary in summaries]
    ```

## Plot With Python

Plotly is not bundled with RetroCast. Install it in the environment that renders the figure:

```bash
pip install plotly
```

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=strata,
        y=values,
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [
                hi - value if hi is not None else 0
                for hi, value in zip(ci_high, values, strict=True)
            ],
            "arrayminus": [
                value - lo if lo is not None else 0
                for value, lo in zip(values, ci_low, strict=True)
            ],
        },
        name="Solv-0[buyables]",
    )
)
fig.show()
```

## Plot With Rust

`AnalysisReport` is a serializable Rust type. Feed `plot_rows` into the plotting or application framework that owns your presentation layer. For example, a service can return the complete report as JSON:

```rust
let json = serde_json::to_string_pretty(&report)?;
std::fs::write("analysis.json", json)?;
```

RetroCast does not require a Rust plotting crate and does not impose a chart style on consumers.

## Compare Reports

Use the exact serialized metric key when comparing models. Align reports by benchmark, stock label, match level, and bootstrap settings before plotting them together.

=== "Python 0.8.x"

    ```python
    metric = "solv_0[buyables]_rate"
    comparison = {
        name: model_report["metrics"][metric]["value"]
        for name, model_report in reports.items()
    }
    ```

=== "Rust 0.8.x"

    ```rust
    let comparison = reports
        .iter()
        .map(|(name, report)| (name, report.metrics[metric].value))
        .collect::<Vec<_>>();
    ```

=== "Python 0.7.1"

    ```python
    metric = "solv_0[buyables]_rate"
    comparison = {
        name: model_report.metrics[metric].value
        for name, model_report in reports.items()
    }
    ```

Confidence intervals communicate uncertainty within a model. Use paired target-level analysis when making inferential claims about the difference between two models.
