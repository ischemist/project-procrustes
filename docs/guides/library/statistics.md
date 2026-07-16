---
icon: lucide/chart-no-axes-combined
---

# Statistics

Analysis turns an `Evaluation` into a small `AnalysisReport`. It computes overall metric summaries, route-depth strata, bootstrap confidence intervals, and runtime summaries.

## Analyze An Evaluation

=== "Python 0.8.x"

    ```python
    report = retrocast.analyze(
        evaluation,
        ks=[1, 5, 10, 50],
        prefix_depths=[1, 2, 3],
        n_boot=10_000,
        seed=42,
        workers=12,
    )
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::analyze::analyze;

    let report = analyze(
        &evaluation,
        &[1, 5, 10, 50],
        &[1, 2, 3],
        10_000,
        42,
        12,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.workflow import analyze

    report = analyze(
        evaluation,
        ks=(1, 5, 10, 50),
        n_boot=10_000,
        seed=42,
    )
    ```

Python 0.8.x returns a normal dictionary because the report is small. Rust and Python 0.7.1 return typed `AnalysisReport` values.

For an artifact already on disk, Python can keep deserialization native:

```python
report = retrocast.analyze_file(
    "evaluation.json.gz",
    execution_stats_path="execution_stats.json.gz",
    workers=12,
)
```

## Metric Summary

Each metric maps to a summary containing `value`, `count`, optional confidence bounds, and an optional reliability warning.

=== "Python 0.8.x"

    ```python
    summary = report["metrics"]["solv_0[buyables]_rate"]

    print(summary["value"])
    print(summary["count"])
    print(summary.get("ci_low"), summary.get("ci_high"))
    print(summary.get("reliability"))
    ```

=== "Rust 0.8.x"

    ```rust
    let summary = &report.metrics["solv_0[buyables]_rate"];

    println!("{}", summary.value);
    println!("{}", summary.count);
    println!("{:?} {:?}", summary.ci_low, summary.ci_high);
    println!("{:?}", summary.reliability);
    ```

=== "Python 0.7.1"

    ```python
    summary = report.metrics["solv_0[buyables]_rate"]

    print(summary.value)
    print(summary.count)
    print(summary.ci_low, summary.ci_high)
    print(summary.reliability)
    ```

| Field | Meaning |
| --- | --- |
| `value` | Estimated metric value. |
| `count` | Number of targets used. |
| `ci_low`, `ci_high` | Bootstrap confidence bounds, when computed. |
| `reliability` | Warning such as low sample size or an extreme probability. |

## Solv-N Rate

`solv_0[buyables]_rate` is the fraction of targets with at least one candidate satisfying both Tier-0 validity and the task constraints labeled `buyables`.

The bracketed label comes from the task:

- `Solv-0[buyables]`: stock termination against one stock.
- `Solv-0[buyables+depth]`: stock termination plus a route-depth constraint.
- `Solv-0[buyables+leaf]`: stock termination plus required leaves.

## MRR@Solv-N

`solv_0[buyables]_mrr` is the mean reciprocal rank of the first candidate satisfying Solv-0 for each target.

A target contributes `1 / rank` when a satisfying candidate exists and `0` otherwise. MRR measures whether satisfying routes appear early; it is distinct from reconstructing a known benchmark route.

## Acceptable-Route Reconstruction

When benchmark targets contain `acceptable_routes`, analysis emits metrics such as:

```text
acceptable_reconstruction_top_10[buyables]
acceptable_prefix_depth_2_top_10[buyables]
```

These ask whether one of the first K task-satisfying candidates matches an acceptable route or a requested route prefix. Reconstruction metrics are omitted when the benchmark supplies no acceptable routes.

## Stratification

Analysis automatically creates route-depth strata when a target has an acceptable route or an explicit route-depth constraint.

=== "Python 0.8.x"

    ```python
    metric = "solv_0[buyables]_rate"
    for stratum, metrics in report["by_stratum"].items():
        print(stratum, metrics[metric]["value"])
    ```

=== "Rust 0.8.x"

    ```rust
    let metric = "solv_0[buyables]_rate";
    for (stratum, metrics) in &report.by_stratum {
        println!("{} {}", stratum, metrics[metric].value);
    }
    ```

=== "Python 0.7.1"

    ```python
    metric = "solv_0[buyables]_rate"
    for stratum, metrics in report.by_stratum.items():
        print(stratum, metrics[metric].value)
    ```

The stratification rule lives in the core so both interfaces emit identical groups.

## Bootstrap Reproducibility

`seed` fixes the resampling stream. `workers` may change execution order, but it must not change point estimates or seeded confidence intervals. `n_boot` must be greater than zero; larger values trade runtime for a more stable interval estimate.
