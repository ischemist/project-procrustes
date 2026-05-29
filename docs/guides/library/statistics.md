---
icon: lucide/chart-no-axes-combined
---

# Statistics

`analyze(...)` turns an `Evaluation` into an `AnalysisReport`.

```python
from retrocast.workflow import analyze

report = analyze(evaluation, ks=(1, 5, 10, 50), n_boot=10000, seed=42)
```

`AnalysisReport.metrics` stores overall metric summaries. `AnalysisReport.by_stratum` stores the same summaries split by route-depth strata when possible.

## Metric Summary

Each metric is a `MetricSummary`:

```python
summary = report.metrics["solv_0[buyables]_rate"]

print(summary.value)
print(summary.count)
print(summary.ci_low, summary.ci_high)
print(summary.reliability)
```

Fields:

| Field               | Meaning                                                         |
| ------------------- | --------------------------------------------------------------- |
| `value`             | estimated metric value                                          |
| `count`             | number of targets used                                          |
| `ci_low`, `ci_high` | bootstrap confidence interval bounds, when computed             |
| `reliability`       | optional warning such as low sample size or extreme probability |

## Solv-N Rate

`solv_0[buyables]_rate` is the fraction of targets with at least one candidate satisfying both Tier-0 validity and the task constraints labeled `buyables`.

The bracketed label comes from the task. For example:

- `Solv-0[buyables]`: stock termination against a stock.
- `Solv-0[buyables+depth]`: stock termination plus route-depth constraints.
- `Solv-0[buyables+leaf]`: stock termination plus required leaves.

## MRR@Solv-N

`mrr_solv_0[buyables]` is the mean reciprocal rank of the first candidate satisfying Solv-0 for each target.

A target contributes `1 / rank` if a satisfying candidate exists, otherwise `0`.

MRR is a ranking companion to Solv-N. It answers whether the model places satisfying routes early, not whether the route reconstructs a known benchmark route.

## Acceptable-Route Reconstruction

If benchmark targets contain `acceptable_routes`, analysis also emits metrics such as:

```text
acceptable_reconstruction_top_10[buyables]
```

This asks whether one of the first K task-satisfying candidates matches an acceptable route. Reconstruction metrics are omitted when no targets have acceptable routes.

## Stratification

By default, `analyze(...)` stratifies by route depth when the target has an acceptable route or an explicit route-depth constraint.

```python
for stratum, metrics in report.by_stratum.items():
    print(stratum, metrics["solv_0[buyables]_rate"].value)
```

You can pass a custom `stratify_by` function:

```python
def stock_label(target_result):
    return target_result.effective_constraints.stock


report = analyze(evaluation, stratify_by=stock_label)
```
