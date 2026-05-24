---
icon: lucide/chart-no-axes-combined
---

# Statistics

RetroCast uses bootstrap resampling to calculate confidence intervals for scored model results.

Statistics are grouped into three concepts:

- **Solv-N hierarchy:** route validity and scoped solvability.
- **Rank within Solv-N:** mean reciprocal raw rank of the first valid or solvable route.
- **Benchmark route reconstruction:** conservative Top-K proxy against benchmark acceptable routes.

## Compute Statistics

```python title="Generate statistical summary"
from retrocast.api import compute_model_statistics

stats = compute_model_statistics(results, n_boot=10000, seed=42)

tier_0 = stats.tier_0_validity.overall
print(f"Tier-0 validity: {tier_0.value:.1%} "
      f"[{tier_0.ci_lower:.1%}, {tier_0.ci_upper:.1%}]")

solv_0 = stats.solv_0.overall
print(f"Solv-0[STR]: {solv_0.value:.1%} "
      f"[{solv_0.ci_lower:.1%}, {solv_0.ci_upper:.1%}]")

mrr = stats.mrr_solv_0.overall
print(f"MRR Solv-0[STR]: {mrr.value:.3f} "
      f"[{mrr.ci_lower:.3f}, {mrr.ci_upper:.3f}]")

for k in [1, 5, 10]:
    topk = stats.top_k_accuracy[k].overall
    print(f"Top-{k}: {topk.value:.1%} [{topk.ci_lower:.1%}, {topk.ci_upper:.1%}]")

print("\nStratified by depth:")
for depth, metric in stats.solv_0.by_group.items():
    print(f"  {depth}: {metric.value:.1%} [{metric.ci_lower:.1%}, {metric.ci_upper:.1%}]")
```

Typical metrics include Tier-0 validity, Solv-0\[STR\], raw-rank MRR diagnostics,
Top-K benchmark route reconstruction, and stratified performance by route depth.

## Metric Semantics

`stats.tier_0_validity` is the fraction of targets with at least one Tier-0-valid
candidate. It does not require stock termination.

`stats.solv_0` is the fraction of targets with at least one `Solv-0[STR]`
candidate. In the stock scope, this means Tier-0 validity plus stock
termination.

`stats.mrr_tier_0` and `stats.mrr_solv_0` are rank diagnostics. A target
contributes `1 / rank` for the first matching candidate and `0` if there is no
matching candidate. Example: `MRR Solv-0[STR] = 0.900` means the first
Solv-0\[STR\] candidate is usually near rank 1, but this is not a reconstruction
accuracy.

`stats.top_k_accuracy[k]` is benchmark route reconstruction after stock-scope
filtering. It asks whether one of the first `k` stock-terminated candidates
matches an acceptable benchmark route. This is a conservative proxy, not a
Solv-N metric.

## Reliability Flags

Each metric has a reliability flag:

```python title="Inspect reliability"
metric = stats.solv_0.overall
print(metric.n_samples)
print(metric.reliability.code)
print(metric.reliability.message)
```

- `OK`: no obvious sample-size or boundary warning.
- `LOW_N`: fewer than 30 targets.
- `EXTREME_P`: the estimate is near 0 or 1, so bootstrap intervals may collapse.

## Runtime

Runtime fields are stored in seconds:

```python title="Runtime summaries"
print(stats.total_wall_time)
print(stats.mean_wall_time)
print(stats.total_cpu_time)
print(stats.mean_cpu_time)
```

??? example "Example output"

    ```
    Tier-0 validity: 91.0% [86.8%, 95.0%]
    Solv-0[STR]: 45.3% [42.1%, 48.6%]
    MRR Solv-0[STR]: 0.612 [0.570, 0.653]
    Top-1: 23.5% [20.8%, 26.3%]
    Top-5: 38.2% [35.1%, 41.4%]
    Top-10: 42.7% [39.5%, 45.9%]

    Stratified by depth:
      depth 2: 65.2% [58.3%, 72.1%]
      depth 3: 52.8% [47.2%, 58.4%]
      depth 4: 38.1% [32.5%, 43.8%]
      depth 5: 24.3% [19.1%, 29.6%]
    ```
