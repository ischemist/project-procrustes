---
icon: lucide/chart-no-axes-combined
---

# Statistics

RetroCast uses bootstrap resampling to calculate confidence intervals for scored model results.

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

print("\nStratified by length:")
for length, metric in stats.solv_0.by_group.items():
    print(f"  Length {length}: {metric.value:.1%} [{metric.ci_lower:.1%}, {metric.ci_upper:.1%}]")
```

Typical metrics include Tier-0 validity, Solv-0[STR], raw-rank MRR diagnostics,
Top-K benchmark route reconstruction, and stratified performance by route length.

??? example "Example output"

    ```
    Tier-0 validity: 91.0% [86.8%, 95.0%]
    Solv-0[STR]: 45.3% [42.1%, 48.6%]
    MRR Solv-0[STR]: 0.612 [0.570, 0.653]
    Top-1: 23.5% [20.8%, 26.3%]
    Top-5: 38.2% [35.1%, 41.4%]
    Top-10: 42.7% [39.5%, 45.9%]

    Stratified by length:
      Length 2: 65.2% [58.3%, 72.1%]
      Length 3: 52.8% [47.2%, 58.4%]
      Length 4: 38.1% [32.5%, 43.8%]
      Length 5: 24.3% [19.1%, 29.6%]
    ```
