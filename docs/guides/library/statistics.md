---
icon: lucide/chart-no-axes-combined
---

# Statistics

RetroCast uses bootstrap resampling to calculate confidence intervals for scored model results.

## Compute Statistics

```python title="Generate statistical summary"
from retrocast.api import compute_model_statistics

stats = compute_model_statistics(results, n_boot=10000, seed=42)

solvability = stats.solvability.overall
print(f"Solvability: {solvability.value:.1%} "
      f"[{solvability.ci_lower:.1%}, {solvability.ci_upper:.1%}]")

for k in [1, 5, 10]:
    topk = stats.top_k[k].overall
    print(f"Top-{k}: {topk.value:.1%} [{topk.ci_lower:.1%}, {topk.ci_upper:.1%}]")

print("\nStratified by length:")
for length, metric in stats.solvability.by_group.items():
    print(f"  Length {length}: {metric.value:.1%} [{metric.ci_lower:.1%}, {metric.ci_upper:.1%}]")
```

Typical metrics include overall solvability, Top-K accuracy, ground-truth match rate, and stratified performance by route length.

??? example "Example output"

    ```
    Solvability: 45.3% [42.1%, 48.6%]
    Top-1: 23.5% [20.8%, 26.3%]
    Top-5: 38.2% [35.1%, 41.4%]
    Top-10: 42.7% [39.5%, 45.9%]

    Stratified by length:
      Length 2: 65.2% [58.3%, 72.1%]
      Length 3: 52.8% [47.2%, 58.4%]
      Length 4: 38.1% [32.5%, 43.8%]
      Length 5: 24.3% [19.1%, 29.6%]
    ```
