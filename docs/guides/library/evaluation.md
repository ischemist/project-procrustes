---
icon: lucide/clipboard-check
---

# Evaluation

Evaluation scores benchmark-keyed routes against a stock file. If you start from raw planner output, adapt and collect routes first.

## Tracking Runtime

```python title="Measure inference time"
from retrocast.utils import ExecutionTimer

timer = ExecutionTimer()

for target in benchmark.targets.values():
    with timer.measure(target.id):
        raw_output = model.predict(target.smiles)

    # ... adapt/store results ...

exec_stats = timer.to_model()
```

## Score Predictions

```python title="Evaluate routes against stock"
from retrocast.api import load_benchmark, load_stock_file, score_predictions

benchmark = load_benchmark("data/1-benchmarks/definitions/mkt-cnv-160.json.gz")
stock = load_stock_file("data/1-benchmarks/stocks/buyables-stock.txt")

# dict[target_id, list[Route]]
predictions = {"target-001": [route1, route2], "target-002": [route3]}

results = score_predictions(
    benchmark=benchmark,
    predictions=predictions,
    stock=stock,
    model_name="Experimental-Model-V1",
)

for target_id, evaluation in results.results.items():
    print(f"\nTarget: {target_id}")
    print(f"  Is solvable: {evaluation.is_solvable}")
    print(f"  Top-1 solved: {evaluation.top_1_is_solved}")
    print(f"  GT rank: {evaluation.gt_rank}")
    print(f"  Best route length: {evaluation.best_route_length}")
```

Predictions must be keyed by benchmark target ID. Each route is evaluated by checking whether all leaves are present in stock and whether the route matches the benchmark ground truth.

## Complete Evaluation Sketch

```python title="Adapt, collect, score"
from retrocast import adapt_provider_output, collect_benchmark_predictions, load_benchmark
from retrocast.adapters import AiZynthFinderAdapter
from retrocast.api import load_stock_file, score_predictions

benchmark = load_benchmark("benchmark.json.gz")
stock = load_stock_file("stock.txt")
adapter = AiZynthFinderAdapter()

predictions = adapt_provider_output(raw_provider_output, adapter)
collected = collect_benchmark_predictions(predictions, benchmark)

results = score_predictions(
    benchmark=benchmark,
    predictions=collected.routes_by_target,
    stock=stock,
    model_name="my-model",
)
```
