---
icon: lucide/clipboard-check
---

# Evaluation

Evaluation scores benchmark-keyed routes or candidate records against a stock
file. Route chemistry stays unchanged; scoring writes evaluation annotations
into `ScoredCandidate` records.

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
    print(f"  First Tier-0 rank: {evaluation.first_valid_rank(tier=0)}")
    print(f"  First Solv-0[STR] rank: {evaluation.first_solv_rank(tier=0, scope='stock')}")
    print(f"  Benchmark reconstruction rank: {evaluation.reconstruction_rank(scope='stock')}")

    for candidate in evaluation.candidates:
        tier_0 = candidate.satisfies_validity(tier=0)
        solv_0 = candidate.satisfies_solv(tier=0, scope="stock")
        print(f"  rank {candidate.rank}: tier-0={tier_0}, solv-0[str]={solv_0}")
```

Predictions must be keyed by benchmark target ID. Each route is evaluated for
Tier-0 validity, stock termination, and benchmark-reference matching.

`Solv-0[STR]` means Tier-0 validity plus the default stock-termination scope.
It is not the same as benchmark route reconstruction.

## Inspect Scored Candidates

The Python API keeps typed tier dictionaries, while saved artifacts use readable
labels such as `"tier 0"`.

```python title="Route-level validity and Solv-N"
target_eval = results.results["target-001"]
candidate = target_eval.candidates[0]

print(candidate.rank)
print(candidate.satisfies_validity(tier=0))
print(candidate.satisfies_constraints(scope="stock"))
print(candidate.satisfies_solv(tier=0, scope="stock"))

tier_0 = candidate.validity.tiers[0]
print(tier_0.status)
```

Failed adaptation slots are still candidates. They have no route, but they keep
their raw rank and carry a typed failure record:

```python title="Handle failed candidates"
for candidate in target_eval.candidates:
    if candidate.route is None:
        print(candidate.rank, candidate.adapter_failure.code)
        continue

    print(candidate.rank, candidate.route.length)
```

## Reaction-Level Failures

Tier validity is route-level only when every reaction passes that tier. Reaction
annotations show where a route failed.

```python title="Find failing reactions"
for candidate in target_eval.candidates:
    if candidate.route is None:
        continue

    reactions = list(candidate.route.iter_reactions())
    for annotation in candidate.validity.reactions:
        if annotation.satisfies_validity(tier=0):
            continue

        route_reaction = reactions[annotation.reaction_index - 1]
        failure_codes = [check.code for check in annotation.tiers[0].checks]
        print(
            annotation.reaction_index,
            route_reaction.product.smiles,
            failure_codes,
        )
```

## Stored Shape

`evaluation.json.gz` stores chemistry once, on the route. Validity annotations
point back to reactions by index.

```json title="Candidate validity artifact"
{
  "validity": {
    "tier 0": {
      "status": "pass"
    },
    "reactions": [
      {
        "reaction_index": 1,
        "validity": {
          "tier 0": {
            "status": "fail",
            "checks": [{"code": "tier0.empty_reactants"}]
          }
        }
      }
    ]
  }
}
```

Passing tiers omit `checks`. Failing checks may include `message` and `details`
when the evaluator has useful context.

## Target-Level Ranks

Target-level ranks answer three different questions:

```python title="Rank diagnostics"
print(target_eval.first_valid_rank(tier=0))
print(target_eval.first_solv_rank(tier=0, scope="stock"))
print(target_eval.reconstruction_rank(scope="stock"))
```

- `first_valid_rank(tier=0)` is the raw rank of the first Tier-0-valid route.
- `first_solv_rank(tier=0, scope="stock")` is the raw rank of the first route
  that is Tier-0-valid and stock-terminated.
- `reconstruction_rank(scope="stock")` is the effective benchmark-reference
  rank after filtering to the stock scope.

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

## Candidate-Preserving Scoring

When you need raw-rank diagnostics over failed adaptation slots, preserve
candidates before scoring:

```python title="Preserve failed candidates"
from retrocast.workflow.adapt import adapt_target_keyed_candidate_records
from retrocast.workflow.score import score_candidate_records

adapted = adapt_target_keyed_candidate_records(raw_by_target, benchmark, adapter)

results = score_candidate_records(
    benchmark=benchmark,
    candidates=adapted.records,
    stock=stock,
    stock_name="buyables-stock",
    model_name="my-model",
)
```

Failed candidates have `route=None`, `adapter_failure` set, and a failed Tier-0
result after scoring. This is what keeps `MRR Tier-0` and `MRR Solv-0[STR]`
honest for sequence-style planners.
