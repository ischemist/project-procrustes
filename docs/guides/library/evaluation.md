---
icon: lucide/clipboard-check
---

# Evaluation

Evaluation scores collected `Candidate`s against a `Task`.

## Tracking Runtime

Use `ExecutionTimer` around model inference,

```python
from retrocast.utils import ExecutionTimer

timer = ExecutionTimer()
raw_by_target = {}

for target_id, target in task.targets.items():
    with timer.measure(target_id):
        raw_by_target[target_id] = model.predict(target.smiles)

execution_stats = timer.to_model()
```

Then pass `execution_stats` when scoring:

```python
from retrocast.metrics import RequiredLeavesChecker, RouteDepthChecker, StockTerminationChecker

evaluation = score(
    predictions,
    task,
    constraint_checkers=[
        StockTerminationChecker(stocks={"buyables": stock_inchikeys})
    ],
    execution_stats=execution_stats,
)

report = analyze(evaluation)
print(report.runtime.total_wall_time)
print(report.runtime.mean_cpu_time)
```

The CLI report prints runtime when the loaded `Evaluation` contains per-target `wall_time` or `cpu_time` values.

## Score Collected Candidates

```python
from retrocast.metrics import RequiredLeavesChecker, RouteDepthChecker, StockTerminationChecker
from retrocast.workflow import score

evaluation = score(
    predictions,
    task,
    constraint_checkers=[
        StockTerminationChecker(stocks={"buyables": stock_inchikeys}),
        RequiredLeavesChecker(),
        RouteDepthChecker(),
    ],
)
```

`predictions` is `dict[target_id, list[Candidate]]`, usually produced by `collect_candidates(...)` or `ingest_candidates(...)`.

The default scoring path always includes Tier-0 validity. A candidate with a route passes Tier-0 adaptation validity. A candidate with a `FailureRecord` fails Tier-0.

## Solv-N Separation

Solv-N is defined as:

```text
Solv-i[task] = Tier-i route validity + task constraint satisfaction
```

That separation appears directly in the model:

```python
candidate.validity      # RouteValidity
candidate.constraints   # ConstraintResult
```

Use the convenience methods on `ScoredCandidate`:

```python
scored = evaluation.targets["target-001"].candidates[0]

print(scored.satisfies_validity(Tier.ZERO))
print(scored.satisfies_task())
print(scored.satisfies_solv(tier=0))
```

## Inspect Failed Adaptation Slots

Failed adaptation slots stay in the ranked list.

```python
for scored in evaluation.targets["target-001"].candidates:
    if scored.failed_adaptation():
        print(scored.rank, scored.failure.code)
        continue

    print(scored.rank, scored.route.target.smiles)
```

Failed candidates have `route=None`, carry the original `FailureRecord`, and fail Tier-0 validity.

## Reaction-Level Validity

External Tier checkers can report route-level and reaction-level validity.

```python
for scored in evaluation.targets["target-001"].candidates:
    if not scored.has_route():
        continue

    for reaction in scored.validity.reactions:
        tier_1 = reaction.tiers.get(Tier.ONE)
        if tier_1 is None:
            continue
        print(reaction.reaction_id, tier_1.status)
```

Reaction ids are route-local path ids such as `rc:r:/` and `rc:r:/1/0`. See [Route Node IDs](../../dev/reference/route-node-ids.md).

## Custom Tier Checkers

Tier-0 is reserved for adaptation validity. Additional Tier checkers implement the `TierChecker` protocol.

```python
class MyTierOneChecker:
    tier = Tier.ONE
    name = "my-tier-one"

    def check_route(self, route: Route) -> RouteValidity:
        ...


evaluation = score(
    predictions,
    task,
    tier_checkers=[MyTierOneChecker()],
    constraint_checkers=[
        StockTerminationChecker(stocks={"buyables": stock_inchikeys})
    ],
)
```

```python
from retrocast.api import score_predictions

evaluation = score_predictions(
    predictions,
    task,
    stocks={"buyables": stock_inchikeys},
)
```
