---
icon: lucide/clipboard-check
---

# Evaluation

Evaluation scores collected `Candidate`s against a `Task`. It records route validity, task-constraint satisfaction, acceptable-route matches, and optional per-target runtime without mutating the route tree.

## Score Collected Candidates

=== "Python 0.8.x"

    ```python
    stocks = {"buyables": list(stock_inchikeys)}

    evaluation = retrocast.score(
        predictions,
        task,
        stocks,
        match_level="full",
        acceptable_route_match="prefix",
        workers=12,
    )
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::score::score_owned;

    let evaluation = score_owned(
        predictions,
        task,
        &stocks,
        "full",
        "prefix",
        None,
        12,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.metrics import StockTerminationChecker
    from retrocast.workflow import score

    evaluation = score(
        predictions,
        task,
        constraint_checkers=[
            StockTerminationChecker(stocks={"buyables": stock_inchikeys}),
        ],
    )
    ```

Python 0.8.x receives `NativeEvaluation`. Passing `NativePredictions` to `score` consumes the handle and moves the candidate graph into the evaluation. Rust expresses the same ownership directly through `score_owned`. Python 0.7.1 returns an independent Pydantic `Evaluation` and retains its input models.

Write or inspect predictions before scoring if you need them afterward:

```python
predictions.write("candidates.json.gz")
evaluation = retrocast.score(predictions, task, stocks)
```

## Solv-N Separation

Solv-N is defined as:

```text
Solv-i[task] = Tier-i route validity + task constraint satisfaction
```

That separation appears directly in every scored candidate:

```text
candidate.validity      # Tier results
candidate.constraints   # Effective task-constraint result
```

=== "Python 0.8.x"

    ```python
    snapshot = evaluation.to_dict()
    scored = snapshot["targets"]["target-001"]["candidates"][0]

    tier_0_passes = scored["validity"]["tiers"]["0"]["status"] == "pass"
    task_passes = scored["constraints"]["status"] == "pass"
    solv_0 = tier_0_passes and task_passes
    ```

=== "Rust 0.8.x"

    ```rust
    let scored = &evaluation.targets["target-001"].candidates[0];

    let tier_0_passes = scored.satisfies_validity(0);
    let task_passes = scored.satisfies_task();
    let solv_0 = scored.satisfies_solv(0);
    ```

=== "Python 0.7.1"

    ```python
    scored = evaluation.targets["target-001"].candidates[0]

    tier_0_passes = scored.satisfies_validity(0)
    task_passes = scored.satisfies_task()
    solv_0 = scored.satisfies_solv(0)
    ```

Tier-0 validity comes from adaptation. A candidate with a route passes Tier 0; a candidate containing a `FailureRecord` fails it.

## Inspect Failed Adaptation Slots

Failed adaptation slots stay in the ranked list.

=== "Python 0.8.x"

    ```python
    for scored in snapshot["targets"]["target-001"]["candidates"]:
        if failure := scored.get("failure"):
            print(scored["rank"], failure["code"])
            continue

        print(scored["rank"], scored["route"]["target"]["smiles"])
    ```

=== "Rust 0.8.x"

    ```rust
    for scored in &evaluation.targets["target-001"].candidates {
        if let Some(failure) = &scored.failure {
            println!("{} {}", scored.rank, failure.code);
            continue;
        }

        if let Some(route) = &scored.route {
            println!("{} {}", scored.rank, route.target.smiles);
        }
    }
    ```

=== "Python 0.7.1"

    ```python
    for scored in evaluation.targets["target-001"].candidates:
        if scored.failed_adaptation():
            print(scored.rank, scored.failure.code)
            continue

        print(scored.rank, scored.route.target.smiles)
    ```

The `Candidate` invariant guarantees exactly one of `route` or `failure`.

## Track Runtime

Execution statistics are target-id maps. Record model inference outside RetroCast, then attach the measurements during scoring.

=== "Python 0.8.x"

    ```python
    execution_stats = {
        "wall_time": {"target-001": 1.42, "target-002": 0.87},
        "cpu_time": {"target-001": 1.31, "target-002": 0.79},
    }

    evaluation = retrocast.score(
        predictions,
        task,
        stocks,
        execution_stats=execution_stats,
    )
    report = retrocast.analyze(evaluation)

    print(report["runtime"]["total_wall_time"])
    print(report["runtime"]["mean_cpu_time"])
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::model::ExecutionStats;

    let execution_stats = ExecutionStats {
        wall_time,
        cpu_time,
    };
    let evaluation = score_owned(
        predictions,
        task,
        &stocks,
        "full",
        "prefix",
        Some(&execution_stats),
        12,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.utils import ExecutionTimer

    timer = ExecutionTimer()
    with timer.measure("target-001"):
        raw_by_target["target-001"] = model.predict(target.smiles)

    evaluation = score(
        predictions,
        task,
        constraint_checkers=checkers,
        execution_stats=timer.to_model(),
    )
    report = analyze(evaluation)
    print(report.runtime.total_wall_time)
    ```

Runtime is optional. Analysis summarizes only targets that contain measurements.

## Task Constraints

The task carries default constraints and optional per-target overrides. The native scorer currently implements these schema-2 constraint kinds:

- `retrocast.stock_termination`
- `retrocast.required_leaves`
- `retrocast.route_depth`

Stock sets are supplied separately because they can be large. The `stock` field on a task constraint selects a named set from the `stocks` map passed to `score`.

## Acceptable-Route Matching

`match_level` controls molecular identity during stock and acceptable-route comparison:

- `full`
- `no_stereo`
- `connectivity`

`acceptable_route_match="prefix"` accepts a target-rooted prefix of a benchmark route. `"exact"` requires the full route identity.

Reaction ids inside validity details use route-local paths such as `rc:r:/` and `rc:r:/1/0`. See [Route Node IDs](../../dev/reference/route-node-ids.md).

## Extension Boundary

Task constraints are data; their execution belongs to `retrocast-core`. The Python binding does not call a Python callback once per candidate because that would create different scoring semantics and parallel behavior for the two interfaces.

Add a new production constraint kind in the Rust scorer, expose it through the shared schema, and cover the same serialized result through Python and Rust contract tests.
