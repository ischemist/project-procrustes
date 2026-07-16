---
icon: lucide/cable
---

# Library API

RetroCast 0.8.x has one production implementation and two public interfaces. `retrocast-core` owns the schemas, chemistry, adapters, scoring, and analysis; the Python package binds directly to it. The frozen pure-Python 0.7.1 package remains in the repository as a differential-testing oracle.

The public schema-2 workflow is the same from either language:

```text
adapt -> collect -> score -> analyze
```

Use the library API inside notebooks, model loops, training-set curation programs, or applications with their own file layout. Use the standalone CLI when RetroCast should manage files and project directories for you.

## Minimal Example

=== "Python 0.8.x"

    ```python
    import json

    import retrocast

    task = json.loads(benchmark_path.read_text())
    stocks = {"buyables": list(stock_inchikeys)}

    predictions = retrocast.ingest_file(
        raw_path,
        "paroutes",
        benchmark_path,
        workers=12,
    )
    evaluation = retrocast.score(predictions, task, stocks, workers=12)
    report = retrocast.analyze(evaluation, workers=12)
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::{
        adapt::ingest_file,
        adapters::built_in,
        analyze::analyze,
        io::read_json,
        model::Task,
        route::AdaptMode,
        score::{score_owned, Stocks},
    };

    let task: Task = read_json(&benchmark_path)?;
    let adapter = built_in("paroutes").expect("built-in adapter");
    let predictions = ingest_file(
        &raw_path,
        adapter.as_ref(),
        &task,
        AdaptMode::Strict,
        None,
        12,
    )?;
    let evaluation = score_owned(
        predictions,
        task,
        &stocks,
        "full",
        "prefix",
        None,
        12,
    )?;
    let report = analyze(
        &evaluation,
        &[1, 3, 5, 10, 20, 50, 100],
        &[1, 2, 3],
        10_000,
        42,
        12,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast import get_adapter
    from retrocast.io import load_benchmark
    from retrocast.metrics import StockTerminationChecker
    from retrocast.workflow import analyze, ingest_candidates, score

    task = load_benchmark(benchmark_path)
    adapter = get_adapter("paroutes")
    predictions = ingest_candidates(raw_payload, adapter, task)
    evaluation = score(
        predictions,
        task,
        constraint_checkers=[
            StockTerminationChecker(stocks={"buyables": stock_inchikeys}),
        ],
    )
    report = analyze(evaluation)
    ```

The linked tabs remember the selected implementation across the documentation site. Python 0.7.1 examples document the frozen oracle; they are not available from the published 0.8.x wheel.

## Data Ownership

Rust code receives ordinary typed values. Python receives `NativePredictions` and `NativeEvaluation` handles so corpus-sized route graphs stay in Rust between stages.

`score` consumes `NativePredictions` and moves its routes into `NativeEvaluation`. Write or inspect predictions before scoring if you need a snapshot:

=== "Python 0.8.x"

    ```python
    predictions.write("candidates.json.gz")
    preview = predictions.to_dict()

    evaluation = retrocast.score(predictions, task, stocks)
    ```

=== "Rust 0.8.x"

    ```rust
    retrocast_core::io::write_json(&candidates_path, &predictions)?;

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
    from retrocast.io import save_collected_candidates

    save_collected_candidates(predictions, "candidates.json.gz")
    evaluation = score(predictions, task, constraint_checkers=checkers)
    ```

Calling `.to_dict()` or `.json()` materializes a Python snapshot. File-based entry points and `.write()` keep artifact I/O in Rust.

## Main Objects

| Object | Role |
| --- | --- |
| `Route` | Canonical chemistry tree: `Molecule -> Reaction -> Molecule`. |
| `Candidate` | One ranked prediction slot containing either a valid `Route` or a `FailureRecord`. |
| `Task` | Targets and task constraints. |
| `ScoredCandidate` | A ranked candidate with validity, constraint, and acceptable-route annotations. |
| `Evaluation` | Scored candidates grouped by target for one task. |
| `AnalysisReport` | Solv-N, MRR@Solv-N, acceptable-route reconstruction, and confidence intervals. |

Python 0.8.x exposes corpus-sized collections through native handles and returns the small `AnalysisReport` as a normal dictionary. Rust uses these types directly. Python 0.7.1 uses Pydantic models throughout.

## Choose A Guide

| Task | Start here |
| --- | --- |
| Adapt planner output while preserving failed prediction slots | [Adaptation](adaptation.md) |
| Score collected candidates and inspect validity results | [Evaluation](evaluation.md) |
| Compute and interpret metric summaries | [Statistics](statistics.md) |
| Turn reports into plots and comparisons | [Visualization](visualization.md) |
| Look up public functions quickly | [Reference](reference.md) |

## Mental Model

`Route` is chemistry only. It does not know its planner rank, benchmark target, stock result, or validity status.

`Candidate` is the evaluation accounting object. It keeps planner rank and contains either a route or a typed adaptation failure. Failed prediction slots therefore count toward Solv-0 and MRR instead of disappearing during adaptation.

`Evaluation` stores scoring results without changing the route tree. Tier validity and task-constraint satisfaction remain separate:

```text
Solv-i[task] = Tier-i route validity + task constraint satisfaction
```

For the data-model contract, see [Schema Design](/dev/rationale/schema-design).
