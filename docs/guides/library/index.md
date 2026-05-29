---
icon: lucide/cable
---

# Python Library

RetroCast can be used as a normal Python library. The public schema-2 workflow is the same one used by the CLI:

```text
adapt -> collect -> score -> analyze
```

Use the library API when you want to run RetroCast inside notebooks, custom model loops, training-set curation scripts, or your own file layout.

## Minimal Example

```python
from retrocast import get_adapter
from retrocast.io import load_benchmark
from retrocast.metrics.constraints import TaskConstraintChecker
from retrocast.workflow import analyze, ingest_candidates, score

task = load_benchmark(benchmark_path)
adapter = get_adapter("paroutes")

predictions = ingest_candidates(raw_payload, adapter, task)
evaluation = score(
    predictions,
    task,
    constraint_checker=TaskConstraintChecker(stocks={"buyables": stock_inchikeys}),
)
report = analyze(evaluation)
```

## Main Objects

| Object | Role |
| --- | --- |
| `Route` | Canonical chemistry tree: `Molecule -> Reaction -> Molecule`. |
| `Candidate` | One ranked prediction slot: either a valid `Route` or a `FailureRecord`. |
| `Task` / `Benchmark` | Targets plus task constraints. |
| `ScoredCandidate` | A ranked candidate with validity, constraint, and acceptable-route annotations. |
| `Evaluation` | Scored candidates grouped by target for one task. |
| `AnalysisReport` | Solv-N, MRR@Solv-N, acceptable-route reconstruction, and confidence intervals. |

## Choose A Guide

| Task | Start here |
| --- | --- |
| Choose between route-only and candidate-preserving adaptation | [Adaptation](adaptation.md) |
| Score collected candidates and inspect validity results | [Evaluation](evaluation.md) |
| Compute and interpret metric summaries | [Statistics](statistics.md) |
| Make plots from reports or legacy statistics objects | [Visualization](visualization.md) |
| Look up public functions quickly | [Reference](reference.md) |

## Mental Model

`Route` is chemistry only. It does not know its planner rank, benchmark target, stock result, or validity status.

`Candidate` is the evaluation accounting object. It keeps planner rank and contains either a route or a typed adaptation failure. Use candidates whenever failed prediction slots should count toward Solv-0 and MRR.

`Evaluation` stores scoring results without changing the route tree. Tier validity and task-constraint satisfaction are separate so Solv-N stays explicit:

```text
Solv-i[task] = Tier-i route validity + task constraint satisfaction
```

For the deeper data-model rationale, see [Schema Design](/rationale/schema-design).
