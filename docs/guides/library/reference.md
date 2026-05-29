---
icon: lucide/book-open-text
---

# Library Reference

This page lists the main schema-2 entry points. See the neighboring guides for context and examples.

## Adaptation

| Function | Purpose | Returns |
| --- | --- | --- | --- |
| `adapt_route(raw_route, adapter)` | Adapt one raw route record | `Route | None` |
| `adapt_routes(raw_payload, adapter)` | Adapt a raw artifact and keep only successful routes | `list[Route]` |
| `adapt_candidates(raw_payload, adapter)` | Adapt a raw artifact while preserving failed prediction slots | `list[Candidate]` |
| `collect_candidates(candidates, task)` | Map candidates onto task targets | `dict[str, list[Candidate]]` |
| `collect_routes(routes, task)` | Map route-only outputs onto task targets | `dict[str, list[Route]]` |
| `ingest_candidates(raw_payload, adapter, task)` | `adapt_candidates + collect_candidates` | `dict[str, list[Candidate]]` |
| `ingest_routes(raw_payload, adapter, task)` | `adapt_routes + collect_routes` | `dict[str, list[Route]]` |

## Scoring and Analysis

| Function | Purpose | Returns |
| --- | --- | --- |
| `score(predictions, task, constraint_checker=...)` | Score collected candidates | `Evaluation` |
| `score_candidate(candidate, target=..., constraints=..., ...)` | Score one candidate | `ScoredCandidate` |
| `score_target(candidates, target=..., constraints=..., ...)` | Score one target's candidates | `TargetResult` |
| `analyze(evaluation, ks=(1, 5, 10, 50), n_boot=10000)` | Summarize an evaluation | `AnalysisReport` |
| `score_predictions(predictions, task, stock=..., stock_name=...)` | Convenience wrapper around `score(...)` | `Evaluation` |
| `analyze_evaluation(evaluation, n_boot=10000)` | Convenience wrapper around `analyze(...)` | `AnalysisReport` |

## IO

| Function | Purpose | Shape |
| --- | --- | --- |
| `load_benchmark(path)` / `save_benchmark(...)` | Benchmark artifacts | `Benchmark` |
| `load_candidates(path)` / `save_candidates(...)` | Flat candidate artifacts | `list[Candidate]` |
| `load_collected_candidates(path)` / `save_collected_candidates(...)` | Collected candidate artifacts | `dict[str, list[Candidate]]` |
| `load_evaluation(path)` / `save_evaluation(...)` | Scored artifacts | `Evaluation` |
| `load_analysis_report(path)` / `save_analysis_report(...)` | Analysis artifacts | `AnalysisReport` |
| `load_stock_file(path)` | Stock files | `set[InChIKeyStr]` or `set[SmilesStr]` |

## Core Models

| Model                                                            | Purpose                    |
| ---------------------------------------------------------------- | -------------------------- |
| `Route`, `Molecule`, `Reaction`                                  | Canonical route tree       |
| `RoutePath`, `ReactionId`, `MoleculeId`                          | Route-local addressing     |
| `Candidate`, `FailureRecord`                                     | Adaptation accounting      |
| `Target`, `TaskConstraints`, `Task`, `Benchmark`                 | Problem definition         |
| `CheckResult`, `TierResult`, `RouteValidity`, `ConstraintResult` | Scoring details            |
| `ScoredCandidate`, `TargetResult`, `Evaluation`                  | Scored prediction artifact |
| `MetricSummary`, `AnalysisReport`                                | Analysis artifact          |

## Adapters

```python
from retrocast.adapters import ADAPTER_TYPES, get_adapter

print(sorted(ADAPTER_TYPES))
adapter = get_adapter("paroutes")
```

Supported adapters include `aizynthfinder`, `askcos`, `directmultistep`, `dreamretroer`, `molbuilder`, `multistepttl`, `paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`, `syntheseus`, and `ursa`.
