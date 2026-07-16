---
icon: lucide/book-open-text
---

# Library Reference

This page lists the current native workflow surface and the frozen pure-Python oracle used for differential testing.

## Workflow

=== "Python 0.8.x"

    | Function | Purpose | Returns |
    | --- | --- | --- |
    | `adapt(raw, adapter, *, mode="strict", target=None, source_key=None, max_candidates=None, workers=1)` | Adapt one planner payload | `list[dict]` |
    | `ingest(raw, adapter, task, *, mode="strict", max_candidates=None, workers=1)` | Adapt and collect in-memory input | `NativePredictions` |
    | `ingest_file(raw_path, adapter, task_path, *, mode="strict", max_candidates=None, workers=1)` | Read, adapt, and collect in Rust | `NativePredictions` |
    | `score(predictions, task, stocks, *, match_level="full", acceptable_route_match="prefix", execution_stats=None, workers=1)` | Consume predictions and score them | `NativeEvaluation` |
    | `analyze(evaluation, *, ks=..., prefix_depths=..., n_boot=10000, seed=42, workers=1)` | Calculate metrics and intervals | `dict` |
    | `analyze_file(evaluation_path, *, ..., execution_stats_path=None)` | Read and analyze an evaluation in Rust | `dict` |
    | `pipeline(raw_path, benchmark_path, stock_path, output_dir, *, ...)` | Run ingest, score, and analysis in one native process | timing and throughput `dict` |

=== "Rust 0.8.x"

    | Function | Purpose | Returns |
    | --- | --- | --- |
    | `adapters::adapt_candidates_with_workers(...)` | Adapt one planner payload | `Result<Vec<Candidate>>` |
    | `adapt::ingest(...)` | Adapt and collect an in-memory `Value` | `Result<Predictions>` |
    | `adapt::ingest_file(...)` | Stream, adapt, and collect an artifact | `Result<Predictions>` |
    | `score::score_owned(...)` | Consume predictions and score them | `Result<Evaluation>` |
    | `analyze::analyze(...)` | Calculate metrics and intervals | `Result<AnalysisReport>` |
    | `pipeline::run_pipeline(...)` | Run the native file pipeline | `Result<PipelineStats>` |

=== "Python 0.7.1"

    | Function | Purpose | Returns |
    | --- | --- | --- |
    | `adapt_route(raw_route, adapter)` | Adapt one raw route | `Route | None` |
    | `adapt_routes(raw_payload, adapter)` | Keep successful routes | `list[Route]` |
    | `adapt_candidates(raw_payload, adapter)` | Preserve ranked failures | `list[Candidate]` |
    | `collect_candidates(candidates, task)` | Map candidates onto targets | `dict[str, list[Candidate]]` |
    | `ingest_candidates(raw_payload, adapter, task)` | Adapt and collect | `dict[str, list[Candidate]]` |
    | `score(predictions, task, constraint_checkers=...)` | Score Pydantic candidates | `Evaluation` |
    | `analyze(evaluation, ks=..., n_boot=10000)` | Calculate metrics | `AnalysisReport` |

Adapter names are stable lowercase strings: `aizynthfinder`, `askcos`, `directmultistep`, `dreamretroer`, `molbuilder`, `multistepttl`, `paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`, `syntheseus`, and `ursa`.

## Native Handles

`NativePredictions` exposes:

- `write(path)` to write JSON or JSON gzip from Rust
- `to_dict()` to create a Python snapshot
- `json()` to create a JSON string

`score` consumes the prediction payload. Later access through the old handle raises `RuntimeError`.

`NativeEvaluation` exposes the same materialization methods plus `metric_label()`. Analysis borrows the evaluation because its result is small.

Rust callers own `Predictions`, `Evaluation`, and `AnalysisReport` directly and use Serde for explicit serialization.

## Core Models

| Model | Purpose |
| --- | --- |
| `Route`, `Molecule`, `Reaction` | Canonical route tree |
| `Candidate`, `FailureRecord` | Adaptation accounting |
| `Target`, `Constraint`, `Task` | Problem definition |
| `ScoredCandidate`, `TargetResult`, `Evaluation` | Scored output |
| `MetricSummary`, `RuntimeSummary`, `AnalysisReport` | Analysis output |
| `ExecutionStats` | Optional per-target runtime input |

The Python boundary represents individual models as JSON-compatible dictionaries. Corpus-sized prediction and evaluation collections remain native handles.

## Chemistry

=== "Python 0.8.x"

    | Function | Purpose |
    | --- | --- |
    | `canonicalize_smiles(smiles, remove_mapping=False, ignore_stereo=False)` | Canonicalize with RDKit C++ |
    | `get_inchi_key(smiles, level="full")` | Calculate an InChIKey |
    | `reduce_inchi_key(inchikey, level)` | Reduce to `no_stereo` or `connectivity` |
    | `molecular_descriptors(smiles)` | Return heavy atoms, molecular weight, and chiral centers |

=== "Rust 0.8.x"

    | Function | Purpose |
    | --- | --- |
    | `chem::canonicalize(...)` | Canonicalize with the RDKit C++ bridge |
    | `chem::inchi_key(...)` | Calculate an InChIKey |
    | `route::reduce_inchikey(...)` | Reduce an InChIKey to a match level |
    | `chem::descriptors(...)` | Calculate molecular descriptors |

=== "Python 0.7.1"

    | Function | Purpose |
    | --- | --- |
    | `retrocast.chem.canonicalize_smiles(...)` | Canonicalize with Python RDKit |
    | `retrocast.chem.get_inchi_key(...)` | Calculate an InChIKey |
    | `retrocast.chem.reduce_inchi_key(...)` | Reduce an InChIKey to a match level |

No RDKit object crosses the Rust or Python API. Invalid chemical input raises `ValueError` in Python and returns `EngineError` in Rust.

## Artifact I/O

=== "Python 0.8.x"

    ```python
    predictions.write("candidates.json.gz")
    evaluation.write("evaluation.json.gz")
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::io::{read_json, write_json};

    let evaluation: Evaluation = read_json(&path)?;
    write_json(&output_path, &evaluation)?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.io import save_collected_candidates, save_evaluation

    save_collected_candidates(predictions, "candidates.json.gz")
    save_evaluation(evaluation, "evaluation.json.gz")
    ```

All three interfaces infer gzip from the path and write the same schema-v2 artifacts.

## Runtime Identity

=== "Python 0.8.x"

    ```python
    print(retrocast.__version__)
    print(retrocast.__engine__)  # "rust"
    print(retrocast.engine_info())
    ```

=== "Rust 0.8.x"

    ```rust
    println!("{}", retrocast_core::VERSION);
    println!("{}", retrocast_core::chem::version());
    ```

=== "Python 0.7.1"

    ```python
    print(retrocast.__version__)
    ```

`engine_info()` reports the RetroCast version, `RDKit C++`, and the linked RDKit version.

## Errors

- Python uses `ValueError` for invalid chemistry, `OSError` for artifact-path failures, and `RuntimeError` for schema, adapter, or workflow failures.
- Rust returns `retrocast_core::error::EngineError` from core operations.

See [Error Handling](../../dev/reference/errors.md) for stable failure codes and candidate-level accounting.
