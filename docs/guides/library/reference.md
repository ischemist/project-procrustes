---
icon: lucide/book-open-text
---

# Library Reference

## Workflow

| Function | Purpose | Returns |
| --- | --- | --- |
| `adapt(raw, adapter, *, mode="strict", target=None, source_key=None, max_candidates=None, workers=1)` | Adapt one planner payload | `list[dict]` |
| `ingest(raw, adapter, task, *, mode="strict", max_candidates=None, workers=1)` | Adapt and collect in-memory input | `NativePredictions` |
| `ingest_file(raw_path, adapter, task_path, *, mode="strict", max_candidates=None, workers=1)` | Read, adapt, and collect in Rust | `NativePredictions` |
| `score(predictions, task, stocks, *, match_level="full", acceptable_route_match="prefix", execution_stats=None, workers=1)` | Consume predictions and score them | `NativeEvaluation` |
| `analyze(evaluation, *, ks=..., prefix_depths=..., n_boot=10000, seed=42, workers=1)` | Calculate metrics and intervals | `dict` |
| `analyze_file(evaluation_path, *, ..., execution_stats_path=None)` | Read and analyze an evaluation in Rust | `dict` |
| `pipeline(raw_path, benchmark_path, stock_path, output_dir, *, ...)` | Run ingest, score, and analysis in one native process | `dict` of timing and throughput statistics |

Adapter names are stable lowercase strings: `aizynthfinder`, `askcos`, `directmultistep`, `dreamretroer`, `molbuilder`, `multistepttl`, `paroutes`, `retrochimera`, `retrostar`, `synllama`, `synplanner`, `syntheseus`, and `ursa`.

## Native Handles

`NativePredictions` exposes:

- `write(path)` to write JSON or JSON gzip from Rust
- `to_dict()` to create a Python snapshot
- `json()` to create a JSON string

`score` consumes the prediction payload. Later access through the old handle raises `RuntimeError`.

`NativeEvaluation` exposes the same three materialization methods plus `metric_label()`. Analysis borrows the evaluation because its result is small.

## Chemistry

| Function | Purpose |
| --- | --- |
| `canonicalize_smiles(smiles, remove_mapping=False, ignore_stereo=False)` | Canonicalize with RDKit C++ |
| `get_inchi_key(smiles, level="full")` | Calculate an InChIKey |
| `reduce_inchi_key(inchikey, level)` | Reduce to `no_stereo` or `connectivity` |
| `molecular_descriptors(smiles)` | Return heavy atoms, molecular weight, and chiral centers |

Invalid chemical input raises `ValueError`. Schema, adapter, and native workflow failures raise `RuntimeError`; missing artifact paths raise `OSError`.

## Runtime Identity

`retrocast.__version__` reports the package version and `retrocast.__engine__` is always `"rust"`. `retrocast.engine_info()` returns the RetroCast version, `"RDKit C++"`, and the linked RDKit version.
