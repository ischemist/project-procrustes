---
icon: lucide/history
---

# Changelog

## Pre-1.0 Compatibility Policy

RetroCast is still pre-`1.0.0`. Until `1.0.0`, we will make breaking changes to the core schema or workflows whenever we see a compelling reason to do so. We'll try to keep some [sane deprecation schedule](dev/reference/deprecations.md) unless it incurs a significant complexity cost.

Practically speaking, you should treat `2-raw` (or wherever you store the raw planner outputs) as the source of truth and `3-processed`, `4-scored`, `5-results` as regenerable.

For most use cases, this should not be a problem since complete evaluation through ingest, score, and analyze is decently fast. If you are using (or planning to use) RetroCast for large evaluation runs, please feel free [to get in touch](ischemist.com/contact).

## v0.8.1 (unreleased)

v0.8.1 keeps corpus-sized artifacts inside Rust across ingest, score, and analyze. It also narrows ASKCOS pathway graphs before route casting and exposes `--workers` consistently across project commands.

The standalone `evaluate` command now processes adaptation, scoring, artifact writing, and metric preparation per target. Candidate and evaluation fragments are assembled in deterministic target order, while analysis retains compact scalar contributions instead of every scored route.

In single eight-worker measurements, standalone evaluation processed the 160-target, 25,762-candidate ASKCOS fixture in 13.65 seconds with 276 MiB peak RSS and the 400-target, 3,589-candidate AiZynthFinder fixture in 1.98 seconds with 61 MiB peak RSS. Candidate, evaluation, and analysis values matched the materialized implementation exactly.

## v0.8.0

v0.8.0 replaces the Python execution engine with one Rust core shared by the Python package and a standalone `retrocast` executable.

### Highlights

- Ported schema validation, every built-in planner adapter, route operations, ingest, scoring, analysis, statistics, artifact IO, provenance, datasets, curation, and training-release workflows to `retrocast-core`.
- Added a standalone executable with direct-file and project-mode `adapt`, `collect`, `ingest`, `score`, `analyze`, `evaluate`, `verify`, and dataset commands.
- Added PyO3 bindings behind the existing `import retrocast` interface. Untouched ingest and evaluation values remain Rust-owned between stages without intermediate JSON serialization.
- Replaced Python RDKit with a narrow RDKit C++ bridge for canonical SMILES, InChIKeys, and molecular descriptors.
- Added bounded native parallelism through `workers`, with the Python binding releasing the GIL while the core executes.
- Added reproducible cross-platform wheel and standalone-bundle builds for Linux x86-64, Windows x86-64, macOS arm64, and macOS x86-64.

### Distribution and migration

`pip install retrocast` now installs a native wheel containing the PyO3 extension and repaired RDKit libraries. It no longer installs the Python `rdkit` package. Source installations require Rust, a C++20 compiler, Boost headers, and RDKit C++.

Standalone archives contain the `retrocast` executable and its native libraries and do not require Python or Conda at runtime. This native packaging change is the reason for the minor-version bump.

## v0.7.0

v0.7.0 updates RetroCast to new [schema design](/dev/rationale/schema-design), check that page for the full mental model.

### Highlights

- Promoted the schema-2 models, adapters, workflow, metrics, IO, and CLI into the main `retrocast` package.
- Replaced the old prediction-wrapper path with `Candidate`, which preserves either a successful `Route` or a `FailureRecord` for benchmark accounting.
- Split the workflow around the schema-2 path: `adapt -> collect -> score -> analyze`.
- Kept route-local node ids derived from tree position rather than serialized into `Molecule` or `Reaction` objects.
- Added route signatures for full-route, prefix-depth, reaction, and subtree comparison.
- Implemented [Solv-N scoring](/dev/rationale/solv-n-evaluation): Tier-N route validity plus task-constraint satisfaction.

### Migration Notes

This is a hard pre-1.0 schema break. Processed, scored, and result artifacts produced by older schema-1 workflows should be regenerated from raw planner payloads.

## v0.6.0

v0.6.0 makes RetroCast more useful as a general route-standardization library, not only as a benchmark runner. The main design change is the split between adapting provider output into canonical predictions and collecting those predictions onto benchmark targets for scoring.

For the full machine-generated change list, see the GitHub comparison after the release tag is published: [v0.5.3...v0.6.0](https://github.com/ischemist/project-procrustes/compare/v0.5.3...v0.6.0).

### Highlights

- Added `PredictedRoute`, an envelope around canonical `Route` chemistry for provider-level rank, score, confidence, source row, and metadata.
- Split adaptation from benchmark collection with `adapt_provider_output(...)`, `adapt_target_keyed_provider_output(...)`, and `collect_benchmark_predictions(...)`.
- Kept `retrocast ingest` as the project-mode convenience command for the common raw-output-to-benchmark-routes workflow.
- Added route-corpus IO and CLI support for streamable prediction artifacts such as `.jsonl.gz`.
- Added support for flat LLM completion corpora with `<synthesis_step>` XML blocks.
- Added new adapter coverage, including MolBuilder, and renamed public adapter classes/slugs toward canonical planner names.
- Added stable error codes for adapter, IO, CLI, and workflow boundaries, with ingest failure counts persisted into manifests.
- Added PaRoutes training-set release utilities and hosted dataset loaders.
- Improved release packaging so PyPI releases build from the release tag, support intentional dev releases, and use Hatchling/Hatch VCS for dynamic versions.

### Adapter Workflow Split

Earlier versions centered the public workflow on `ingest`, which did two jobs at once:

1. standardize raw planner output into canonical `Route` objects
2. collect those routes onto benchmark targets and write `routes.json.gz`

That shape worked for benchmark runs, but it made RetroCast feel less like a general route-standardization library. In v0.6.0, standardization and benchmark collection are separate library workflows. `ingest` still exists as the one-command wrapper that runs both.

| Before v0.6.0 | v0.6.0 replacement | Why |
| --- | --- | --- |
| `adapt_single_route(raw, target, adapter_name)` | `adapt_route(raw_route, adapter)` | The one-route API no longer requires target context when the raw route carries its own target. |
| `adapt_routes(raw, target, adapter_name)` | `adapt_provider_output(raw_provider_output, adapter)` | Standardization can now handle one provider output without benchmark target context. |
| `retrocast ingest` as the main way to adapt benchmark predictions | `retrocast adapt` then `retrocast collect` for ad-hoc use | Exposes standardization and benchmark collection as separate steps. |
| `retrocast ingest` for project-mode benchmark runs | still `retrocast ingest` | Project mode keeps the one-command convenience wrapper. |
| `Route.rank` | `PredictedRoute.rank` during provider-output adaptation, list order in scoring inputs | Keeps canonical `Route` free of benchmark/list-position metadata while preserving provider rank. |

Provider-output adaptation APIs now return `PredictedRoute`, an envelope around canonical `Route` chemistry. It carries provider-level metadata such as rank, score, confidence, and source row provenance. `adapt_route(...)` remains the single-payload chemistry API and returns `Route | None`; use `adapt_prediction(...)` when you explicitly want a one-off prediction envelope.

Scoring artifacts remain benchmark-keyed `dict[target_id, list[Route]]` so existing evaluation semantics stay stable. Prediction manifest `content_hash` values are now order-sensitive within each target because route list order is the ranking signal. Re-exported manifests can therefore differ from older manifests even when the route structures are otherwise identical.

### Deprecations

The v0.6.0 compatibility layer intentionally warns instead of removing the old names immediately. These surfaces are scheduled for removal in v0.9.0:

- legacy adapter slugs such as `aizynth`, `dms`, and `dreamretro`
- legacy adapter class aliases such as `AizynthAdapter`, `DMSAdapter`, and `DreamRetroAdapter`
- target-local adaptation helpers `adapt_single_route(...)` and `adapt_routes(...)`

See the [deprecation schedule](dev/reference/deprecations.md) for the canonical removal plan.

### Migration

For one raw route-like payload:

```python
from retrocast import adapt_route
from retrocast.adapters import DirectMultiStepAdapter

adapter = DirectMultiStepAdapter()
route = adapt_route(raw_route, adapter)
```

For one raw prediction payload where you want an envelope:

```python
from retrocast import adapt_prediction
from retrocast.adapters import DirectMultiStepAdapter

adapter = DirectMultiStepAdapter()
prediction = adapt_prediction(raw_route, adapter, rank=1)
```

For one raw provider output containing one or many routes:

```python
from retrocast import adapt_provider_output
from retrocast.adapters import AiZynthFinderAdapter

adapter = AiZynthFinderAdapter()
predictions = adapt_provider_output(raw_provider_output, adapter)
```

For raw output already keyed by target ID or target SMILES:

```python
from retrocast import adapt_target_keyed_provider_output, load_benchmark
from retrocast.adapters import AiZynthFinderAdapter

benchmark = load_benchmark("benchmark.json.gz")
adapter = AiZynthFinderAdapter()
predictions = adapt_target_keyed_provider_output(raw_mapping, benchmark, adapter)
```

To produce benchmark-keyed routes for scoring:

```python
from retrocast import collect_benchmark_predictions

collected = collect_benchmark_predictions(predictions, benchmark)
routes_by_target = collected.routes_by_target
```

Ad-hoc CLI users can now run the two steps explicitly:

```bash
retrocast adapt \
  --input raw_predictions.json.gz \
  --adapter aizynthfinder \
  --input-kind provider-output \
  --output route-corpus.jsonl.gz

retrocast collect \
  --input route-corpus.jsonl.gz \
  --benchmark benchmark.json.gz \
  --output routes.json.gz
```

Project-mode users can keep using `retrocast ingest`.
