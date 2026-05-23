---
icon: lucide/lightbulb
---

# Concepts and Architecture

RetroCast provides a standardized framework for evaluating retrosynthesis models. It addresses the fragmentation of output formats in the field by _decoupling_ the model's internal representation from the evaluation logic.

!!! abstract "Core idea"

    **Adapters** translate diverse model outputs into a canonical schema, enabling apples-to-apples comparison across any retrosynthesis algorithm. We decouple the model's internal representation from the evaluation logic.

## The Core Philosophy: Adapters as an Air Gap

Retrosynthesis models produce diverse output formats. Some output bipartite graphs (AiZynthFinder), others output precursor maps (Retro\*), and some output recursive dictionaries (DirectMultiStep). Comparing these directly requires writing bespoke evaluation code for every model, which leads to bugs and inconsistent metrics.

RetroCast introduces an **adapter layer** between the model and the evaluation pipeline:

```mermaid
graph LR
    A[Model Output<br/>Native Format] --> B[Adapter<br/>Translation Layer]
    B --> C[Canonical Schema<br/>Route Objects]
    C --> D[Evaluation Pipeline<br/>Metrics & Analysis]

```

**The flow:**

1. **The Model** runs independently and saves output in its native format
2. **The Adapter** reads this format and transforms it into the canonical RetroCast schema
3. **The Pipeline** performs scoring and analysis on canonical objects, unaware of the original format

!!! success "Why this matters"

    This architecture ensures that metrics like stock-termination rate and route length are calculated identically for every model.

## The Canonical Data Model

RetroCast defines a strict, recursive object model in `retrocast.models.chem`. Structurally, this model is a **directed acyclic bipartite graph** consisting of alternating molecule nodes and reaction nodes.

### A Resolved AND/OR Tree

Many retrosynthesis frameworks (e.g., Syntheseus, AiZynthFinder) utilize an _AND/OR graph_ to represent the entire search space. In these graphs, a Molecule node (OR) may have multiple child Reaction nodes (AND), representing competing choices.

The RetroCast `Route` object represents a _resolved instance_ of this graph: a single, specific pathway, with a Molecule node having at most one child Reaction node.

### Schema Definition

The schema enforces the minimal information required for rigorous evaluation while allowing for extensibility via metadata dictionaries.

=== "Route"

    ```python title="retrocast.models.chem.Route"
    class Route(BaseModel):
        """The root container for a single prediction."""
        target: Molecule

        # Provenance
        metadata: dict[str, Any]
        retrocast_version: str

        # Computed Properties
        @property
        def length(self) -> int: ...  # (1)!
        @property
        def leaves(self) -> set[Molecule]: ...  # (2)!
        @property
        def signature(self) -> str: ...  # (3)!
    ```

    1. Longest path from target to any leaf
    2. All starting materials in the route
    3. Cryptographic hash for deduplication

    Route ordering is intentionally external to the `Route` object. In memory,
    list order is the canonical ranking signal; scoring writes explicit ranks
    onto `ScoredRoute` results.

=== "Molecule"

    ```python title="retrocast.models.chem.Molecule"
    class Molecule(BaseModel):
        """Represents a chemical node (OR node)."""
        # Core Identity
        smiles: SmilesStr
        inchikey: InchiKeyStr  # (1)!

        # Tree Structure: 0 or 1 reaction step
        synthesis_step: ReactionStep | None  # (2)!

        # Extensibility
        metadata: dict[str, Any]

        # Computed Properties
        @property
        def is_leaf(self) -> bool: ...
    ```

    1. Primary ID for hashing/equality
    2. `None` for leaf molecules (starting materials)

=== "ReactionStep"

    ```python title="retrocast.models.chem.ReactionStep"
    class ReactionStep(BaseModel):
        """Represents a reaction node (AND node)."""
        # Tree Structure: N reactant children
        reactants: list[Molecule]  # (1)!

        # Chemical Details (Optional)
        mapped_smiles: str | None
        template: str | None
        reagents: list[str] | None
        solvents: list[str] | None

        # Extensibility
        metadata: dict[str, Any]

        # Computed Properties
        @property
        def is_convergent(self) -> bool: ...  # (2)!
    ```

    1. Must have ≥1 reactant molecules
    2. Returns `True` if ≥2 reactants are non-leaves

### Rationale: Why Bipartite?

We deem the bipartite structure, explicitly separating `Molecule` and `ReactionStep`, natural for representing multistep routes and it allows precise attribution of data:

- **Molecule properties** (e.g., "is purchasable", molecular weight) belong to the `Molecule` node
- **Reaction properties** (e.g., template scores, probability, patent IDs) belong to the `ReactionStep` node

### Interchange Format

!!! info "You don't need to change your model"

    Model developers are **not required** to use this schema internally. RetroCast treats it as an interchange format: the adapter casts your native output into this structure. Extra data (attention weights, search trees, etc.) can be preserved in `metadata` dictionaries if you need them for downstream analysis.

### Route vs. PredictedRoute

`Route` owns the canonical chemistry tree. It is the right object when you want to inspect route length, leaves, reactions, structural signatures, or chemistry-level metadata.

`PredictedRoute` owns provider-level prediction context around a `Route`: rank, score, confidence, source row index, source record ID, and source key. Library workflows return `PredictedRoute` when they are adapting a provider output rather than one isolated route. For the practical decision table, see [Library Adaptation](guides/library/adaptation.md#choose-a-workflow).

### Route vs. Candidate vs. Evaluation

Solv-N evaluation separates chemistry from evaluation annotations.

- `Route`: canonical chemistry only. It has no validity, stock, score, or benchmark-reference labels.
- `CandidateRecord`: one raw ranked model slot after adaptation accounting. It contains either a canonical `Route` or a typed adapter failure.
- `ScoredCandidate`: one candidate plus evaluation annotations: tier validity, scope constraint results, acceptable-route matching, and adapter failure when present.
- `TargetEvaluation`: target-level summaries derived from scored candidates.

`ScoredRoute` is a legacy compatibility view for older scored artifacts and notebooks. It is not the canonical Solv-N model. New tier and Solv-N annotations should be read from `ScoredCandidate.validity`, `ScoredCandidate.constraint_results`, and target-level rank dictionaries.

## Data Organization and Lifecycle

RetroCast enforces a structured directory layout to manage the transformation of data from raw predictions to final statistics. This structure ensures ==reproducibility and traceability==.

```mermaid
graph TD
    A[1-benchmarks<br/>Definitions & Stocks] --> B[2-raw<br/>Model Output]
    B --> C[3-processed<br/>Route Objects]
    C --> D[4-scored<br/>Evaluated Routes]
    D --> E[5-results<br/>Statistics & Reports]

    B -.->|retrocast ingest| C
    C -.->|retrocast score| D
    D -.->|retrocast analyze| E

```

All paths below are relative to your **data directory**, which defaults to `data/retrocast/` but can be customized via:

- CLI flag: `retrocast --data-dir /custom/path`
- Environment variable: `RETROCAST_DATA_DIR=/custom/path`
- Config file: `data_dir: /custom/path`

Run `retrocast config` to see your resolved paths.

### 1. Benchmarks (`1-benchmarks/`)

**Immutable** evaluation task definitions.

- `definitions/`: Gzipped JSON files defining targets (IDs and SMILES)
- `stocks/`: Text files with available building blocks (one SMILES per line)

### 2. Raw Data (`2-raw/`)

**Read-only** artifacts generated by models.

- Structure: `2-raw/<model>/<benchmark>/<filename>`
- Shape: model-native provider output. RetroCast does not impose one schema here.
- Manifest: `2-raw/<model>/<benchmark>/manifest.json` declares the adapter and raw result filename.

Raw data can be a target-keyed mapping, a list of route-like payloads, JSONL, CSV-derived records, or another adapter-supported provider format. The adapter is the boundary that makes this usable.

### 3. Processed Data (`3-processed/`)

Generated by: `retrocast ingest`

- **Primary artifact:** `3-processed/<benchmark>/<model>/routes.json.gz`
- **Shape:** `dict[target_id, list[Route]]`
- **Operations:** Adapt raw payloads into prediction envelopes around canonical routes, collect routes onto the benchmark, deduplicate per target, optional sampling (keep first _n_ routes)

Candidate-audit mode may also write:

- **Candidate artifact:** `3-processed/<benchmark>/<model>/candidates.json.gz`
- **Shape:** `CandidateRecordsArtifact`
- **Payload:** `metadata: CandidateAuditMetadata`, `records: dict[target_id, list[CandidateRecord]]`

`routes.json.gz` is the route-only compatibility artifact used by ordinary scoring. `candidates.json.gz` is the rank-preserving artifact required for candidate-denominator metrics such as raw-rank MRR and candidate-level tier pass rates.

### 4. Scored Data (`4-scored/`)

Generated by: `retrocast score`

- **Primary artifact:** `4-scored/<benchmark>/<model>/<stock>/evaluation.json.gz`
- **Shape:** `EvaluationResults`
- **Payload:** metric scopes, per-target `ScoredCandidate` records, target-level rank summaries, and provenance metadata
- **Metrics:** Tier validity, scope constraints, Solv-i summaries, and benchmark-reference matching
- **Independence:** Same routes can be scored against multiple stocks without re-processing

For ordinary route-only scoring, `EvaluationResults.metadata["candidate_audit"]` records `candidate_denominator="route_only"` and `preserves_failed_candidates=false`. This prevents downstream code from treating saved routes as a complete raw candidate stream.

`TargetEvaluation.routes` exists only as a deprecated compatibility property over successful scored candidates or old loaded artifacts. New `evaluation.json.gz` files do not store `routes`. New code should use `TargetEvaluation.candidates`.

### 5. Results (`5-results/`)

Generated by: `retrocast analyze`

- **Primary artifact:** `5-results/<benchmark>/<model>/<stock>/statistics.json.gz`
- **Shape:** `ModelStatistics`
- **Statistics:** Bootstrap confidence intervals for stock termination, Tier-0 validity, Solv-0, and Top-K accuracy
- **Artifacts:** JSON statistics, Markdown reports, HTML visualizations

`ModelStatistics.solvability` is a deprecated alias for stock-termination rate. New code should use `ModelStatistics.stock_termination`.

## Artifact Shape Reference

| Stage | File | Stored shape | Notes |
| --- | --- | --- | --- |
| `1-benchmarks` | `<benchmark>.json.gz` | `BenchmarkSet` | Target definitions and acceptable routes. |
| `1-benchmarks` | stock files | text or curated stock artifact | Available starting materials used by scope constraints. |
| `2-raw` | provider result file | adapter-native | Read-only model output. Adapter-specific. |
| `2-raw` | `manifest.json` | manifest with adapter directives | Declares how to read the raw file. |
| `3-processed` | `routes.json.gz` | `dict[str, list[Route]]` | Canonical successful routes, keyed by benchmark target. |
| `3-processed` | `candidates.json.gz` | `CandidateRecordsArtifact` | Optional rank-preserving candidate audit artifact. |
| `4-scored` | `evaluation.json.gz` | `EvaluationResults` | Canonical scored candidates plus target summaries. |
| `5-results` | `statistics.json.gz` | `ModelStatistics` | Bootstrap summaries and report-ready metrics. |

## Provenance and Verification

Reproducibility is a primary design goal. RetroCast tracks the lineage of every data artifact using **Manifests**.

!!! info "Cryptographic audit trail"

    Every generated file (e.g., `routes.json.gz`) has a companion manifest (`routes.manifest.json`) containing SHA256 hashes of inputs and outputs.

A manifest records:

1. **Action:** The command or script that generated the file
2. **Inputs:** Paths and SHA256 hashes of all source files
3. **Parameters:** Configuration arguments (stock name, random seed, etc.)
4. **Outputs:** Paths and hashes of generated files

### Verification

The `retrocast verify` command audits the data pipeline with a two-phase check:

!!! example "Verification phases"

    **Phase 1: Logical Consistency**
    Ensures the input hash in a child manifest matches the output hash of the parent manifest

    **Phase 2: Physical Integrity**
    Ensures the file on disk matches the hash recorded in its manifest

This system detects:

- :warning: Data corruption
- :warning: Manual tampering
- :warning: Out-of-order execution steps
