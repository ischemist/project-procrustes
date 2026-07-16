---
icon: lucide/code-xml
---

# Writing a Custom Adapter

Every retrosynthesis planner has its own output format: nested molecule/reaction trees, route strings, or graph payloads. An adapter traverses one raw format and casts each prediction into the canonical `Route` model described in [Schema Design](/dev/rationale/schema-design).

```text
raw artifact -> RawRouteEntry -> Route -> Candidate
```

Built-in adapters execute in `retrocast-core` for both front ends. Python selects them by name; new production adapters implement the Rust trait and are registered in the core.

## Use A Built-In Adapter

=== "Python 0.8.x"

    ```python
    candidates = retrocast.adapt(raw_payload, "paroutes", workers=12)
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::adapters::{adapt_candidates_with_workers, built_in};

    let adapter = built_in("paroutes").expect("built-in adapter");
    let candidates = adapt_candidates_with_workers(
        raw_payload,
        adapter.as_ref(),
        AdaptMode::Strict,
        None,
        None,
        None,
        12,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast import adapt_candidates, get_adapter

    adapter = get_adapter("paroutes")
    candidates = adapt_candidates(raw_payload, adapter)
    ```

The adapter identifier is part of the public artifact contract. Keep it lowercase and stable.

## Contract

An adapter implements two operations:

```rust
use serde_json::Value;

use retrocast_core::{
    adapters::RawRouteEntry,
    error::Result,
    model::{Route, Target},
    route::AdaptMode,
};

pub trait Adapter: Send + Sync {
    fn name(&self) -> &'static str;

    fn entries(
        &self,
        payload: Value,
        source_key: Option<&str>,
    ) -> Result<Vec<RawRouteEntry>>;

    fn cast(
        &self,
        raw_route: Value,
        mode: AdaptMode,
        target: Option<&Target>,
    ) -> Result<Route>;
}
```

`entries(...)` traverses an artifact. It splits a multi-target file or a list of completions into ranked route entries while preserving source order and target hints.

`cast(...)` constructs one route. It returns a canonical `Route` or a typed `EngineError` when the input cannot be represented safely.

## Raw Entries

`RawRouteEntry` carries one raw route record and the provenance needed to preserve evaluation accounting:

```rust
pub struct RawRouteEntry {
    pub payload: Value,
    pub source_key: Option<String>,
    pub source_row_index: Option<usize>,
    pub source_record_id: Option<String>,
    pub target_hint_id: Option<String>,
    pub target_hint_smiles: Option<String>,
    pub source_order: Option<usize>,
}
```

`payload` is passed to `cast(...)`.

`source_order` is the planner's rank slot when the format provides one. The adaptation workflow preserves it as `Candidate.rank`; otherwise rank falls back to entry order.

`target_hint_id` and `target_hint_smiles` let a failed prediction remain attached to the correct benchmark target even when no route can be built.

## Route Construction

Every adapter returns the same chemistry shape:

```rust
Route {
    target: Molecule {
        smiles: canonical_target_smiles,
        inchikey: target_inchikey,
        product_of: Some(Box::new(Reaction {
            reactants,
            ..
        })),
        ..
    },
    ..
}
```

The implementation must:

- validate the raw shape before transforming it
- canonicalize every raw SMILES through `retrocast_core::chem`
- calculate molecule identity through the RDKit C++ bridge
- build an alternating `Molecule -> Reaction -> Molecule` tree
- preserve useful planner data in `annotations` on the relevant object
- verify the route root against `target` when one is supplied

Adapters return `Route`, not `Candidate`. The workflow converts a cast error into a ranked `FailureRecord` with stable code and context.

## Adapt Modes

`AdaptMode::Strict` rejects invalid chemistry, cycles, and route structures that cannot be represented as a tree.

`AdaptMode::Prune` asks how much of the target-rooted prediction remains valid before a corrupted branch. An adapter may remove a branch only when the cut is unambiguous. It must still fail if pruning removes the target or leaves a reaction with no reactants.

=== "Python 0.8.x"

    ```python
    candidates = retrocast.adapt(raw_payload, "myadapter", mode="prune")
    ```

=== "Rust 0.8.x"

    ```rust
    let mode = AdaptMode::Prune;
    ```

=== "Python 0.7.1"

    ```python
    candidates = adapt_candidates(raw_payload, adapter, mode="prune")
    ```

## Workflow Use

The candidate-preserving workflow is the benchmark path:

=== "Python 0.8.x"

    ```python
    candidates = retrocast.adapt(
        raw_payload,
        "myadapter",
        max_candidates=100,
    )
    predictions = retrocast.ingest(
        raw_payload,
        "myadapter",
        task,
        max_candidates=100,
    )
    ```

=== "Rust 0.8.x"

    ```rust
    let candidates = adapt_candidates_with_workers(
        raw_payload,
        &adapter,
        AdaptMode::Strict,
        None,
        None,
        Some(100),
        workers,
    )?;

    let predictions = ingest(
        raw_payload,
        &adapter,
        &task,
        AdaptMode::Strict,
        Some(100),
        workers,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast import adapt_candidates, ingest_candidates

    candidates = adapt_candidates(
        raw_payload,
        adapter,
        max_candidates=100,
    )
    predictions = ingest_candidates(
        raw_payload,
        adapter,
        task,
        max_candidates=100,
    )
    ```

`max_candidates` means the first N raw prediction slots, not the first N successful routes. Failed slots remain in rank order so adaptation errors cannot inflate Tier-0 or MRR metrics.

## Common Raw Shapes

Most adapters reduce to one of three patterns.

### Bipartite Trees

Some planners already produce an alternating graph:

```text
molecule -> reaction -> molecule -> reaction -> ...
```

Validate node roles and child links before recursively building the route. A molecule child under a molecule, or a missing referenced node, is a schema failure.

### Precursor Maps

Route strings and reaction tables can often be reduced to a product-to-reactants map:

```json
{
  "target": ["intermediate", "stock_a"],
  "intermediate": ["stock_b", "stock_c"]
}
```

Choose the root product, reject cycles, and walk the map into the alternating route tree.

### Plain Molecule Trees

Some raw routes contain nested molecule nodes without explicit reaction nodes. Each parent-child expansion becomes one `Reaction`; adapter-specific callbacks or helpers extract SMILES and annotations.

## Target Validation

When `target` is supplied, the canonical route root must match it. A route for another molecule is not a failed synthesis of the requested target; it is the wrong object.

```rust
if route.target.smiles != target.smiles {
    return Err(EngineError::TargetMismatch {
        adapter: "myadapter",
        target_id: target.id.clone(),
        expected: target.smiles.to_string(),
        actual: route.target.smiles.to_string(),
    });
}
```

Some formats cannot be interpreted without a target. Missing target context should then produce a typed adapter error.

## Adapter Errors

Messages are for humans. `code` and `context` are the stable fields aggregated by workflows.

Keep `context["adapter"]` lowercase and aligned with the registry slug. Use display capitalization only in the message.

| Code | Meaning |
| --- | --- |
| `adapter.schema_invalid` | Raw payload does not match the expected shape. |
| `adapter.target_mismatch` | Adapted root is not the requested target. |
| `adapter.cycle_detected` | The graph revisits a molecule in one route path. |
| `adapter.node_type_invalid` | A molecule appears where a reaction is required, or the reverse. |
| `adapter.node_missing` | An edge references a missing node. |
| `adapter.route_string_empty` | A route string is empty after parsing. |
| `adapter.route_string_invalid` | Step boundaries or reactant/product fields are malformed. |
| `adapter.route_transform_failed` | A route-local transform failed without a narrower code. |
| `adapter.unsupported_feature` | A valid request asks for an unimplemented adapter feature. |

Chemistry failures use `chem.*` codes, usually `chem.invalid_smiles` or `chem.runtime_error`. See [Error Handling](errors.md) for candidate-level accounting.

### Examples

`adapter.cycle_detected`: the raw route loops back to a molecule already in the same path.

```json
{
  "smiles": "CCO",
  "children": [
    {
      "smiles": "CC=O",
      "children": [{"smiles": "CCO", "children": []}]
    }
  ]
}
```

`adapter.node_type_invalid`: a bipartite molecule has a molecule child where a reaction is required.

```json
{
  "type": "mol",
  "smiles": "CCO",
  "children": [{"type": "mol", "smiles": "CC=O", "children": []}]
}
```

`adapter.route_string_invalid`: a reaction step omits the product side.

```text
CCO.CN>>
```

## Registration

Add the adapter module under `retrocast-core/src/adapters`, export its type from `adapters/mod.rs`, add its canonical slug to `BUILT_IN_ADAPTERS`, and resolve it in `built_in(...)`.

Python and CLI discovery use that same registry; no separate front-end registration is required.

## Test Checklist

Cover:

- one valid representative payload
- empty and malformed top-level payloads
- invalid SMILES and RDKit failures
- cycles and invalid node alternation
- strict versus prune behavior
- target mismatch and missing target context
- source order, explicit ranks, and target hints
- failed-slot preservation through ingest and score
- deterministic results with one and multiple workers
- serialized parity through the Python binding

Keep fixtures small. A single route that exercises the raw format is more diagnostic than a large planner dump.
