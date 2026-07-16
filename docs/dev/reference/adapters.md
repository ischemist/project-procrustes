---
icon: lucide/code-xml
---

# Writing a Custom Adapter

Every built-in adapter is a Rust implementation of `retrocast_core::adapters::Adapter`. Python selects it by the same stable string used by the standalone CLI; Python adapter classes are not part of the native package.

## Contract

An adapter owns two boundaries:

```rust
pub trait Adapter: Send + Sync {
    fn name(&self) -> &'static str;

    fn entries(
        &self,
        raw: serde_json::Value,
        source_key: Option<&str>,
    ) -> Result<Vec<RawRouteEntry>>;

    fn cast(
        &self,
        raw: serde_json::Value,
        mode: AdaptMode,
        target: Option<&Target>,
    ) -> Result<Route>;
}
```

`entries` splits one planner artifact into ranked route records while preserving source order and target hints. `cast` converts one record into the schema-v2 `Route -> Molecule -> Reaction -> Molecule` tree.

Adapters must:

- validate the planner-specific shape before transforming it
- canonicalize every SMILES and calculate identity through `retrocast_core::chem`
- preserve useful planner fields in annotations
- reject a route whose canonical root differs from the requested target
- return an `EngineError` with enough context to create a stable `FailureRecord`

## Modes and Ranks

`AdaptMode::Strict` rejects the route when any branch is invalid. `AdaptMode::Prune` may remove an invalid branch only when the raw representation makes the cut unambiguous and the remaining route is chemically well formed.

`max_candidates` always means the first N planner slots, not the first N successful routes. Failed slots remain ranked candidates so malformed predictions cannot inflate Tier-0 validity or MRR.

## Common Shapes

Shared builders in `adapters/common.rs` and `adapters/bipartite.rs` cover three recurring planner formats:

- alternating molecule and reaction trees
- product-to-reactants precursor maps
- plain nested molecule trees with implicit reactions

Keep format-specific traversal in the adapter. Keep schema construction, chemistry, cycle checks, and target validation in shared Rust helpers when their semantics are identical.

## Registration

Register the adapter in `adapters::built_in`:

```rust
pub fn built_in(name: &str) -> Option<Box<dyn Adapter>> {
    match name {
        "aizynthfinder" => Some(Box::new(AiZynthFinderAdapter)),
        "myadapter" => Some(Box::new(MyAdapter)),
        _ => None,
    }
}
```

Use a stable lowercase slug. Add it to the error message in `pipeline.rs` and to the CLI and Python binding documentation.

## Failure Codes

Messages are for humans. Workflows aggregate the stable `FailureRecord.code`:

| Code                           | Meaning                                                        |
| ------------------------------ | -------------------------------------------------------------- |
| `adapter.schema_invalid`       | Raw input does not match the adapter's format                  |
| `adapter.target_mismatch`      | The canonical route root differs from the requested target     |
| `adapter.cycle_detected`       | One route path revisits a molecule                             |
| `adapter.node_type_invalid`    | A bipartite slot contains the wrong node kind                  |
| `adapter.node_missing`         | A raw edge references an absent node                           |
| `adapter.route_string_invalid` | A route string has malformed steps or missing sides            |
| `adapter.unsupported_feature`  | A valid request asks for behavior the adapter cannot represent |
| `chem.invalid_smiles`          | RDKit C++ rejected a chemical string                           |

## Tests

Add format examples to the Rust integration suites under `retrocast-core/tests`. Cover successful casting, malformed artifact shapes, target mismatch, invalid chemistry, rank preservation, strict/prune differences, and deterministic output across worker counts.

Run focused tests with:

```bash
cargo test --manifest-path packages/retrocast-rs/Cargo.toml -p retrocast-core myadapter
```
