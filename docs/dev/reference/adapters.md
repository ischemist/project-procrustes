---
icon: lucide/code-xml
---

# Writing a Custom Adapter

Every retrosynthesis planner has its own output format: nested molecule/reaction trees, route strings, custom xml payloads. The job of a single adapter is to handle one type of raw format and cast it into the canonical `Route` model (see [schema design](/dev/rationale/schema-design) for a mental model of `Route`).

The adapter-level data flow is:

```text
raw artifact -> RawRouteEntry -> Route
```

## Contract

An adapter implements two methods:

```python
from collections.abc import Iterator
from typing import Any, Protocol

from retrocast.adapters.base import AdaptMode, RawRouteEntry
from retrocast.models.route import Route
from retrocast.models.task import Target


class Adapter(Protocol):
    def iter_raw_routes(
        self,
        raw_payload: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]: ...

    def cast(
        self,
        raw_route: Any,
        *,
        mode: AdaptMode = "strict",
        target: Target | None = None,
    ) -> Route: ...
```

`iter_raw_routes(...)` is for artifact traversal. If a planner stores all target predictions in one file, this is where the file is split. If a planner stores one completion per JSONL row, this is where rows become entries. It should preserve source order and any target hints present in each raw route record.

`cast(...)` is for route construction. It receives one raw route record and returns one `Route`, or raises a typed adapter/chemistry error if the input cannot be represented as a valid route.

## Raw Entries

`RawRouteEntry` is the envelope carried between traversal and casting. It contains one raw route record plus the small amount of provenance needed by workflows.

```python
@dataclass(frozen=True, slots=True)
class RawRouteEntry:
    payload: Any
    source_key: str | None = None
    source_row_index: int | None = None
    source_record_id: str | None = None
    target_hint_id: str | None = None
    target_hint_smiles: str | None = None
    source_order: int | None = None
```

`payload` is the raw route record that will later be passed to `cast(...)`. The name is broad because adapters receive many planner-specific shapes; in prose, this document calls it a raw route record.

`source_order` is the planner's rank slot when the raw format exposes one. `adapt_candidates(...)` preserves it as `Candidate.rank`; if it is absent, rank falls back to iteration order.

`target_hint_id` and `target_hint_smiles` are used when the raw route record itself identifies the target. They matter most for failed prediction slots: if no `Route` can be built, collection can still attach the failure to the right target.

## Route Construction

`cast(...)` should return the same shape regardless of the planner that produced the route:

```python
Route(
    target=Molecule(
        smiles=canonical_target_smiles,
        inchikey=target_inchikey,
        product_of=Reaction(reactants=[...]),
    )
)
```

The basic rules are:

- validate raw input with pydantic models or explicit checks before transforming it.
- canonicalize every raw SMILES with `retrocast.chem.canonicalize_smiles`.
- compute molecule identity with `get_inchi_key` after canonicalization.
- build the route as `Route -> Molecule -> Reaction -> Molecule`.
- preserve useful planner data in `annotations` on the relevant `Route`, `Molecule`, or `Reaction`.
- when `target` is provided, verify that the route root matches `target.smiles` after canonicalization.

Adapters return `Route`, not `Candidate`. A failed `cast(...)` should raise `AdapterSchemaError`, `AdapterLogicError`, or a `ChemError`; the caller decides whether that becomes `None`, a skipped route, or a `FailureRecord`.

## Adapt Modes

By default, adaptation is strict. If any SMILES is invalid, if the route contains a cycle, or if the route cannot be represented as a tree, the raw route record fails adaptation.

```python
AdaptMode = Literal["strict", "prune"]
```

`prune` exists for a narrower question: "how much of this prediction was valid before the corrupted branch?" In prune mode, an adapter may drop an invalid branch and return the longest valid prefix route. This is only acceptable when the raw representation makes the cut unambiguous. If pruning removes the target or leaves a reaction with no reactants, the adapter should fail.

## Workflow Use

The workflow layer exposes three adaptation paths:

```python
adapt_route(raw_route_payload, adapter, *, mode="strict", target=None) -> Route | None
adapt_routes(raw_payload, adapter, *, mode="strict", max_routes=None) -> list[Route]
adapt_candidates(raw_payload, adapter, *, mode="strict", max_candidates=None) -> list[Candidate]
```

`adapt_route(...)` is the one-off convenience function. It tries to cast one raw route record and returns `None` for expected adapter or chemistry failures.

`adapt_routes(...)` returns only valid routes. `max_routes` counts successful routes, so malformed raw route records do not consume the limit.

`adapt_candidates(...)` is the benchmarking path. `max_candidates` means "the first N raw prediction slots", not "the first N successful routes". Failed slots become `FailureRecord`s so Tier-0 validity and MRR are not inflated by silently dropping bad predictions.

## Common Raw Shapes

Most adapters end up looking like one of the following patterns. The helpers in `retrocast.adapters.common` cover the repetitive parts: canonicalization, recursive traversal, cycle detection, and prune-mode behavior.

### Bipartite Trees

Some planners already output an alternating molecule/reaction tree:

```text
molecule -> reaction -> molecule -> reaction -> ...
```

Use `build_bipartite_molecule(...)` when raw nodes expose molecule/reaction roles and child links. This is the closest raw shape to the schema-2 model.

### Precursor Maps

Some planners output route strings or reaction tables that can be reduced to a product-to-reactants map:

```python
{
    "target": ["intermediate", "stock_a"],
    "intermediate": ["stock_b", "stock_c"],
}
```

Use `build_molecule_from_precursor_map(...)` for this shape. The adapter parses the raw format into the map, chooses the root product, and lets the helper walk the route.

### Plain Molecule Trees

Some raw routes are nested molecule trees where a parent molecule is made from its children, but there is no explicit reaction node.

Use `build_plain_tree_molecule(...)` for this shape. The adapter provides callbacks for SMILES, children, molecule annotations, and reaction fields.

## Target Validation

If `target` is provided, the adapted root must match it. A route for the wrong target is not a bad route for the requested target; it is the wrong object.

```python
from retrocast.adapters.errors import adapter_target_mismatch

if target is not None and route.target.smiles != target.smiles:
    raise adapter_target_mismatch(
        "myadapter",
        target.id,
        expected_smiles=target.smiles,
        actual_smiles=route.target.smiles,
    )
```

Some raw formats cannot even be interpreted without a target. For example, a route string may describe disconnections but not name the root in a reliable way. In that case, missing `target` is an adapter error.

## Adapter Errors

Adapter errors are part of evaluation accounting. Messages are for humans, but `code` and `context` are what workflows aggregate.

Keep `context["adapter"]` lowercase and aligned with the registry slug. Use display capitalization, such as `DreamRetro` or `AiZynthFinder`, only in the human message.

Common codes:

| code | meaning |
| --- | --- |
| `adapter.schema_invalid` | raw payload does not match the adapter's expected input shape |
| `adapter.target_mismatch` | the adapted route root is not the requested target |
| `adapter.cycle_detected` | the raw graph revisits the same molecule in one route path |
| `adapter.node_type_invalid` | a graph slot contains a molecule where a reaction is required, or the reverse |
| `adapter.node_missing` | a raw edge references a node absent from the lookup table |
| `adapter.route_string_empty` | a string route field is empty after parsing |
| `adapter.route_string_invalid` | a string route field has malformed step boundaries or missing reactants/products |
| `adapter.route_transform_failed` | fallback for a route-local transform failure that lacks a narrower code |
| `adapter.unsupported_feature` | a valid request asks for an adapter feature that is not implemented |

Chemistry failures use `chem.*` codes, usually `chem.invalid_smiles` or `chem.runtime_error`. For the shared exception shape and workflow-level accounting, see [Error Handling](errors.md).

### Adapter Error Examples

`adapter.cycle_detected`: the raw route loops back to a molecule already in the same path.

```json
{
  "smiles": "CCO",
  "children": [
    {
      "smiles": "CC=O",
      "children": [{ "smiles": "CCO", "children": [] }]
    }
  ]
}
```

`adapter.node_type_invalid`: a bipartite molecule has a molecule child where a reaction node is required.

```json
{
  "type": "mol",
  "smiles": "CCO",
  "children": [{ "type": "mol", "smiles": "CC=O", "children": [] }]
}
```

`adapter.route_string_invalid`: a route string exists, but it does not match the adapter grammar.

```json
{ "succ": true, "routes": "CCO>CC=O" }
```

`adapter.schema_invalid`: the raw field has the wrong type before route transformation starts.

```json
{ "succ": true, "routes": 123 }
```

If `adapter.route_transform_failed` appears often in manifests, promote the recurring case to a more specific code.

## Registration

Canonical adapter modules live under `retrocast.adapters` without an `_adapter` suffix, for example `retrocast.adapters.paroutes`.

Register the adapter in `retrocast.adapters.registry`:

```python
ADAPTER_TYPES = {
    "aizynthfinder": AiZynthFinderAdapter,
    "myadapter": MyAdapter,
}
```

Use a stable lowercase slug.

## Tests

Each adapter should use `tests.adapters.base.AdapterContractSuite` and add adapter-specific regression tests for its raw format.

The contract suite checks that the adapter:

- extracts raw route entries from a valid raw artifact.
- rejects malformed raw artifacts with `adapter.schema_invalid`.
- casts one valid raw route into a schema-2 `Route`.
- rejects malformed route payloads with `adapter.schema_invalid`.
- rejects target mismatches with `adapter.target_mismatch`.
- rejects invalid SMILES in `strict` mode.
- prunes invalid leaves in `prune` mode when the adapter claims prune support.

Run focused adapter tests with:

```bash
uv run pytest packages/retrocast-rs/tests/adapters/test_myadapter.py
```
