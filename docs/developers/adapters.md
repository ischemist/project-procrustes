---
icon: lucide/code-xml
---

# Writing a Custom Adapter

The adapter is the **"air gap"** between a model's internal representation and RetroCast's canonical `Route` schema. If you are integrating a new model, you will likely need to write a new adapter.

!!! tip "When do I need a custom adapter?"

    You need a custom adapter when integrating a new retrosynthesis model whose output format differs from existing adapters. RetroCast already supports 10+ models—check the [supported adapters](#common-architecture-patterns) first!

## The Adapter Contract

A RetroCast adapter is route-first. It inherits from `BaseAdapter` and implements ==two explicit responsibilities==:

1. `iter_raw_entries(...)`: split a provider artifact into raw route-like entries
2. `cast(...)`: convert ==one== raw route-like payload into ==one== canonical `Route`

```python title="src/retrocast/adapters/my_adapter.py"
from collections.abc import Iterator
from typing import Any

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.models.chem import Route, TargetIdentity


class MyModelAdapter(BaseAdapter):
    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        for row_index, raw_route in enumerate(raw_data, start=1):
            yield RawRouteEntry(
                payload=raw_route,
                source_key=source_key,
                source_row_index=row_index,
                source_order=row_index,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        expected_target: TargetIdentity | None = None,
        ignore_stereo: bool = False,
    ) -> Route:
        """
        Args:
            raw_route: One raw route-like payload from the provider artifact.
            expected_target: Optional benchmark or workflow target to validate against.
            ignore_stereo: Whether SMILES canonicalization should strip stereochemistry.

        Returns:
            One valid canonical Route object.
        """
        target_molecule = ...
        return Route(target=target_molecule, metadata={})
```

`iter_raw_entries(...)` is where corpus traversal lives. `cast(...)` should not know or care whether the caller is adapting a flat corpus, a target-local payload, or a benchmark-keyed prediction map.

Provider-output adaptation is the built-in RetroCast workflow for converting a raw provider artifact into canonical predictions. Adapter authors do not create a separate provider-output adapter class. Instead, implement `iter_raw_entries(...)` to split the raw artifact into route-like records and `cast(...)` to convert one record into one canonical `Route`. RetroCast's workflow layer then calls those hooks, wraps each `Route` in a `PredictedRoute` envelope, and preserves provider-level provenance such as ordering, source row metadata, or scalar prediction fields. Put chemistry structure on the canonical route; put provider-level ordering or scalar prediction metadata on `RawRouteEntry` or route metadata so corpus workflows can preserve it on the prediction envelope.

## Common Architecture Patterns

Most retrosynthesis models output data in one of three patterns. RetroCast provides helper functions (`retrocast.adapters.common`) to handle the heavy lifting for these patterns, including recursion and cycle detection.

!!! info "Use the helpers whenever possible"

    The built-in helpers (`build_molecule_from_bipartite_node`, `build_molecule_from_precursor_map`) handle canonicalization, cycle detection, and validation automatically. Don't reinvent the wheel!

### Pattern A: Bipartite Graph Recursion

**Used by:** AiZynthFinder, SynPlanner, Syntheseus  
**Helper:** `build_molecule_from_bipartite_node`

In this pattern, the output is a nested JSON tree where ==Molecule nodes point to Reaction nodes==, which point to reactant Molecule nodes.

To use this, your raw data structure must conform (via duck typing or Protocol) to the `BipartiteMolNode` interface: it must have `smiles` (str), `type` ("mol"), and `children` (list of reaction nodes).

```python hl_lines="4 11"
from collections.abc import Iterator

from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.adapters.errors import adapter_target_mismatch
from retrocast.adapters.common import build_molecule_from_bipartite_node
from retrocast.models.chem import Route

class MyAdapter(BaseAdapter):
    def iter_raw_entries(self, raw_data, *, source_key=None) -> Iterator[RawRouteEntry]:
        validated = MyRawOutput.model_validate(raw_data)
        for row_index, tree_root in enumerate(validated.trees, start=1):
            yield RawRouteEntry(payload=tree_root, source_key=source_key, source_order=row_index)

    def cast(self, raw_route, *, expected_target=None, ignore_stereo=False):
        # The helper handles the recursive tree construction # (1)!
        target_mol = build_molecule_from_bipartite_node(raw_route)

        # Verify the root matches the expected target when one is provided
        if expected_target is not None and target_mol.smiles != expected_target.smiles:
            raise adapter_target_mismatch(...)

        return Route(target=target_mol, metadata={})
```

1. The helper automatically handles canonicalization and cycle detection

### Pattern B: Precursor Map

**Used by:** Retro\*, DreamRetro, SynLlama  
**Helper:** `build_molecule_from_precursor_map`

In this pattern, the output is a ==flat list of reactions== or a string representation (e.g., `P >> R1.R2 | R1 >> R3`). The tree structure is implicit in the connectivity.

You simply need to parse the raw format into a Python dictionary mapping `Product SMILES -> [Reactant SMILES, ...]`.

```python hl_lines="8 11"
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import build_molecule_from_precursor_map
from retrocast.models.chem import Route

class MyAdapter(BaseAdapter):
    def cast(self, raw_route, *, expected_target=None, ignore_stereo=False):
        # 1. Parse your model's specific string format
        # Input: "target >> int_1.int_2 | int_1 >> sm_1.sm_2"
        # Output: {"target": ["int_1", "int_2"], "int_1": ["sm_1", "sm_2"]}
        precursor_map = self._parse_custom_string(raw_route["route_string"]) # (1)!

        if expected_target is None:
            raise ValueError("expected_target is required for this adapter")

        # 2. Build the tree
        # The helper walks the map recursively starting from the target SMILES # (2)!
        target_mol = build_molecule_from_precursor_map(expected_target.smiles, precursor_map)

        return Route(target=target_mol, metadata={})
```

1. Implement this parsing method specific to your model's format
2. The helper handles the recursive tree walking and validation

### Pattern C: Custom / Mixed

**Used by:** DirectMultiStep (DMS), ASKCOS  
**Helper:** None (roll your own)

Some models have unique structures that don't fit the above patterns (e.g., graphs defined by edge lists, or recursive trees that don't strictly alternate molecule/reaction nodes).

??? example "Reference implementation"

    See `retrocast.adapters.dms_adapter` for a complete example of implementing a custom recursive builder with cycle detection.

### Flat Completion Corpora: Ursa LLM

`UrsaLlmAdapter` is the reference example of a route-first adapter for flat raw artifacts. Its input is a completion corpus containing `<synthesis_step>` XML blocks, usually as `.jsonl` or `.jsonl.gz`, not a benchmark-keyed `results.json.gz` mapping.

- canonical adapter key: `ursa-llm`
- raw artifact shape: list of records with `completion` text and `meta.product_smiles`
- adapter seam:
  - `iter_raw_entries(...)` yields one `RawRouteEntry` per completion row
  - `cast(...)` parses one completion string into one canonical `Route`

When `meta.product_smiles` is present, Ursa raw completions can be adapted into a prediction route corpus directly, without a benchmark and without a preprocessing step that re-keys the file by target. benchmark alignment happens later in the explicit `collect` workflow.

## Implementation Guidelines

!!! warning "Critical requirements"

    Your adapter **must** handle:

    1. **Canonicalization** - Use `retrocast.chem.canonicalize_smiles` for all SMILES
    2. **Cycle detection** - Ensure no molecule appears twice in a path
    3. **Target validation** - Verify the root matches `expected_target.smiles` when an expected target is provided

## Adaptation Errors

Adapter error `code` values are lowercase machine contracts. Keep adapter names in `context["adapter"]` lowercase so callers can aggregate by `ADAPTER_MAP` key. Human-facing messages should use proper model capitalization, such as `DreamRetro`, `DMS`, or `MolBuilder`.

For the shared exception shape, wrapping rules, and workflow-level failure accounting, see [Error Handling](errors.md).

An adapter may raise these expected errors from `cast`:

| exception | code | when it is raised | caller policy |
| --- | --- | --- | --- |
| `AdapterSchemaError` | `adapter.schema_invalid` | the raw route payload or entry does not match the adapter's declared input schema | workflow records the failure and continues |
| `AdapterLogicError` | `adapter.target_mismatch` | a transformed route root does not match the benchmark target | adapter logs/skips that route when other routes may still be usable |
| `AdapterLogicError` | `adapter.cycle_detected` | route graph revisits the same molecule in one path | adapter logs/skips that route when the failure is route-local |
| `AdapterLogicError` | `adapter.node_type_invalid` | a bipartite graph node has the wrong role, such as molecule where reaction is required | adapter logs/skips that route when the failure is route-local |
| `AdapterLogicError` | `adapter.node_missing` | graph edges reference a node absent from the raw lookup tables | adapter logs/skips that route when the failure is route-local |
| `AdapterLogicError` | `adapter.route_string_empty` | a string-based route payload is empty after parsing | adapter logs/skips that route when the failure is route-local |
| `AdapterLogicError` | `adapter.route_string_invalid` | a string-based route payload has malformed step boundaries or missing reactants/products | adapter logs/skips that route when the failure is route-local |
| `AdapterLogicError` | `adapter.route_transform_failed` | fallback for route-local transform failures that do not fit a narrower code | adapter logs/skips that route when the failure is route-local |
| `UnsupportedAdapterFeatureError` | `adapter.unsupported_feature` | a valid request asks for a feature the adapter does not support | caller should treat this as a fatal configuration/request failure |
| `ChemError` | `chem.invalid_smiles`, `chem.runtime_error` | raw route molecules cannot be canonicalized or processed by RDKit | adapter logs/skips that route when the failure is route-local |

Adapter resolution happens before `cast` and raises `AdapterResolutionError` (`adapter.unknown` or `adapter.resolution_missing`) when the CLI or manifest cannot select an adapter.

Use `retrocast.adapters.errors.adapter_schema_error()` and `adapter_target_mismatch()` where they fit; they keep messages, codes, and context consistent.

During `ingest`, workflow boundaries aggregate target-level `AdapterError` and `ChemError` failures by `error.code`. In the ingest CLI path, that summary is persisted in manifest `statistics.failures_by_code`.

### Adapter Error Examples

These examples use tiny fake payloads to show what each route-local code means.

`adapter.cycle_detected`: the route graph loops back to a molecule already in the same path.

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

`adapter.node_type_invalid`: a graph slot contains the wrong kind of node. Bipartite adapters expect molecule -> reaction -> molecule.

```json
{
  "type": "mol",
  "smiles": "CCO",
  "children": [{ "type": "mol", "smiles": "CC=O", "children": [] }]
}
```

`adapter.node_missing`: an edge/reference points to an id or SMILES key that is absent from the raw lookup tables.

```json
{
  "uuid2smiles": { "root": "CCO", "rxn1": "CC=O>>CCO" },
  "node_dict": { "CCO": { "type": "chemical", "smiles": "CCO" } },
  "pathways": [[{ "source": "root", "target": "rxn1" }]]
}
```

Here `rxn1` resolves to `CC=O>>CCO`, but `node_dict["CC=O>>CCO"]` is missing.

`adapter.route_string_empty`: a string-based route field is empty after trimming/splitting.

```json
{ "succ": true, "routes": "" }
```

`adapter.route_string_invalid`: the route string exists, but its grammar is malformed.

```json
{ "succ": true, "routes": "CCO>CC=O" }
```

RetroStar and DreamRetro-style route steps need product, metadata/reagents, and reactants, e.g. `CCO>0.9>CC=O.[H][H]`.

`adapter.target_mismatch`: the adapter produced a valid route, but the root is not the benchmark target.

```json
{
  "target_id": "ethanol",
  "expected_smiles": "CCO",
  "actual_root_smiles": "CCC"
}
```

`adapter.schema_invalid`: the raw payload does not match the adapter's top-level expected schema.

```json
{ "succ": true, "routes": 123 }
```

If `routes` must be a string, this is schema-invalid.

`adapter.route_transform_failed`: fallback for route-local transformation failures that do not fit a sharper bucket. If this shows up often in manifests or logs, promote the recurring case to a dedicated code.

### 1. Define Pydantic Schemas

Always define Pydantic models for the **raw** input format. This separates validation logic from transformation logic and ensures bad data is rejected early.

```python
class MyRawNode(BaseModel):
    smiles: str
    probability: float
    children: list["MyRawNode"]
```

### 2. Canonicalization

RetroCast relies on exact SMILES matching.

- **Always** canonicalize raw SMILES using `retrocast.chem.canonicalize_smiles`
- The standard helpers (`build_molecule_...`) do this automatically
- If writing a custom builder, you must call it explicitly
- Always check that the root of your built tree matches `target.smiles`

### 3. Cycle Detection

Retrosynthetic graphs must be acyclic trees.

- The standard helpers include cycle detection (raising `AdapterLogicError` with `adapter.cycle_detected` if a node appears twice in a path)
- If writing a custom builder, maintain a `visited` set during recursion

### 4. Metadata

Do not discard model-specific data (scores, template IDs, etc.). Store route-level prediction scores in `Route.metadata` so `PredictedRoute` can expose them, and store step- or molecule-local data in the `metadata` dictionary of the relevant `Molecule` or `ReactionStep`. This data is preserved throughout the pipeline and can be used for custom analysis later.

## Registration

Once your adapter logic is written, you must register it so the CLI can find it.

=== "1. Register the Adapter"

    Add your adapter to the map in `src/retrocast/adapters/__init__.py`:

    ```python title="src/retrocast/adapters/__init__.py"
    from retrocast.adapters.my_adapter import MyModelAdapter

    ADAPTER_TYPES = {
        "aizynth": AizynthAdapter,
        # ...
        "my-model": MyModelAdapter, # (1)!
    }
    ```

    1. Use a descriptive, lowercase key with hyphens

=== "2. Update Config"

    When using the adapter in a project, reference the key from `ADAPTER_TYPES` in your `retrocast-config.yaml`:

    ```yaml title="retrocast-config.yaml"
    models:
      experimental-run-1:
        adapter: my-model # (1)!
        raw_results_filename: predictions.json
    ```

    1. Must match the key you added to `ADAPTER_TYPES`

## Testing Your Adapter

RetroCast provides a strict test harness to ensure your adapter behaves correctly. Create a test file inheriting from `BaseAdapterTest`.

```python title="tests/adapters/test_my_adapter.py"
from tests.adapters.test_base_adapter import BaseAdapterTest
from retrocast.adapters.my_adapter import MyAdapter

class TestMyAdapter(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return MyAdapter()

    @pytest.fixture
    def raw_valid_route_data(self):
        # Return a sample JSON blob that represents a valid output
        return {"smiles": "CCO", "tree": ...}

    @pytest.fixture
    def raw_unsuccessful_run_data(self):
        # Return data representing a failed prediction (empty list, success=False, etc.)
        return {"success": False}

    @pytest.fixture
    def raw_invalid_schema_data(self):
        # Return malformed data that should fail Pydantic validation
        return {"malformed": True}

    # ... implement target identity fixtures ...
```

Run the tests using `pytest`:

```bash
pytest tests/adapters/test_my_adapter.py
```

!!! success "What the test suite validates"

    The `BaseAdapterTest` automatically verifies that your adapter:

    1. Correctly parses valid data
    2. Rejects invalid schemas with `AdapterSchemaError`
    3. Correctly identifies target mismatches
    4. Handles failed predictions gracefully
