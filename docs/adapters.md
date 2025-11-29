# Writing a Custom Adapter

The adapter is the "air gap" between a model's internal representation and RetroCast's canonical `Route` schema. If you are integrating a new model, you will likely need to write a new adapter.

## The Adapter Contract

A RetroCast adapter is a class that inherits from `BaseAdapter`. It implements a single method, `cast`, which validates raw model output and transforms it into `Route` objects.

```python
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.models.chem import Route, TargetIdentity
from typing import Any, Generator

class MyModelAdapter(BaseAdapter):
    def cast(self, raw_target_data: Any, target: TargetIdentity) -> Generator[Route, None, None]:
        """
        Args:
            raw_target_data: A single entry from the raw results file (dict or list).
            target: The expected target identity (ID and canonical SMILES).
            
        Yields:
            Valid Route objects.
        """
        # 1. Validate raw_target_data (Pydantic recommended)
        # 2. Transform to Route objects
        # 3. Yield valid routes
        yield route
```

## Common Architecture Patterns

Most retrosynthesis models output data in one of three patterns. RetroCast provides helper functions (`retrocast.adapters.common`) to handle the heavy lifting for these patterns, including recursion and cycle detection.

### Pattern A: Bipartite Graph Recursion
**Used by:** AiZynthFinder, SynPlanner, Syntheseus

In this pattern, the output is a nested JSON tree where Molecule nodes point to Reaction nodes, which point to reactant Molecule nodes.

**Helper:** `build_molecule_from_bipartite_node`

To use this, your raw data structure must conform (via duck typing or Protocol) to the `BipartiteMolNode` interface: it must have `smiles` (str), `type` ("mol"), and `children` (list of reaction nodes).

```python
from retrocast.adapters.common import build_molecule_from_bipartite_node

class MyAdapter(BaseAdapter):
    def cast(self, raw_data, target):
        # Validate that raw_data fits the Bipartite schema
        validated = MyRawOutput.model_validate(raw_data)
        
        for i, tree_root in enumerate(validated.trees):
            try:
                # The helper handles the recursive tree construction
                target_mol = build_molecule_from_bipartite_node(tree_root)
                
                # Verify the root matches the target
                if target_mol.smiles != target.smiles:
                    continue

                yield Route(target=target_mol, rank=i+1, metadata={})
            except Exception:
                continue
```

### Pattern B: Precursor Map
**Used by:** Retro*, DreamRetro, SynLlama

In this pattern, the output is a flat list of reactions or a string representation (e.g., `P >> R1.R2 | R1 >> R3`). The tree structure is implicit in the connectivity.

**Helper:** `build_molecule_from_precursor_map`

You simply need to parse the raw format into a Python dictionary mapping `Product SMILES -> [Reactant SMILES, ...]`.

```python
from retrocast.adapters.common import build_molecule_from_precursor_map

class MyAdapter(BaseAdapter):
    def cast(self, raw_data, target):
        # 1. Parse your model's specific string format
        # Input: "target >> int_1.int_2 | int_1 >> sm_1.sm_2"
        # Output: {"target": ["int_1", "int_2"], "int_1": ["sm_1", "sm_2"]}
        precursor_map = self._parse_custom_string(raw_data["route_string"])
        
        # 2. Build the tree
        # The helper walks the map recursively starting from the target SMILES
        target_mol = build_molecule_from_precursor_map(target.smiles, precursor_map)
        
        yield Route(target=target_mol, rank=1, metadata={})
```

### Pattern C: Custom / Mixed
**Used by:** DirectMultiStep (DMS), ASKCOS

Some models have unique structures that don't fit the above patterns (e.g., graphs defined by edge lists, or recursive trees that don't strictly alternate molecule/reaction nodes).

For these, you must implement your own recursive builder. See `retrocast.adapters.dms_adapter` for a reference implementation.

## Implementation Guidelines

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
*   **Always** canonicalize raw SMILES using `retrocast.chem.canonicalize_smiles`.
*   The standard helpers (`build_molecule_...`) do this automatically.
*   If writing a custom builder, you must call it explicitly.
*   Always check that the root of your built tree matches `target.smiles`.

### 3. Cycle Detection
Retrosynthetic graphs must be acyclic trees.
*   The standard helpers include cycle detection (raising `AdapterLogicError` if a node appears twice in a path).
*   If writing a custom builder, maintain a `visited` set during recursion.

### 4. Metadata
Do not discard model-specific data (scores, template IDs, etc.). Store it in the `metadata` dictionary of the `Molecule` or `ReactionStep`. This data is preserved throughout the pipeline and can be used for custom analysis later.

## Registration

Once your adapter logic is written, you must register it so the CLI can find it.

### 1. Add to Factory
Add your adapter to the map in `src/retrocast/adapters/factory.py`:

```python
from retrocast.adapters.my_adapter import MyModelAdapter

ADAPTER_MAP = {
    "aizynth": AizynthAdapter(),
    # ...
    "my-model": MyModelAdapter(),
}
```

### 2. Update Config
When using the adapter in a project, reference the key from `ADAPTER_MAP` in your `retrocast-config.yaml`:

```yaml
models:
  experimental-run-1:
    adapter: my-model
    raw_results_filename: predictions.json
```

## Testing Your Adapter

RetroCast provides a strict test harness to ensure your adapter behaves correctly. Create a test file inheriting from `BaseAdapterTest`.

```python
# tests/adapters/test_my_adapter.py
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

This test suite automatically verifies that your adapter:
1.  Correctly parses valid data.
2.  Rejects invalid schemas without crashing (returns empty generator).
3.  Correctly identifies target mismatches.
4.  Handles failed predictions gracefully.
