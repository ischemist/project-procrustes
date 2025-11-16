# Adding a New Model Adapter

The adapter is the bridge from a model's unique output format to RetroCast's canonical schema. This document explains how to write one.

## Overview

Most model outputs fall into one of a few common patterns. Identify the pattern, use the appropriate common builder, and your adapter will be trivial.

The adapter is responsible for:
- **Parsing** raw model output (JSON, pickles, text files, etc.)
- **Validating** the structure using Pydantic schemas
- **Transforming** to the canonical `Route` format
- **Handling errors** gracefully (invalid routes, schema mismatches, etc.)

## Common Patterns

### Pattern A: Bipartite Graph

**Examples**: AiZynthFinder, SynPlanner

If the raw output is a JSON tree where molecule nodes point to reaction nodes and vice-versa:

```python
# in retrocast/adapters/bipartite_model_adapter.py
from collections.abc import Generator
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, RootModel, ValidationError
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import build_molecule_from_bipartite_node
from retrocast.schemas import Route, TargetInput
from retrocast.exceptions import RetroCastException
from retrocast.utils.logging import logger

# --- pydantic schemas for raw input validation ---
class BipartiteBaseNode(BaseModel):
    smiles: str
    children: list["BipartiteNode"] = Field(default_factory=list)

class BipartiteMoleculeInput(BipartiteBaseNode):
    type: Literal["mol"]
    in_stock: bool

class BipartiteReactionInput(BipartiteBaseNode):
    type: Literal["reaction"]
    metadata: dict[str, Any] = Field(default_factory=dict)

BipartiteNode = Annotated[BipartiteMoleculeInput | BipartiteReactionInput, Field(discriminator="type")]

class BipartiteRouteList(RootModel[list[BipartiteMoleculeInput]]):
    pass

class BipartiteModelAdapter(BaseAdapter):
    def cast(self, raw_data: Any, target_input: TargetInput) -> Generator[Route, None, None]:
        validated_routes = BipartiteRouteList.model_validate(raw_data)
        for i, root_node in enumerate(validated_routes.root, start=1):
            try:
                # Use the helper to get a Molecule object
                target_molecule = build_molecule_from_bipartite_node(root_node)

                # Verify the target SMILES matches
                if target_molecule.smiles != target_input.smiles:
                    logger.warning(
                        f"Mismatched SMILES for target '{target_input.id}'. "
                        f"Expected {target_input.smiles}, got {target_molecule.smiles}."
                    )
                    continue

                yield Route(target=target_molecule, rank=i, metadata={})
            except (RetroCastException, ValidationError) as e:
                logger.warning(f"Route for '{target_input.id}' failed validation or processing: {e}")

**Key points**:
1. Define Pydantic schemas for the raw input structure matching the BipartiteMolNode/BipartiteRxnNode protocols
2. Use `build_molecule_from_bipartite_node` from `retrocast.adapters.common`
3. Yield `Route` objects with target `Molecule`, rank, and metadata for each valid route

### Pattern B: Precursor Map

**Examples**: Retro*, DreamRetro

If the raw output can be parsed into a `dict[product_smiles, list[reactant_smiles]]`:

```python
# in retrocast/adapters/precursor_model_adapter.py
from collections.abc import Generator
from typing import Any
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.adapters.common import PrecursorMap, build_molecule_from_precursor_map
from retrocast.schemas import Route, TargetInput
from retrocast.exceptions import RetroCastException
from retrocast.utils.logging import logger

class PrecursorModelAdapter(BaseAdapter):
    def _parse_route_string(self, route_str: str) -> PrecursorMap:
        # model-specific logic to parse the string "p1>>r1.r2|p2>>r3..."
        precursor_map: PrecursorMap = {}
        # ... your parsing logic here ...
        return precursor_map

    def cast(self, raw_data: Any, target_input: TargetInput) -> Generator[Route, None, None]:
        try:
            precursor_map = self._parse_route_string(raw_data["routes"])
            # Use the helper to build the tree from the target SMILES
            target_molecule = build_molecule_from_precursor_map(target_input.smiles, precursor_map)
            yield Route(target=target_molecule, rank=1, metadata={})
        except (RetroCastException, KeyError) as e:
            logger.warning(f"Route for '{target_input.id}' failed: {e}")

**Key points**:
1. Write a model-specific parser to extract the precursor map (`dict[SmilesStr, list[SmilesStr]]`)
2. Use `build_molecule_from_precursor_map` from `retrocast.adapters.common`
3. The builder handles recursive tree construction, cycle detection, and validation

### Pattern C: Custom Recursive

**Examples**: DirectMultiStep (DMS)

If the raw output is already a recursive tree but with a different schema:

```python
# in retrocast/adapters/custom_model_adapter.py
from collections.abc import Generator
from typing import Any
from pydantic import BaseModel, RootModel, Field
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.domain.chem import canonicalize_smiles, get_inchi_key
from retrocast.schemas import Molecule, ReactionStep, Route, TargetInput

# --- pydantic schemas for raw input validation ---
class CustomTree(BaseModel):
    smiles: str
    children: list["CustomTree"] = Field(default_factory=list)

class CustomRouteList(RootModel[list[CustomTree]]):
    pass

class CustomModelAdapter(BaseAdapter):
    def _build_molecule(self, custom_node: CustomTree) -> Molecule:
        # Logic to convert one custom node to one Molecule
        canon_smiles = canonicalize_smiles(custom_node.smiles)
        inchikey = get_inchi_key(canon_smiles)
        
        synthesis_step = None
        if custom_node.children:
            reactants = [self._build_molecule(child) for child in custom_node.children]
            synthesis_step = ReactionStep(
                reactants=reactants,
                mapped_smiles=None,
                template=None,
                reagents=None,
                solvents=None,
                metadata={}
            )
        
        return Molecule(
            smiles=canon_smiles,
            inchikey=inchikey,
            synthesis_step=synthesis_step,
            metadata={}
        )

    def cast(self, raw_data: Any, target_input: TargetInput) -> Generator[Route, None, None]:
        validated_routes = CustomRouteList.model_validate(raw_data)
        for i, root_node in enumerate(validated_routes.root, start=1):
            target_molecule = self._build_molecule(root_node)
            yield Route(target=target_molecule, rank=i, metadata={})
```

**Key points**:
1. Define Pydantic schemas for the raw tree structure
2. Write a recursive builder (`_build_molecule`) that traverses the raw tree
3. Construct the canonical `Molecule` tree with `ReactionStep` objects linking reactants
4. Always canonicalize SMILES and generate InChIKeys for all molecules

## Integration Steps

Once your adapter class is implemented, follow these steps:

### 1. Write Tests

Create `tests/adapters/test_new_adapter.py` and inherit from `BaseAdapterTest`:

```python
# in tests/adapters/test_new_adapter.py
import pytest
from tests.adapters.test_base_adapter import BaseAdapterTest
from retrocast.adapters.new_model_adapter import NewModelAdapter
from retrocast.schemas import TargetInput

class TestNewModelAdapterUnit(BaseAdapterTest):
    @pytest.fixture
    def adapter_instance(self):
        return NewModelAdapter()

    @pytest.fixture
    def raw_valid_route_data(self) -> Any:
        # return a valid json blob for your model
        ...

    @pytest.fixture
    def raw_unsuccessful_run_data(self) -> Any:
        # return data representing a failed prediction
        ...

    @pytest.fixture
    def raw_invalid_schema_data(self) -> Any:
        # return malformed data that should fail validation
        ...

    @pytest.fixture
    def target_input(self) -> TargetInput:
        # return target info matching your valid route

    @pytest.fixture
    def mismatched_target_info(self) -> TargetInput:
        # return target info that doesn't match your route
        ...
```

The `BaseAdapterTest` class provides a standard test suite that verifies:
- Valid routes are parsed correctly
- Invalid schemas are rejected
- Target mismatches are caught
- Error handling works as expected

### 2. Register the Adapter

Add your adapter to the factory in `retrocast/adapters/factory.py`:

```python
# in retrocast/adapters/factory.py
from retrocast.adapters.new_model_adapter import NewModelAdapter

ADAPTER_MAP: dict[str, BaseAdapter] = {
    "aizynth": AizynthAdapter(),
    "retrostar": RetrostarAdapter(),
    # ...
    "new-model": NewModelAdapter(),  # <-- ADD THIS
}
```

### 3. Update Configuration

Add an entry for your model in `retrocast-config.yaml`:

```yaml
# in retrocast-config.yaml
models:
  # ...
  new-model:
    adapter: new-model  # must match the key from ADAPTER_MAP
    raw_results_filename: results.json.gz
    sampling:
      strategy: top-k
      k: 10
```

**Configuration fields**:
- `adapter`: The adapter key from `ADAPTER_MAP`
- `raw_results_filename`: Expected filename in `data/evaluations/<model>/<dataset>/`
- `sampling`: How to sample routes if the model returns more than needed
  - `strategy`: `"top-k"` or `"all"`
  - `k`: Number of routes to keep (for top-k strategy)

### 4. Create Model Scripts

Add numbered scripts to `scripts/<model-name>/` following the pattern:

```
scripts/new-model/
├── 1-download-assets.py      # Download model checkpoints, config files, etc.
├── 2-prepare-data.py          # Convert stock files, prepare inputs (optional)
└── 3-run-new-model.py         # Run inference and save results.json.gz
```

Each script should:
- Include a module-level docstring with usage examples
- Accept `--target-name` argument (dataset name)
- Save output to `data/evaluations/<model-name>/<dataset-name>/results.json.gz`
- Use `uv run --extra <model-extra>` to ensure correct dependencies

**Example script header**:

```python
"""
Run NewModel predictions on a target set.

Usage:
    uv run --extra new-model scripts/new-model/3-run-new-model.py --target-name "uspto-190"
"""
```

## Testing Your Adapter

Run the adapter tests:

```bash
# Test only your adapter
pytest tests/adapters/test_new_adapter.py -v

# Run the full adapter test suite
pytest tests/adapters/ -v
```

Test the full pipeline:

```bash
# 1. Run your model scripts to generate raw output
uv run --extra new-model scripts/new-model/3-run-new-model.py --target-name "uspto-190"

# 2. Process with RetroCast
uv run scripts/process-predictions.py process --model new-model --dataset uspto-190

# 3. Verify the hash is reproducible
uv run scripts/verify-hash.py --model new-model --dataset uspto-190
```

## Common Pitfalls

1. **SMILES canonicalization**: Always canonicalize SMILES using `canonicalize_smiles()` before creating nodes. Different SMILES for the same molecule will break deduplication.

2. **Target mismatch**: Ensure the root molecule in your tree matches `target_info.smiles` (after canonicalization).

3. **Circular references**: The tree must be acyclic. If a molecule appears multiple times, create separate `Molecule` instances (deduplication happens later).

4. **Error handling**: Wrap tree building in try/except and log warnings for invalid routes. Don't fail the entire batch because one route is malformed.

5. **Metadata preservation**: Use the `metadata` fields on `Molecule` and `ReactionStep` to preserve model-specific data (scores, templates, etc.).

## Advanced Topics

### Custom Validation

If your model has special validation requirements, override the `validate()` method:

```python
class NewModelAdapter(BaseAdapter):
    def validate(self, raw_data: Any, target_info: TargetInput) -> None:
        super().validate(raw_data, target_info)
        # additional model-specific checks
        if not isinstance(raw_data, dict):
            raise ValueError("NewModel output must be a dictionary")
```

### Handling Multiple Output Files

If your model produces multiple files per run, override the loading logic:

```python
class NewModelAdapter(BaseAdapter):
    def load_raw_data(self, results_dir: Path) -> Any:
        # load multiple files and combine
        tree_file = results_dir / "trees.pkl"
        scores_file = results_dir / "scores.json"
        # ... custom loading logic
        return combined_data
```

### Metadata Extraction

Preserve model-specific information in metadata fields:

```python
synthesis_step = ReactionStep(
    reactants=[...],
    mapped_smiles=reaction_data.get("mapped_smiles"),
    template=reaction_data.get("template"),
    reagents=reaction_data.get("reagents"),
    solvents=reaction_data.get("solvents"),
    metadata={
        "confidence": reaction_data.get("score"),
        "source": "retro-model-v2",
    }
)
```

This data is preserved in the canonical format and can be used for downstream analysis.

## Reference Implementations

For real-world examples, see:

- **Bipartite graph**: `retrocast/adapters/aizynth_adapter.py`, `retrocast/adapters/synplanner_adapter.py`
- **Precursor map**: `retrocast/adapters/retrostar_adapter.py`, `retrocast/adapters/dreamretro_adapter.py`
- **Custom recursive**: `retrocast/adapters/dms_adapter.py`
- **Pickle files**: `retrocast/adapters/multistepttl_adapter.py`

## Questions?

Open an issue at [github.com/ischemist/project-procrustes/issues](https://github.com/ischemist/project-procrustes/issues).
