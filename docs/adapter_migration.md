# Adapter Migration Guide: From DEPRECATE_schemas to New Route Schema

This document provides a step-by-step guide for migrating adapters from the old `BenchmarkTree` schema (in `DEPRECATE_schemas.py`) to the new `Route` schema (in `schemas.py`).

## Overview of Changes

### Old Schema (`DEPRECATE_schemas.py`)
- **Root**: `BenchmarkTree` containing `TargetInfo` and `MoleculeNode` (retrosynthetic tree root)
- **Node Structure**: `MoleculeNode` → `ReactionNode` → `MoleculeNode` (recursive bipartite graph)
- **Identification**: Path-dependent IDs + content-based hashes
- **Key Fields**: 
  - `MoleculeNode`: `id`, `molecule_hash`, `smiles`, `is_starting_material`, `reactions`
  - `ReactionNode`: `id`, `reaction_smiles`, `reactants`
  - `TargetInfo`: `smiles`, `id`

### New Schema (`schemas.py`)
- **Root**: `Route` containing a `Molecule` (target)
- **Node Structure**: `Molecule` → `ReactionStep` → `Molecule` (recursive tree)
- **Identification**: InChIKey (canonical) + SMILES (display)
- **Key Fields**:
  - `Route`: `target`, `rank`, `solvability`, `metadata`
  - `Molecule`: `smiles`, `inchikey`, `synthesis_step`, `metadata`, `is_leaf` (computed)
  - `ReactionStep`: `reactants`, `mapped_smiles`, `reagents` (list[SmilesStr]), `solvents` (list[SmilesStr]), `metadata`

### Key Differences

| Aspect | Old Schema | New Schema |
|--------|-----------|------------|
| Root object | `BenchmarkTree` | `Route` |
| Target info | Separate `TargetInfo` object | Part of root `Molecule` |
| Molecule ID | Path-dependent string `id` + `molecule_hash` | `inchikey` (canonical identifier) |
| Reaction ID | Path-dependent string `id` | No explicit ID |
| Leaf detection | `is_starting_material` boolean field | `is_leaf` computed property |
| Reaction SMILES | Required `reaction_smiles` field | Optional `mapped_smiles` field |
| Metadata | No metadata support | `metadata` dict on all objects |
| Multiple routes | Generator yields multiple `BenchmarkTree` | Each route has a `rank` field |

## Migration Steps

### Step 1: Update Imports

**Old:**
```python
from retrocast.domain.DEPRECATE_schemas import BenchmarkTree, TargetInfo, MoleculeNode, ReactionNode
```

**New:**
```python
from retrocast.schemas import Route, Molecule, ReactionStep
from retrocast.typing import SmilesStr, InchiKeyStr, ReactionSmilesStr
from retrocast.domain.chem import canonicalize_smiles, get_inchi_key
```

**Note**: `TargetInfo` is still imported from `domain.schemas` for the `adapt()` method signature (but not used in the output).

### Step 2: Update the `adapt()` Method Signature

**Old:**
```python
def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
```

**New:**
```python
def adapt(self, raw_target_data: Any, target_info: TargetInfo) -> Generator[Route, None, None]:
```

The input signature stays the same (uses `TargetInfo` for backward compatibility), but the output changes from `BenchmarkTree` to `Route`.

### Step 3: Update Tree Building Logic

#### For Precursor Map-Based Adapters (e.g., RetroStar, DreamRetro)

**Old Pattern:**
```python
from retrocast.adapters.common import build_tree_from_precursor_map

# Build the tree
retrosynthetic_tree = build_tree_from_precursor_map(
    smiles=target_info.smiles, 
    precursor_map=precursor_map
)

# Wrap in BenchmarkTree
return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)
```

**New Pattern:**
```python
from retrocast.domain.chem import get_inchi_key

# Build the tree recursively with new schema
target_molecule = self._build_molecule_from_precursor_map(
    smiles=target_info.smiles,
    precursor_map=precursor_map
)

# Wrap in Route with rank
return Route(target=target_molecule, rank=route_rank, metadata={})
```

#### For Bipartite Graph-Based Adapters (e.g., AiZynthFinder, SynPlanner)

**Old Pattern:**
```python
from retrocast.adapters.common import build_tree_from_bipartite_node

retrosynthetic_tree = build_tree_from_bipartite_node(
    raw_mol_node=raw_route_dict,
    path_prefix="retrocast-mol-root"
)

return BenchmarkTree(target=target_info, retrosynthetic_tree=retrosynthetic_tree)
```

**New Pattern:**
```python
# Build with new recursive function
target_molecule = self._build_molecule_from_bipartite_node(
    raw_mol_node=raw_route_dict
)

return Route(target=target_molecule, rank=route_rank, metadata={})
```

### Step 4: Implement New Recursive Builder

You'll need to create a new recursive function that builds `Molecule` objects instead of `MoleculeNode` objects. The key changes:

1. **Add InChIKey generation**: Every molecule needs an `inchikey`
2. **Remove path-based IDs**: No more `id` fields
3. **Change field names**: `reactions` → `synthesis_step`, `is_starting_material` → computed `is_leaf`
4. **Add metadata**: Populate `metadata` dicts where appropriate

**Example for Precursor Map:**
```python
def _build_molecule_from_precursor_map(
    self,
    smiles: SmilesStr,
    precursor_map: dict[SmilesStr, list[SmilesStr]],
    visited: set[SmilesStr] | None = None,
) -> Molecule:
    """Recursively build a Molecule tree from a precursor map."""
    if visited is None:
        visited = set()
    
    # Cycle detection
    if smiles in visited:
        logger.warning(f"Cycle detected for {smiles}, treating as leaf")
        return Molecule(
            smiles=smiles,
            inchikey=get_inchi_key(smiles),
            synthesis_step=None,
            metadata={}
        )
    
    new_visited = visited | {smiles}
    
    # Check if this is a leaf (not in precursor map)
    if smiles not in precursor_map:
        return Molecule(
            smiles=smiles,
            inchikey=get_inchi_key(smiles),
            synthesis_step=None,
            metadata={}
        )
    
    # Build reactants recursively
    reactant_molecules = []
    for reactant_smiles in precursor_map[smiles]:
        reactant_mol = self._build_molecule_from_precursor_map(
            smiles=reactant_smiles,
            precursor_map=precursor_map,
            visited=new_visited
        )
        reactant_molecules.append(reactant_mol)
    
    # Create the reaction step
    synthesis_step = ReactionStep(
        reactants=reactant_molecules,
        mapped_smiles=None,  # Add if available from raw data
        reagents=None,  # list[SmilesStr] if available
        solvents=None,  # list[SmilesStr] if available
        metadata={}
    )
    
    # Create the molecule with its synthesis step
    return Molecule(
        smiles=smiles,
        inchikey=get_inchi_key(smiles),
        synthesis_step=synthesis_step,
        metadata={}
    )
```

**Example for Bipartite Graph:**
```python
def _build_molecule_from_bipartite_node(
    self,
    raw_mol_node: BipartiteMolNode,
) -> Molecule:
    """Recursively build a Molecule tree from a bipartite graph node."""
    canon_smiles = canonicalize_smiles(raw_mol_node.smiles)
    
    # Check if this is a leaf
    is_leaf = raw_mol_node.in_stock or not bool(raw_mol_node.children)
    
    if is_leaf:
        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=None,
            metadata={}
        )
    
    # In a tree, molecule has at most one reaction
    if len(raw_mol_node.children) > 1:
        logger.warning(f"Molecule {canon_smiles} has multiple reactions, using first only")
    
    raw_reaction_node = raw_mol_node.children[0]
    
    # Build reactants recursively
    reactant_molecules = []
    for reactant_raw in raw_reaction_node.children:
        reactant_mol = self._build_molecule_from_bipartite_node(reactant_raw)
        reactant_molecules.append(reactant_mol)
    
    # Create the reaction step
    synthesis_step = ReactionStep(
        reactants=reactant_molecules,
        mapped_smiles=None,  # Populate if available
        reagents=None,  # list[SmilesStr] if available
        solvents=None,  # list[SmilesStr] if available
        metadata={}
    )
    
    return Molecule(
        smiles=canon_smiles,
        inchikey=get_inchi_key(canon_smiles),
        synthesis_step=synthesis_step,
        metadata={}
    )
```

### Step 5: Handle Multiple Routes and Ranking

If the model produces multiple routes per target, ensure each route gets the correct rank:

**Old:**
```python
# Just yield trees without ranking
for i, raw_route in enumerate(raw_routes_list):
    tree = self._transform(raw_route, target_info)
    yield tree
```

**New:**
```python
# Yield routes with rank (1-indexed)
for i, raw_route in enumerate(raw_routes_list):
    route = self._transform(raw_route, target_info, rank=i + 1)
    yield route
```

### Step 6: Populate Metadata Fields

The new schema supports metadata at multiple levels. Use this to preserve model-specific information:

- **Route metadata**: Overall route scores, search time, etc.
- **Molecule metadata**: Node scores, purchasability flags, etc.
- **ReactionStep metadata**: Template scores, template IDs, probabilities, etc.

**Example:**
```python
# If your raw data has a route score
route_metadata = {"route_score": raw_route.get("score", 0.0)}

# If your raw data has reaction templates
reaction_metadata = {"template_id": raw_reaction.get("template")}

synthesis_step = ReactionStep(
    reactants=reactant_molecules,
    metadata=reaction_metadata
)

route = Route(
    target=target_molecule,
    rank=rank,
    metadata=route_metadata
)
```

### Step 7: Update Tests

#### Test Organization Structure

Tests for each adapter should be organized into three classes:

1. **`Test{Adapter}Unit`** (inherits from `BaseAdapterTest`): 
   - Provides standard unit tests for common failure modes
   - Tests: success, unsuccessful run, invalid schema, mismatched SMILES
   - Uses minimal fixtures defined in the test class

2. **`Test{Adapter}Contract`** (integration tests):
   - Verifies the adapter produces valid `Route` objects with all required fields
   - Tests that fields are populated (not None) across all routes
   - Uses shared fixtures with `scope="class"` to avoid re-running adaptation
   - Examples: all routes have ranks, inchikeys, templates, mapped_smiles

3. **`Test{Adapter}Regression`** (integration tests):
   - Verifies specific routes match exact expected values
   - Tests specific SMILES strings, route structures, and metadata values
   - Can re-run adaptation for specific test cases
   - Examples: first route has specific reactants, multi-step route structure

#### Update Unit Test Fixtures

The `BaseAdapterTest` expects specific fixture names. Update your fixtures:

**Old:**
```python
@pytest.fixture
def target_info(self):
    return TargetInfo(id="ethanol", smiles="CCO")

@pytest.fixture
def mismatched_target_info(self):
    return TargetInfo(id="ethanol", smiles="CCC")
```

**New:**
```python
@pytest.fixture
def target_input(self):
    return TargetInfo(id="ethanol", smiles="CCO")

@pytest.fixture
def mismatched_target_input(self):
    return TargetInfo(id="ethanol", smiles="CCC")
```

#### Contract Test Example

Contract tests verify all routes meet the schema contract:

```python
@pytest.mark.integration
class TestAskcosAdapterContract:
    """Contract tests: verify the adapter produces valid Route objects with required fields populated."""

    @pytest.fixture(scope="class")
    def adapter(self) -> AskcosAdapter:
        return AskcosAdapter()

    @pytest.fixture(scope="class")
    def routes(self, adapter, raw_askcos_data, methylacetate_target_input):
        """Shared fixture to avoid re-running adaptation for every test."""
        raw_target_data = raw_askcos_data["methylacetate"]
        return list(adapter.adapt(raw_target_data, methylacetate_target_input))

    def test_produces_correct_number_of_routes(self, routes):
        """Verify the adapter produces the expected number of routes."""
        assert len(routes) == 15

    def test_all_routes_have_ranks(self, routes):
        """Verify all routes are properly ranked."""
        ranks = [route.rank for route in routes]
        assert ranks == list(range(1, len(routes) + 1))

    def test_all_routes_have_inchikeys(self, routes):
        """Verify all target molecules have InChIKeys."""
        for route in routes:
            assert route.target.inchikey is not None
            assert len(route.target.inchikey) > 0

    def test_all_reaction_steps_have_templates(self, routes):
        """Verify all reaction steps have templates populated."""
        def check_molecule(mol):
            if mol.synthesis_step is not None:
                assert mol.synthesis_step.template is not None
                assert len(mol.synthesis_step.template) > 0
                for reactant in mol.synthesis_step.reactants:
                    check_molecule(reactant)

        for route in routes:
            check_molecule(route.target)
```

#### Regression Test Example

Regression tests verify specific values and structures:

```python
@pytest.mark.integration
class TestAskcosAdapterRegression:
    """Regression tests: verify specific routes match expected structures and values."""

    @pytest.fixture(scope="class")
    def adapter(self) -> AskcosAdapter:
        return AskcosAdapter()

    @pytest.fixture(scope="class")
    def routes(self, adapter, raw_askcos_data, methylacetate_target_input):
        """Shared fixture to avoid re-running adaptation for every test."""
        raw_target_data = raw_askcos_data["methylacetate"]
        return list(adapter.adapt(raw_target_data, methylacetate_target_input))

    def test_first_route_is_simple_one_step(self, routes):
        """Verify the first route is a simple one-step synthesis."""
        route1 = routes[0]
        assert route1.rank == 1
        
        target = route1.target
        assert target.smiles == "COC(C)=O"
        assert not target.is_leaf
        assert target.synthesis_step is not None

        reaction = target.synthesis_step
        assert len(reaction.reactants) == 2

        reactant_smiles = {r.smiles for r in reaction.reactants}
        assert reactant_smiles == {"CC(=O)Cl", "CO"}
        assert all(r.is_leaf for r in reaction.reactants)

    def test_first_route_mapped_smiles(self, routes):
        """Verify the mapped SMILES for the first route matches expected value."""
        route1 = routes[0]
        reaction = route1.target.synthesis_step
        assert reaction.mapped_smiles == "Cl[C:3]([CH3:4])=[O:5].[CH3:1][OH:2]>>[CH3:1][O:2][C:3]([CH3:4])=[O:5]"
```

#### Key Testing Principles

1. **Separation of Concerns**:
   - Unit tests: adapter behavior with minimal data
   - Contract tests: all routes meet schema requirements
   - Regression tests: specific values match expectations

2. **Don't Cram Everything Into One Test**:
   - Each test should verify one specific thing
   - Use descriptive test names that explain what's being tested
   - Makes failures easier to diagnose

3. **Use Shared Fixtures for Performance**:
   - Integration tests can be slow if re-running adaptation
   - Use `scope="class"` fixtures to run adaptation once
   - Share the results across multiple tests

4. **Test Field Population, Not Just Values**:
   - Contract tests ensure fields are not None
   - Regression tests verify exact values
   - This catches both missing data and incorrect transformations

### Step 8: Update Validation Logic

**Old:**
```python
# SMILES mismatch check
if parsed_target_smiles != target_info.smiles:
    raise AdapterLogicError(f"Mismatched SMILES...")
```

**New:**
```python
# SMILES mismatch check (same as before)
if parsed_target_smiles != target_info.smiles:
    raise AdapterLogicError(f"Mismatched SMILES...")
```

The validation logic remains largely the same, but you should add InChIKey validation where appropriate.

## Common Helper Functions to Update

### If Using `common.py` Helpers

The `common.py` module currently has helpers for the old schema:
- `build_tree_from_precursor_map` → Builds `MoleculeNode`
- `build_tree_from_bipartite_node` → Builds `MoleculeNode`

You have two options:
1. **Create new helpers** with similar names (e.g., `build_molecule_from_precursor_map`)
2. **Implement the logic directly** in each adapter (recommended initially for clarity)

For the migration, implementing the logic directly in each adapter is recommended until patterns emerge.

## Checklist for Each Adapter

### Adapter Code
- [ ] Update imports (remove old schema, add new schema)
- [ ] Import `get_inchi_key` from `retrocast.domain.chem`
- [ ] Change return type from `Generator[BenchmarkTree, ...]` to `Generator[Route, ...]`
- [ ] Create new recursive builder that generates `Molecule` objects
- [ ] Ensure every `Molecule` has both `smiles` and `inchikey`
- [ ] Use `synthesis_step` instead of `reactions` list
- [ ] Remove all `id` and path-based tracking
- [ ] Add `rank` to each yielded `Route`
- [ ] Populate `metadata` dictionaries where model-specific data exists
- [ ] Extract `template` field if available in raw data
- [ ] Extract `mapped_smiles` field if available in raw data

### Tests
- [ ] Update fixture names: `target_info` → `target_input`, `mismatched_target_info` → `mismatched_target_input`
- [ ] Ensure unit tests inherit from `BaseAdapterTest`
- [ ] Create `Test{Adapter}Contract` class for schema validation tests
- [ ] Create `Test{Adapter}Regression` class for exact value tests
- [ ] Add contract test: verify all routes have ranks
- [ ] Add contract test: verify all routes have inchikeys
- [ ] Add contract test: verify all non-leaf molecules have synthesis_steps
- [ ] Add contract test: verify all reaction steps have templates (if applicable)
- [ ] Add contract test: verify all reaction steps have mapped_smiles (if applicable)
- [ ] Add regression tests for specific route structures
- [ ] Use shared fixtures with `scope="class"` for integration tests
- [ ] Run tests and ensure they pass
- [ ] Check that InChIKeys are being generated correctly

## Example: RetroStar Adapter Migration (Reference)

See the `retrostar_adapter.py` for a complete reference implementation after migration.

### Key Changes in RetroStar:
1. `_build_molecule_from_precursor_map` replaces direct call to `build_tree_from_precursor_map`
2. Each molecule includes `inchikey` via `get_inchi_key(smiles)`
3. `synthesis_step` replaces `reactions` list (always single item or None)
4. No more `id` fields or `molecule_hash` tracking
5. Return `Route(target=..., rank=1)` instead of `BenchmarkTree`

## Notes

- **InChIKey Generation**: The `get_inchi_key()` function may be slower than hashing, but provides canonical identification across different SMILES representations.
- **Metadata Flexibility**: Use metadata dictionaries generously to preserve model-specific information that doesn't fit standard fields.
- **Backward Compatibility**: The `TargetInfo` input to `adapt()` is kept for now but may be deprecated once all consumers are updated.
- **Error Handling**: Exception handling logic should remain the same; just update the object types being constructed.
