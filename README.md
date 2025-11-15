# RetroCast: the canonical data model for retrosynthesis

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![coverage](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/ischemist/project-procrustes/master/coverage.json&query=$.totals.percent_covered_display&label=coverage&color=brightgreen&suffix=%25)

a framework for ingesting, validating, canonicalizing, and adapting retrosynthesis model outputs to the unified retrocast standard.

## how to run

the pipeline is a two-stage process:

1.  **generate raw outputs**: run model-specific scripts to perform retrosynthesis and generate the raw output files.
2.  **process with retrocast**: run the main `retrocast` pipeline to convert the raw outputs into the canonical benchmark format.

### stage 1: generate raw outputs

all scripts for generating raw model outputs live in the `scripts/` directory, organized by model name. you need to run these first.

the general pattern is:

1.  navigate to the subdirectory for the model you care about (e.g., `scripts/AiZynthFinder/`).
2.  run the numbered scripts in order. these will download assets, prepare data, and finally run the model inference.
3.  these scripts will save their raw output (usually a `results.json.gz` or similar) to the `data/evaluations/<model-name>/<dataset-name>/` directory.

**example: running `aizynthfinder-mcts` for the `uspto-190` dataset**

```bash
# 1. download model assets (only need to do this once)
uv run scripts/AiZynthFinder/1-download-assets.py data/models/aizynthfinder

# 2. prepare the building block stock file (only need to do this once)
uv run --extra aizyn scripts/AiZynthFinder/2-prepare-stock.py \
    --files data/models/assets/retrocast-bb-stock-v3-canon.csv \
    --source plain \
    --output data/models/assets/retrocast-bb-stock-v3.hdf5 \
    --target hdf5

# 3. run the actual predictions
# this will generate `data/evaluations/aizynthfinder-mcts/uspto-190/results.json.gz`
uv run --extra aizyn scripts/AiZynthFinder/3-run-aizyn-mcts.py --target-name "uspto-190"
```

**note**: each python script contains a module-level docstring that describes its purpose and shows example usage.

### stage 2: process with retrocast

once you have the raw output file from stage 1, you can run the main processing pipeline.

```bash
# process the raw output you just generated
uv run scripts/process-predictions.py process --model aizynthfinder-mcts --dataset uspto-190
```

this command will:
1.  read `retrocast-config.yaml` to find the configuration for `aizynthfinder-mcts`.
2.  load the raw results from `data/evaluations/aizynthfinder-mcts/uspto-190/results.json.gz`.
3.  use the specified `aizynth` adapter to transform the data.
4.  perform deduplication and any other processing steps.
5.  save the final, canonical output and a manifest file to `data/processed/uspto-190/retrocast-model-..../`.

### stage 3 (optional): verify integrity

you can verify that the processing is reproducible by re-calculating the source hash from the original raw files.

```bash
uv run scripts/verify-hash.py --model aizynthfinder-mcts --dataset uspto-190
```

if the hashes match, the process was successful and deterministic.

## architectural principles

`retrocast`'s design is guided by three principles to ensure it is robust, flexible, and maintainable.

1.  **adapters are the air gap**: the core system is agnostic to model output formats. all model-specific parsing and transformation logic is encapsulated in pluggable "adapters". this insulates the core pipeline from the chaos of bespoke external data formats and allows the benchmark to support any model without changing validated logic.
2.  **contracts, not handshakes**: data is validated at every boundary. pydantic schemas are the law, defining the expected structure of the final canonical format and, within each adapter, the raw input. the system refuses to operate on ambiguous or invalid data.
3.  **deterministic & auditable**: every transformation is deterministic and traceable. run outputs are uniquely identified by a cryptographic hash of their inputs (model name + raw file contents), and the entire process is logged in a manifest. this ensures results are reproducible and verifiable.

## processing pipeline

the main `retrocast.core.process_model_run` function orchestrates the workflow:

`load raw data` -> `invoke adapter` -> `transform to Route` -> `deduplicate routes` -> `serialize results & manifest`

## adding a new model adapter

the adapter is the bridge from a model's unique output to `retrocast`'s canonical format. the adapter is responsible for *all* parsing and reconstruction.

most model outputs fall into one of a few common patterns. identify the pattern, use the appropriate common builder, and your adapter will be trivial.

#### pattern a: bipartite graph (e.g., aizynthfinder, synplanner)

if the raw output is a json tree where molecule nodes point to reaction nodes and vice-versa, your job is easy.

1.  **define input schemas**: create pydantic models to validate the raw json.
2.  **implement a recursive builder**: create a `_build_molecule_from_bipartite_node` method that converts the bipartite structure into the `Route` schema.

```python
# in retrocast/adapters/bipartite_model_adapter.py
from typing import Annotated, Any, Literal, Generator
from pydantic import BaseModel, Field, RootModel, ValidationError
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.schemas import Route, Molecule, ReactionStep
from retrocast.domain.chem import get_inchi_key, canonicalize_smiles
# ... other imports

# --- pydantic schemas for raw input validation ---
class BipartiteBaseNode(BaseModel):
    smiles: str
    children: list["BipartiteNode"] = Field(default_factory=list)

class BipartiteMoleculeInput(BipartiteBaseNode):
    type: Literal["mol"]
    in_stock: bool

class BipartiteReactionInput(BipartiteBaseNode):
    type: Literal["reaction"]

BipartiteNode = Annotated[BipartiteMoleculeInput | BipartiteReactionInput, Field(discriminator="type")]
class BipartiteRouteList(RootModel[list[BipartiteMoleculeInput]]):
    pass

class BipartiteModelAdapter(BaseAdapter):
    def _build_molecule_from_bipartite_node(self, raw_mol_node: BipartiteMoleculeInput) -> Molecule:
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
            metadata={}
        )
        
        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata={}
        )

    def adapt(self, raw_data: Any, target_info: TargetInput) -> Generator[Route, None, None]:
        validated_routes = BipartiteRouteList.model_validate(raw_data)
        for rank, root_node in enumerate(validated_routes.root, start=1):
            try:
                target_molecule = self._build_molecule_from_bipartite_node(root_node)
                if target_molecule.smiles != target_info.smiles:
                    raise AdapterLogicError(f"Mismatched SMILES for target {target_info.id}")
                yield Route(target=target_molecule, rank=rank)
            except RetroCastException as e:
                logger.warning(f"route for '{target_info.id}' failed: {e}")
```

#### pattern b: precursor map (e.g., retrostar, dreamretro)

if the raw output is a string or list of reactions that can be parsed into a `dict[product_smiles, list[reactant_smiles]]`, use this pattern.

1.  **parse raw data**: write a model-specific parser that converts the raw format into a precursor map.
2.  **implement a recursive builder**: create a `_build_molecule_from_precursor_map` method that converts the map into a `Route` schema.

```python
# in retrocast/adapters/precursor_model_adapter.py
from typing import Any, Generator
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.schemas import Route, Molecule, ReactionStep
from retrocast.domain.chem import get_inchi_key
from retrocast.typing import SmilesStr
# ... other imports ...

class PrecursorModelAdapter(BaseAdapter):
    def _parse_route_string(self, route_str: str) -> dict[SmilesStr, list[SmilesStr]]:
        # model-specific logic to parse the string "p1>>r1.r2|p2>>r3..."
        precursor_map: dict[SmilesStr, list[SmilesStr]] = {}
        # ... your parsing logic here ...
        return precursor_map

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
            metadata={}
        )
        
        return Molecule(
            smiles=smiles,
            inchikey=get_inchi_key(smiles),
            synthesis_step=synthesis_step,
            metadata={}
        )

    def adapt(self, raw_data: Any, target_info: TargetInput) -> Generator[Route, None, None]:
        try:
            precursor_map = self._parse_route_string(raw_data["routes"])
            target_molecule = self._build_molecule_from_precursor_map(target_info.smiles, precursor_map)
            yield Route(target=target_molecule, rank=1)
        except RetroCastException as e:
            logger.warning(f"route for '{target_info.id}' failed: {e}")
```

#### pattern c: custom recursive (e.g., dms)

if the raw output is already a recursive tree but with a different schema, you'll need a custom recursive builder.

1.  **define input schemas**: create pydantic models for the raw tree structure.
2.  **write a recursive builder**: create a private `_build_molecule` method that traverses the raw tree and constructs the canonical `Molecule` tree.

```python
# in retrocast/adapters/custom_model_adapter.py
from typing import Any, Generator
from pydantic import BaseModel, RootModel
from retrocast.adapters.base_adapter import BaseAdapter
from retrocast.schemas import Route, Molecule, ReactionStep
from retrocast.domain.chem import canonicalize_smiles, get_inchi_key
# ... other imports

# --- pydantic schemas for raw input validation ---
class CustomTree(BaseModel):
    smiles: str
    children: list["CustomTree"] = Field(default_factory=list)

class CustomRouteList(RootModel[list[CustomTree]]):
    pass

class CustomModelAdapter(BaseAdapter):
    def _build_molecule(self, custom_node: CustomTree) -> Molecule:
        """Recursively convert a custom tree node to a retrocast Molecule."""
        canon_smiles = canonicalize_smiles(custom_node.smiles)
        
        if not custom_node.children:
            # Leaf node
            return Molecule(
                smiles=canon_smiles,
                inchikey=get_inchi_key(canon_smiles),
                synthesis_step=None,
                metadata={}
            )
        
        # Build reactants recursively
        reactants = [self._build_molecule(child) for child in custom_node.children]
        
        synthesis_step = ReactionStep(
            reactants=reactants,
            metadata={}
        )
        
        return Molecule(
            smiles=canon_smiles,
            inchikey=get_inchi_key(canon_smiles),
            synthesis_step=synthesis_step,
            metadata={}
        )

    def adapt(self, raw_data: Any, target_info: TargetInput) -> Generator[Route, None, None]:
        validated_routes = CustomRouteList.model_validate(raw_data)
        for rank, root_node in enumerate(validated_routes.root, start=1):
            target_molecule = self._build_molecule(root_node)
            yield Route(target=target_molecule, rank=rank)
```


### final steps (for all patterns)

once your adapter class is implemented:

1.  **write tests**: create `tests/adapters/test_new_adapter.py` with unit and integration tests. organize tests into three classes:
    - `Test{Adapter}Unit`: inherits from `BaseAdapterTest`, provides standard unit tests
    - `Test{Adapter}Contract`: integration tests verifying valid `Route` objects with required fields
    - `Test{Adapter}Regression`: integration tests verifying specific route structures and values

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
        def raw_unsuccessful_run_data(self) -> Any: ...

        @pytest.fixture
        def raw_invalid_schema_data(self) -> Any: ...

        @pytest.fixture
        def target_input(self) -> TargetInput: ...

        @pytest.fixture
        def mismatched_target_input(self) -> TargetInput: ...
    
    @pytest.mark.integration
    class TestNewModelAdapterContract:
        """Contract tests: verify all routes have required fields populated."""
        
        @pytest.fixture(scope="class")
        def adapter(self) -> NewModelAdapter:
            return NewModelAdapter()
        
        @pytest.fixture(scope="class")
        def routes(self, adapter, raw_data, target_input):
            return list(adapter.adapt(raw_data, target_input))
        
        def test_all_routes_have_ranks(self, routes):
            ranks = [route.rank for route in routes]
            assert ranks == list(range(1, len(routes) + 1))
        
        def test_all_routes_have_inchikeys(self, routes):
            for route in routes:
                assert route.target.inchikey is not None
    
    @pytest.mark.integration
    class TestNewModelAdapterRegression:
        """Regression tests: verify specific route structures match expectations."""
        
        @pytest.fixture(scope="class")
        def adapter(self) -> NewModelAdapter:
            return NewModelAdapter()
        
        @pytest.fixture(scope="class")
        def routes(self, adapter, raw_data, target_input):
            return list(adapter.adapt(raw_data, target_input))
        
        def test_first_route_structure(self, routes):
            route = routes[0]
            assert route.rank == 1
            # Add specific assertions about route structure
    ```

2.  **register adapter**: add your new adapter to the map in `retrocast/adapters/factory.py`.

    ```python
    # in retrocast/adapters/factory.py
    from retrocast.adapters.new_model_adapter import NewModelAdapter

    ADAPTER_MAP: dict[str, BaseAdapter] = {
        "aizynth": AizynthAdapter(),
        # ...
        "new-model": NewModelAdapter(), # <-- ADD THIS
    }
    ```

3.  **update config**: add an entry for your model in `retrocast-config.yaml`. this tells the main pipeline runner how to process it.

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
