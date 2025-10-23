# URSA

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![lint](https://github.com/ischemist/project-procrustes/actions/workflows/lint.yml/badge.svg)

a framework for ingesting, validating, canonicalizing, and adapting retrosynthesis model outputs to the ursa benchmark standard.

## architectural principles

`ursa`'s design is guided by three principles to ensure it is robust, flexible, and maintainable.

1.  **adapters are the air gap**: the core system is agnostic to model output formats. all model-specific parsing and transformation logic is encapsulated in pluggable "adapters". this insulates the core pipeline from the chaos of bespoke external data formats and allows the benchmark to support any model without changing validated logic.
2.  **contracts, not handshakes**: data is validated at every boundary. pydantic schemas are the law, defining the expected structure of the final canonical format and, within each adapter, the raw input. the system refuses to operate on ambiguous or invalid data.
3.  **deterministic & auditable**: every transformation is deterministic and traceable. run outputs are uniquely identified by a cryptographic hash of their inputs (model name + raw file contents), and the entire process is logged in a manifest. this ensures results are reproducible and verifiable.

## processing pipeline

the main `ursa.core.process_model_run` function orchestrates the workflow:

`load raw data` -> `invoke adapter` -> `transform to BenchmarkTree` -> `deduplicate routes` -> `serialize results & manifest`

## adding a new model adapter

the adapter is the bridge from a model's unique output to `ursa`'s canonical format. the adapter is responsible for *all* parsing and reconstruction.

most model outputs fall into one of a few common patterns. identify the pattern, use the appropriate common builder, and your adapter will be trivial.

#### pattern a: bipartite graph (e.g., aizynthfinder, synplanner)

if the raw output is a json tree where molecule nodes point to reaction nodes and vice-versa, your job is easy.

1.  **define input schemas**: create pydantic models to validate the raw json.
2.  **call the common builder**: use `build_tree_from_bipartite_node`.

```python
# in ursa/adapters/bipartite_model_adapter.py
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field, RootModel, ValidationError
from ursa.adapters.base_adapter import BaseAdapter
from ursa.adapters.common import build_tree_from_bipartite_node
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
    def adapt(self, raw_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        validated_routes = BipartiteRouteList.model_validate(raw_data)
        for root_node in validated_routes.root:
            try:
                tree = build_tree_from_bipartite_node(root_node, "ursa-mol-root")
                yield BenchmarkTree(target=target_info, retrosynthetic_tree=tree)
            except UrsaException as e:
                logger.warning(f"route for '{target_info.id}' failed: {e}")
```

#### pattern b: precursor map (e.g., retrostar, dreamretro)

if the raw output is a string or list of reactions that can be parsed into a `dict[product_smiles, list[reactant_smiles]]`, use this pattern.

1.  **parse raw data**: write a model-specific parser that converts the raw format into a `PrecursorMap`.
2.  **call the common builder**: use `build_tree_from_precursor_map`.

```python
# in ursa/adapters/precursor_model_adapter.py
from ursa.adapters.base_adapter import BaseAdapter
from ursa.adapters.common import PrecursorMap, build_tree_from_precursor_map
# ... other imports ...

class PrecursorModelAdapter(BaseAdapter):
    def _parse_route_string(self, route_str: str) -> PrecursorMap:
        # model-specific logic to parse the string "p1>>r1.r2|p2>>r3..."
        precursor_map: PrecursorMap = {}
        # ... your parsing logic here ...
        return precursor_map

    def adapt(self, raw_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        try:
            precursor_map = self._parse_route_string(raw_data["routes"])
            tree = build_tree_from_precursor_map(target_info.smiles, precursor_map)
            yield BenchmarkTree(target=target_info, retrosynthetic_tree=tree)
        except UrsaException as e:
            logger.warning(f"route for '{target_info.id}' failed: {e}")
```

#### pattern c: custom recursive (e.g., dms)

if the raw output is already a recursive tree but with a different schema, you'll need a custom recursive builder.

1.  **define input schemas**: create pydantic models for the raw tree structure.
2.  **write a recursive builder**: create a private `_build_molecule_node` method that traverses the raw tree and constructs the canonical `MoleculeNode` tree.

```python
# in ursa/adapters/custom_model_adapter.py
from pydantic import BaseModel, RootModel
from ursa.adapters.base_adapter import BaseAdapter
from ursa.domain.schemas import MoleculeNode, ReactionNode
# ... other imports

# --- pydantic schemas for raw input validation ---
class CustomTree(BaseModel):
    smiles: str
    children: list["CustomTree"]

class CustomRouteList(RootModel[list[CustomTree]]):
    pass

class CustomModelAdapter(BaseAdapter):
    def _build_molecule_node(self, custom_node: CustomTree, ...) -> MoleculeNode:
        # logic to convert one custom node to one ursa node
        canon_smiles = canonicalize_smiles(custom_node.smiles)
        reactions = []
        if custom_node.children:
            reactants = [self._build_molecule_node(child) for child in custom_node.children] # recursive call
            reactions.append(ReactionNode(...))
        return MoleculeNode(smiles=canon_smiles, reactions=reactions, ...)

    def adapt(self, raw_data: Any, target_info: TargetInfo) -> Generator[BenchmarkTree, None, None]:
        validated_routes = CustomRouteList.model_validate(raw_data)
        for root_node in validated_routes.root:
            tree = self._build_molecule_node(root_node)
            yield BenchmarkTree(target=target_info, retrosynthetic_tree=tree)
```


### final steps (for all patterns)

once your adapter class is implemented:

1.  **write tests**: create `tests/adapters/test_new_adapter.py`, inherit from `BaseAdapterTest`, and provide the required fixtures. this test harness provides a standard suite of tests for free.

    ```python
    # in tests/adapters/test_new_adapter.py
    import pytest
    from tests.adapters.test_base_adapter import BaseAdapterTest
    from ursa.adapters.new_model_adapter import NewModelAdapter
    from ursa.domain.schemas import TargetInfo

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
        def target_info(self) -> TargetInfo: ...

        @pytest.fixture
        def mismatched_target_info(self) -> TargetInfo: ...
    ```

2.  **register adapter**: add your new adapter to the map in `ursa/adapters/factory.py`.

    ```python
    # in ursa/adapters/factory.py
    from ursa.adapters.new_model_adapter import NewModelAdapter

    ADAPTER_MAP: dict[str, BaseAdapter] = {
        "aizynth": AizynthAdapter(),
        # ...
        "new-model": NewModelAdapter(), # <-- ADD THIS
    }
    ```

3.  **update config**: add an entry for your model in `ursa-config.yaml`. this tells the main pipeline runner how to process it.

    ```yaml
    # in ursa-config.yaml
    models:
      # ...
      new-model:
        adapter: new-model  # must match the key from ADAPTER_MAP
        raw_results_filename: results.json.gz
        sampling:
          strategy: top-k
          k: 10
    ```
