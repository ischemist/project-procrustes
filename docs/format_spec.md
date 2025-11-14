# `retrocast` Unified Route Specification

This document specifies the canonical Python object model for representing a single retrosynthetic route. The object model is defined using `pydantic` for validation and type safety.

The primary goal is to provide a semantically rich, computationally useful, and standardized representation that can be produced by various model-specific "adapters". All downstream analysis, storage, and visualization operate on these objects.

## Core Principles

1.  **Semantic Naming**: Classes and fields are named according to their chemical meaning (e.g., `Molecule`, `ReactionStep`) rather than abstract graph theory terms.
2.  **Decoupling Prediction from Evaluation**: The model's output (the route structure) is kept separate from evaluation metrics (e.g., solvability). The model predicts a tree; the analysis pipeline determines if its leaves are in stock.
3.  **Canonical Identification**: Molecules are primarily identified by their IUPAC standard InChIKey for robust, version-independent hashing and comparison. SMILES strings are used for display and model compatibility.
4.  **Tree Enforcement**: The data structure enforces that a route is a tree (not a DAG). An intermediate molecule is the product of exactly one reaction.

## Object Model Schema

The object model consists of three main `pydantic` classes: `Route`, `Molecule`, and `ReactionStep`.

### 1. `Route`

The top-level container for a single, complete retrosynthetic pathway prediction.

-   **`target: Molecule`**: The root `Molecule` object of the synthesis tree.
-   **`rank: int`**: The rank of this route in the model's list of predictions (e.g., 1 for top-1, 2 for top-2).
-   **`solvability: dict[str, bool]`**: (Computed Field) A dictionary mapping the name of a `BuildingBlockSet` to a boolean indicating whether the route is solvable against that set. This field is populated by the analysis pipeline, not the adapter.
-   **`metadata: dict[str, Any]`**: An open dictionary for storing any metadata relevant to the entire route (e.g., model-specific global scores, runtime for this route).
-   **`@computed_field leaves: set[Molecule]`**: A property that returns the set of all unique starting materials (leaf molecules) for the route.
-   **`@computed_field depth: int`**: A property that calculates the depth of the route, defined as the number of reactions in the longest path from the target to a leaf.

### 2. `Molecule`

Represents a specific molecule within the context of a route.

-   **`smiles: SmilesStr`**: The canonical SMILES string of the molecule.
-   **`inchikey: InchiKeyStr`**: The IUPAC standard InChIKey. **This is the primary identifier for hashing and equality checks.**
-   **`synthesis_step: Optional[ReactionStep]`**: The `ReactionStep` that produces this molecule. If `None`, this molecule is a leaf node (a starting material).
-   **`metadata: dict[str, Any]`**: An open dictionary for molecule-specific metadata, such as a node score or a "purchasability" flag from the original model output.
-   **`@computed_field is_leaf: bool`**: A boolean property that is `True` if `synthesis_step` is `None`.

### 3. `ReactionStep`

Represents a single retrosynthetic reaction.

-   **`reactants: list[Molecule]`**: A list of `Molecule` objects that are the precursors in this reaction.
-   **`mapped_smiles: Optional[ReactionSmilesStr]`**: (Optional) The atom-mapped reaction SMILES string, if provided by the model.
-   **`reagents: Optional[str]`**: (Optional) A SMILES string representing reagents used in the forward reaction (e.g., `"O.ClS(=O)(=O)Cl"`).
-   **`solvents: Optional[str]`**: (Optional) A SMILES string or name representing solvents.
-   **`metadata: dict[str, Any]`**: An open dictionary for reaction-specific metadata, such as a reaction probability, template ID, or patent reference.

## Example Usage (Conceptual)

Below is a conceptual Python representation of a simple route for benzene from acetylene.

```python
# Conceptual representation; not literal instantiation code.

# This route represents: HC#CH + HC#CH + HC#CH -> c1ccccc1
route_example = Route(
    rank=1,
    target=Molecule(
        smiles="c1ccccc1",
        inchikey="UHOVQNZJYSORNB-UHFFFAOYSA-N",
        synthesis_step=ReactionStep(
            reactants=[
                Molecule(smiles="C#C", inchikey="HSMOJBHZKKGOKP-UHFFFAOYSA-N"), # is_leaf = True
                Molecule(smiles="C#C", inchikey="HSMOJBHZKKGOKP-UHFFFAOYSA-N"), # is_leaf = True
                Molecule(smiles="C#C", inchikey="HSMOJBHZKKGOKP-UHFFFAOYSA-N"), # is_leaf = True
            ],
            metadata={"template_score": 0.95}
        )
    )
)

# After analysis:
# route_example.solvability["enamine_real"] will be True or False.
# route_example.depth will be 1.
# route_example.leaves will be a set containing one Molecule object for acetylene.
```

## Adapter Responsibilities

An adapter's job is to parse a model's raw output and instantiate a list of these `Route` objects for each target molecule.

-   Adapters **MUST** generate both `smiles` and `inchikey` for every `Molecule`.
-   Adapters **MUST** correctly construct the recursive `Molecule` -> `ReactionStep` -> `Molecule` structure.
-   Adapters **SHOULD** populate the `metadata` fields with any relevant data from the source format.
-   Adapters **SHOULD NOT** populate the `solvability` field; this is the responsibility of the downstream analysis pipeline.
-   Adapters **MUST** ensure the `target` molecule of the final `Route` object matches the expected target for the prediction.
