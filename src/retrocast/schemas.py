from __future__ import annotations

from typing import Any, NewType

from pydantic import BaseModel, Field, computed_field

# Using NewTypes for semantic clarity, as you did. Good call.
SmilesStr = NewType("SmilesStr", str)
ReactionSmilesStr = NewType("ReactionSmilesStr", str)
InchiKeyStr = NewType("InchiKeyStr", str)


class TargetInput(BaseModel):
    """Input data for adapter processing. Provides target molecule identity."""

    id: str = Field(..., description="The original identifier for the target molecule.")
    smiles: SmilesStr = Field(..., description="The canonical SMILES string of the target molecule.")


class Molecule(BaseModel):
    """Represents a molecule instance within a specific route."""

    smiles: SmilesStr
    inchikey: InchiKeyStr  # The TRUE canonical identifier.

    # A molecule is formed by at most ONE reaction step in a tree.
    # If this is None, the molecule is a leaf.
    synthesis_step: ReactionStep | None = None

    # Generic bucket for model-specific data (e.g., scores, flags).
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def is_leaf(self) -> bool:
        """A molecule is a leaf if it has no reaction leading to it."""
        return self.synthesis_step is None

    def get_leaves(self) -> set[Molecule]:
        """Recursively find all leaf nodes (starting materials) from this point."""
        if self.is_leaf:
            return {self}

        leaves = set()
        # Should not be None if not a leaf, but type checker wants this
        if self.synthesis_step:
            for reactant in self.synthesis_step.reactants:
                leaves.update(reactant.get_leaves())
        return leaves

    def __hash__(self):
        # Allow Molecule objects to be added to sets based on their identity
        return hash(self.inchikey)

    def __eq__(self, other):
        return isinstance(other, Molecule) and self.inchikey == other.inchikey


class ReactionStep(BaseModel):
    """Represents a single retrosynthetic reaction step."""

    reactants: list[Molecule]

    # Explicitly add your desired optional fields
    mapped_smiles: ReactionSmilesStr | None = None
    reagents: str | None = None  # SMILES or names, e.g. "O.ClS(=O)(=O)Cl"
    solvents: str | None = None

    # Generic bucket for reaction-specific data (e.g., template scores, patent IDs).
    metadata: dict[str, Any] = Field(default_factory=dict)


class Route(BaseModel):
    """The root object for a single, complete synthesis route prediction."""

    target: Molecule
    rank: int  # The rank of this prediction (e.g., 1 for top-1)

    # This will be populated by the analysis pipeline, not the adapter.
    # It maps a building block set name to a boolean.
    solvability: dict[str, bool] = Field(default_factory=dict)

    # Metadata for the entire route
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def depth(self) -> int:
        """Calculates the depth (longest path of reactions) of the route."""

        def _get_depth(node: Molecule) -> int:
            if node.is_leaf:
                return 0
            # A non-leaf must have a synthesis_step
            assert node.synthesis_step is not None, "Non-leaf node without synthesis_step"
            return 1 + max(_get_depth(r) for r in node.synthesis_step.reactants)

        return _get_depth(self.target)

    @computed_field
    @property
    def leaves(self) -> set[Molecule]:
        """Returns the set of all unique starting materials for the route."""
        return self.target.get_leaves()

    def get_signature(self) -> str:
        """
        Generates a canonical, order-invariant hash for the entire route,
        perfect for deduplication. This is your _generate_tree_signature logic.
        """
        import hashlib

        memo = {}

        def _get_node_sig(node: Molecule) -> str:
            if node.inchikey in memo:
                return memo[node.inchikey]

            if node.is_leaf:
                return node.inchikey

            assert node.synthesis_step is not None, "Non-leaf node without synthesis_step"
            reactant_sigs = sorted([_get_node_sig(r) for r in node.synthesis_step.reactants])

            sig_str = "".join(reactant_sigs) + ">>" + node.inchikey
            sig_hash = hashlib.sha256(sig_str.encode()).hexdigest()
            memo[node.inchikey] = sig_hash
            return sig_hash

        return _get_node_sig(self.target)


# We need to tell Pydantic to rebuild the forward references
Molecule.model_rebuild()
