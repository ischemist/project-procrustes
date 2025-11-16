from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field, computed_field

from retrocast.typing import InchiKeyStr, ReactionSmilesStr, SmilesStr

# Type alias for a reaction signature: (frozenset of reactant InchiKeys, product InchiKey)
ReactionSignature = tuple[frozenset[str], str]


def _get_retrocast_version() -> str:
    """Get the current retrocast version for provenance tracking."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("retrocast")
    except PackageNotFoundError:
        return "0.0.0.dev0+unknown"


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

    mapped_smiles: ReactionSmilesStr | None = None
    template: str | None = None  # Reaction template string (e.g., SMARTS pattern)
    reagents: list[SmilesStr] | None = None  # List of reagent SMILES, e.g. ["O", "ClS(=O)(=O)Cl"]
    solvents: list[SmilesStr] | None = None  # List of solvent SMILES

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

    # Version of retrocast that created this route (for provenance tracking)
    retrocast_version: str = Field(
        default_factory=_get_retrocast_version,
        description="Version of retrocast that created this route",
    )

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

    def get_content_hash(self) -> str:
        """
        Generates a deterministic hash of the complete route content.

        Unlike get_signature() which only considers tree topology (InchiKeys),
        this method includes ALL data: rank, metadata, solvability, retrocast_version,
        and all reaction details (mapped_smiles, templates, reagents, solvents, etc.).

        This is useful for verifying that two routes are semantically identical,
        including all metadata and provenance information.
        """
        import hashlib
        import json

        # Exclude computed fields to ensure deterministic serialization
        # (sets like 'leaves' have non-deterministic iteration order across processes)
        route_dict = self.model_dump(mode="json", exclude={"leaves", "depth"})
        route_json = json.dumps(route_dict, sort_keys=True)
        return hashlib.sha256(route_json.encode()).hexdigest()

    def get_reaction_signatures(self) -> set[ReactionSignature]:
        """
        Extracts all unique reaction signatures from the route.

        Each reaction is represented as a tuple of (frozenset of reactant InchiKeys, product InchiKey).
        This provides a lightweight, hashable representation for comparing reactions across routes.

        Returns:
            Set of ReactionSignature tuples, one for each unique reaction in the route.

        Example use case:
            # Check if two routes share any reactions
            route1_reactions = route1.get_reaction_signatures()
            route2_reactions = route2.get_reaction_signatures()
            overlapping = route1_reactions & route2_reactions
        """
        signatures: set[ReactionSignature] = set()

        def _collect_reactions(node: Molecule) -> None:
            if node.is_leaf:
                return

            # Non-leaf node must have a synthesis_step
            assert node.synthesis_step is not None, "Non-leaf node without synthesis_step"

            # Create signature for this reaction
            reactant_keys = frozenset(r.inchikey for r in node.synthesis_step.reactants)
            product_key = node.inchikey
            sig: ReactionSignature = (reactant_keys, product_key)
            signatures.add(sig)

            # Recursively collect from reactants
            for reactant in node.synthesis_step.reactants:
                _collect_reactions(reactant)

        _collect_reactions(self.target)
        return signatures


# We need to tell Pydantic to rebuild the forward references
Molecule.model_rebuild()


class RunStatistics(BaseModel):
    """A Pydantic model to hold and calculate statistics for a processing run."""

    total_routes_in_raw_files: int = 0
    routes_failed_transformation: int = 0  # Includes both validation and transformation failures
    successful_routes_before_dedup: int = 0
    final_unique_routes_saved: int = 0
    targets_with_at_least_one_route: set[str] = Field(default_factory=set)
    routes_per_target: dict[str, int] = Field(default_factory=dict)

    @property
    def total_failures(self) -> int:
        """Total number of routes that failed validation or transformation."""
        return self.routes_failed_transformation

    @property
    def num_targets_with_routes(self) -> int:
        """The count of unique targets that have at least one valid route."""
        return len(self.targets_with_at_least_one_route)

    @property
    def duplication_factor(self) -> float:
        """Ratio of successful routes before and after deduplication. 1.0 means no duplicates."""
        if self.final_unique_routes_saved == 0:
            return 0.0
        ratio = self.successful_routes_before_dedup / self.final_unique_routes_saved
        return round(ratio, 2)

    @property
    def min_routes_per_target(self) -> int:
        """Minimum number of routes per target that has at least one route."""
        if not self.routes_per_target:
            return 0
        return min(self.routes_per_target.values())

    @property
    def max_routes_per_target(self) -> int:
        """Maximum number of routes per target that has at least one route."""
        if not self.routes_per_target:
            return 0
        return max(self.routes_per_target.values())

    @property
    def avg_routes_per_target(self) -> float:
        """Average number of routes per target that has at least one route."""
        if not self.routes_per_target:
            return 0.0
        return round(statistics.mean(self.routes_per_target.values()), 2)

    @property
    def median_routes_per_target(self) -> float:
        """Median number of routes per target that has at least one route."""
        if not self.routes_per_target:
            return 0.0
        return round(statistics.median(self.routes_per_target.values()), 2)

    def to_manifest_dict(self) -> dict[str, int | float]:
        """Generates a dictionary suitable for including in the final manifest."""
        return {
            "total_routes_in_raw_files": self.total_routes_in_raw_files,
            "total_routes_failed_or_duplicate": self.total_failures
            + (self.successful_routes_before_dedup - self.final_unique_routes_saved),
            "final_unique_routes_saved": self.final_unique_routes_saved,
            "num_targets_with_at_least_one_route": self.num_targets_with_routes,
            "duplication_factor": self.duplication_factor,
            "min_routes_per_target": self.min_routes_per_target,
            "max_routes_per_target": self.max_routes_per_target,
            "avg_routes_per_target": self.avg_routes_per_target,
            "median_routes_per_target": self.median_routes_per_target,
        }
