import statistics
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ursa.exceptions import SchemaLogicError
from ursa.typing import ReactionSmilesStr, SmilesStr


class RunStatistics(BaseModel):
    """A Pydantic model to hold and calculate statistics for a processing run."""

    total_routes_in_raw_files: int = 0
    routes_failed_validation: int = 0
    routes_failed_transformation: int = 0
    successful_routes_before_dedup: int = 0
    final_unique_routes_saved: int = 0
    targets_with_at_least_one_route: set[str] = Field(default_factory=set)
    routes_per_target: dict[str, int] = Field(default_factory=dict)

    @property
    def total_failures(self) -> int:
        """Total number of routes that were discarded for any reason."""
        return self.routes_failed_validation + self.routes_failed_transformation

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


# -------------------------------------------------------------------
#  Output Schemas (Defines our final, canonical benchmark format)
# -------------------------------------------------------------------


class ReactionNode(BaseModel):
    """
    Represents a single retrosynthetic reaction step in the benchmark tree.
    """

    id: str = Field(..., description="A unique, path-dependent identifier for the reaction.")
    reaction_smiles: ReactionSmilesStr
    reactants: list["MoleculeNode"] = Field(default_factory=list)


class MoleculeNode(BaseModel):
    """
    Represents a single molecule node in the benchmark tree.

    This is the core recursive data structure for the retrosynthetic route.
    It contains the molecule's identity and the reaction(s) that form it.
    """

    id: str = Field(..., description="A unique, path-dependent identifier for this molecule instance.")
    molecule_hash: str = Field(
        ..., description="A content-based hash of the canonical SMILES, identical for identical molecules."
    )
    smiles: SmilesStr
    is_starting_material: bool
    reactions: list[ReactionNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_tree_logic(self) -> "MoleculeNode":
        """
        Enforces the logical consistency of a node in a retrosynthetic tree.
        """
        num_reactions = len(self.reactions)

        # Rule 1: A starting material cannot be the product of a reaction.
        if self.is_starting_material and num_reactions > 0:
            raise SchemaLogicError(
                f"Node {self.id} ({self.smiles}) is a starting material but has {num_reactions} parent reactions."
            )

        # Rule 2: An intermediate must be the product of exactly one reaction in a valid tree.
        if not self.is_starting_material:
            if num_reactions == 0:
                raise SchemaLogicError(f"Node {self.id} ({self.smiles}) is an intermediate but has no parent reaction.")
            if num_reactions > 1:
                raise SchemaLogicError(
                    f"Node {self.id} ({self.smiles}) is part of a DAG (has {num_reactions} reactions), not a tree."
                )
        return self

    def to_simple_tree(self) -> dict[str, Any]:
        """
        recursively converts the node and its descendants into a simple,
        nested dictionary suitable for web visualization.
        """
        children = []
        if self.reactions:
            # assuming one reaction per node as per schema validation
            for reactant_node in self.reactions[0].reactants:
                children.append(reactant_node.to_simple_tree())

        return {"smiles": self.smiles, "children": children}

    def get_depth(self) -> int:
        """
        Recursively calculates the depth of the tree rooted at this node.

        Returns:
            int: 0 for starting materials (leaf nodes), or 1 + max depth of children for intermediates
        """
        if self.is_starting_material:
            return 0

        # For intermediates, we should have exactly one reaction (enforced by check_tree_logic)
        if not self.reactions:
            return 0

        # Get the maximum depth from all reactants
        max_child_depth = 0
        for reactant in self.reactions[0].reactants:
            child_depth = reactant.get_depth()
            max_child_depth = max(max_child_depth, child_depth)

        # Return 1 (this step) + the maximum depth of any child
        return 1 + max_child_depth


class TargetInfo(BaseModel):
    """A simple container for the target molecule's identity."""

    smiles: SmilesStr
    id: str = Field(..., description="The original identifier for the target molecule.")


class BenchmarkTree(BaseModel):
    """
    The root schema for a single, complete retrosynthesis benchmark entry.
    """

    target: TargetInfo
    retrosynthetic_tree: MoleculeNode

    def to_simple_tree(self) -> dict[str, Any]:
        """
        converts the entire retrosynthetic tree into a simple, nested
        dictionary suitable for web visualization.
        """
        return self.retrosynthetic_tree.to_simple_tree()
