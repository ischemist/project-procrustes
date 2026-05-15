from __future__ import annotations

import hashlib
import json
import statistics
import warnings
from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, computed_field

from retrocast.chem import InchiKeyLevel, reduce_inchikey
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


def _normalize_signature_value(value: Any) -> Any:
    """Normalize nested values into a deterministic, order-invariant JSON shape."""
    if isinstance(value, dict):
        return {key: _normalize_signature_value(val) for key, val in sorted(value.items())}

    if isinstance(value, (list, tuple, set, frozenset)):
        normalized_items = [_normalize_signature_value(item) for item in value]
        return sorted(normalized_items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))

    return value


class Molecule(BaseModel):
    """Represents a molecule instance within a specific route."""

    smiles: SmilesStr
    inchikey: InchiKeyStr

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

    @computed_field
    @property
    def is_convergent(self) -> bool:
        """
        A reaction is convergent if it combines two or more intermediates (non-leaf molecules).

        This identifies a point of convergence where multiple synthesized fragments
        are joined together, as opposed to linear synthesis where each step adds
        to a single growing chain.
        """
        intermediate_count = sum(1 for r in self.reactants if not r.is_leaf)
        return intermediate_count >= 2


class RouteReaction(BaseModel):
    """A reaction step paired with the route molecule it produces."""

    product: Molecule
    step: ReactionStep

    @property
    def signature(self) -> ReactionSignature:
        return (frozenset(reactant.inchikey for reactant in self.step.reactants), self.product.inchikey)


@runtime_checkable
class TargetIdentity(Protocol):
    """
    Minimal interface required by adapters.
    Read-only protocol allows covariance (SmilesStr -> str).
    """

    @property
    def id(self) -> str: ...

    @property
    def smiles(self) -> SmilesStr: ...


class TargetInput(BaseModel):
    """Lightweight DTO for ad-hoc adaptation."""

    id: str
    smiles: str


class Route(BaseModel):
    """The root object for a single, complete synthesis route prediction."""

    target: Molecule

    # Metadata for the entire route
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Version of retrocast that created this route (for provenance tracking)
    retrocast_version: str = Field(
        default_factory=_get_retrocast_version,
        description="Version of retrocast that created this route",
    )

    @computed_field
    @property
    def length(self) -> int:
        """Calculates the length (longest path of reactions) of the route."""

        def _get_length(node: Molecule) -> int:
            if node.is_leaf:
                return 0
            # A non-leaf must have a synthesis_step
            assert node.synthesis_step is not None, "Non-leaf node without synthesis_step"
            return 1 + max(_get_length(r) for r in node.synthesis_step.reactants)

        return _get_length(self.target)

    @computed_field
    @property
    def leaves(self) -> set[Molecule]:
        """Returns the set of all unique starting materials for the route."""
        return self.target.get_leaves()

    @computed_field
    @property
    def has_convergent_reaction(self) -> bool:
        """
        Returns True if the route contains at least one convergent reaction.

        A convergent reaction is one that combines two or more intermediates
        (non-leaf molecules). This property recursively traverses the synthesis
        tree to find any point of convergence.
        """

        def _check_convergent(node: Molecule) -> bool:
            if node.is_leaf:
                return False

            assert node.synthesis_step is not None, "Non-leaf node without synthesis_step"

            # Check if this reaction step is convergent
            if node.synthesis_step.is_convergent:
                return True

            # Recursively check all branches
            return any(_check_convergent(reactant) for reactant in node.synthesis_step.reactants)

        return _check_convergent(self.target)

    @computed_field
    @property
    def content_hash(self) -> str:
        """Computed content hash for deduplication."""
        return self.get_content_hash()

    @computed_field
    @property
    def signature(self) -> str:
        """Deprecated computed alias for the structural route signature."""
        return self.get_structural_signature()

    @property
    def structural_signature(self) -> str:
        """Canonical signature of the labeled route structure."""
        return self.get_structural_signature()

    def iter_steps(self) -> list[ReactionStep]:
        """Return reaction steps in deterministic root-first depth-first order."""
        steps: list[ReactionStep] = []

        def _visit(node: Molecule) -> None:
            if node.synthesis_step is None:
                return
            steps.append(node.synthesis_step)
            for reactant in node.synthesis_step.reactants:
                _visit(reactant)

        _visit(self.target)
        return steps

    def iter_reactions(self) -> Iterator[RouteReaction]:
        """Yield reaction/product pairs in deterministic root-first depth-first order."""

        def _visit(node: Molecule) -> Iterator[RouteReaction]:
            if node.synthesis_step is None:
                return
            yield RouteReaction(product=node, step=node.synthesis_step)
            for reactant in node.synthesis_step.reactants:
                yield from _visit(reactant)

        yield from _visit(self.target)

    def get_reactions(self) -> list[RouteReaction]:
        """Return reaction/product pairs in deterministic root-first depth-first order."""
        return list(self.iter_reactions())

    def _build_route_signature(
        self,
        *,
        match_level: InchiKeyLevel,
        annotation_builder: Callable[[ReactionStep], dict[str, Any]] | None = None,
    ) -> str:
        memo: dict[int, str] = {}

        def _node_key(node: Molecule) -> str:
            if match_level == InchiKeyLevel.FULL:
                return node.inchikey
            return reduce_inchikey(node.inchikey, match_level)

        def _get_node_sig(node: Molecule) -> str:
            node_id = id(node)
            if node_id in memo:
                return memo[node_id]

            key = _node_key(node)
            if node.is_leaf:
                memo[node_id] = key
                return key

            assert node.synthesis_step is not None, "Non-leaf node without synthesis_step"
            reactant_sigs = sorted(_get_node_sig(r) for r in node.synthesis_step.reactants)

            payload: dict[str, Any] = {
                "product": key,
                "reactants": reactant_sigs,
            }
            if annotation_builder is not None:
                annotations = annotation_builder(node.synthesis_step)
                if annotations:
                    payload["annotations"] = annotations

            sig_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            memo[node_id] = sig_hash
            return sig_hash

        return _get_node_sig(self.target)

    def get_structural_signature(self, match_level: InchiKeyLevel = InchiKeyLevel.FULL) -> str:
        """
        Generates a canonical, order-invariant signature for the route structure.

        Args:
            match_level: Level of InChI key matching specificity:
                - None or FULL: Exact matching (default)
                - NO_STEREO: Ignore stereochemistry
                - CONNECTIVITY: Match on molecular skeleton only
        """
        return self._build_route_signature(match_level=match_level)

    def get_signature(self, match_level: InchiKeyLevel = InchiKeyLevel.FULL) -> str:
        """Deprecated alias for get_structural_signature()."""
        warnings.warn(
            "Route.get_signature() is deprecated; use Route.get_structural_signature() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_structural_signature(match_level=match_level)

    def get_annotated_signature(
        self,
        *,
        match_level: InchiKeyLevel = InchiKeyLevel.FULL,
        include_mapped_smiles: bool = False,
        include_template: bool = False,
        include_reagents: bool = False,
        include_solvents: bool = False,
        include_step_metadata_keys: tuple[str, ...] = (),
    ) -> str:
        """
        Generates a canonical signature of the route structure plus selected reaction annotations.

        Unlike get_structural_signature(), this method can incorporate reaction-level detail
        such as mapped reaction smiles, templates, reagents, solvents, or selected metadata keys,
        while still excluding route provenance like rank or retrocast version.
        """

        def _build_annotations(step: ReactionStep) -> dict[str, Any]:
            annotations: dict[str, Any] = {}
            if include_mapped_smiles and step.mapped_smiles is not None:
                annotations["mapped_smiles"] = step.mapped_smiles
            if include_template and step.template is not None:
                annotations["template"] = step.template
            if include_reagents and step.reagents is not None:
                annotations["reagents"] = _normalize_signature_value(step.reagents)
            if include_solvents and step.solvents is not None:
                annotations["solvents"] = _normalize_signature_value(step.solvents)
            if include_step_metadata_keys:
                selected_metadata = {
                    key: _normalize_signature_value(step.metadata.get(key))
                    for key in include_step_metadata_keys
                    if key in step.metadata
                }
                if selected_metadata:
                    annotations["metadata"] = selected_metadata
            return annotations

        return self._build_route_signature(match_level=match_level, annotation_builder=_build_annotations)

    def get_content_hash(self) -> str:
        """
        Generates a deterministic hash of the complete route content.

        Unlike get_structural_signature() which only considers route structure,
        this method includes route metadata, retrocast_version, and all reaction
        details (mapped_smiles, templates, reagents, solvents, etc.).

        This is useful for verifying that two routes are semantically identical,
        including all chemistry and attached annotations.
        """

        # Exclude computed fields to ensure deterministic serialization
        # (sets like 'leaves' have non-deterministic iteration order across processes)
        # Also exclude content_hash and signature to avoid circular recursion
        route_dict = self.model_dump(mode="json", exclude={"leaves", "length", "content_hash", "signature"})
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
        return {reaction.signature for reaction in self.iter_reactions()}


# We need to tell Pydantic to rebuild the forward references
Molecule.model_rebuild()


class VendorSource(StrEnum):
    """Enumeration of buyables vendor sources."""

    MCULE = "MC"
    LABNETWORK = "LN"
    EMOLECULES = "EM"
    SIGMA_ALDRICH = "SA"
    CHEMBRIDGE = "CB"


class BuyableMolecule(BaseModel):
    """Represents a molecule available for purchase with commercial metadata."""

    smiles: SmilesStr
    inchikey: InchiKeyStr
    ppg: float | None = Field(None, description="Price per gram in USD")
    source: VendorSource | None = Field(None, description="Source vendor")
    lead_time: str | None = Field(None, description="Lead time for delivery (e.g., '7-21days', '1week')")
    link: str | None = Field(None, description="URL to vendor product page")


class StockStatistics(BaseModel):
    """Statistics for stock canonicalization and deduplication."""

    raw_input_lines: int = 0
    empty_lines: int = 0
    invalid_smiles: int = 0
    inchi_generation_failed: int = 0
    duplicate_smiles: int = 0
    duplicate_inchikeys: int = 0
    unique_molecules: int = 0

    def to_manifest_dict(self) -> dict[str, int]:
        """Generates a dictionary suitable for including in the manifest."""
        return {
            "raw_input_lines": self.raw_input_lines,
            "empty_lines": self.empty_lines,
            "invalid_smiles": self.invalid_smiles,
            "inchi_generation_failed": self.inchi_generation_failed,
            "duplicate_smiles": self.duplicate_smiles,
            "duplicate_inchikeys": self.duplicate_inchikeys,
            "unique_molecules": self.unique_molecules,
            "total_filtered": self.empty_lines
            + self.invalid_smiles
            + self.inchi_generation_failed
            + self.duplicate_smiles
            + self.duplicate_inchikeys,
        }


class RunStatistics(BaseModel):
    """A Pydantic model to hold and calculate statistics for a processing run."""

    total_routes_in_raw_files: int = 0  # Raw route-like entries visited during adaptation.
    routes_failed_transformation: int = 0  # Includes both validation and transformation failures
    successful_routes_before_dedup: int = 0
    final_unique_routes_saved: int = 0
    targets_with_at_least_one_route: set[str] = Field(default_factory=set)
    routes_per_target: dict[str, int] = Field(default_factory=dict)
    failures_by_code: dict[str, int] = Field(default_factory=dict)
    failures_by_target: dict[str, dict[str, int]] = Field(default_factory=dict)

    def record_failure(self, code: str, *, target_id: str | None = None) -> None:
        """Record a workflow-classified target failure."""
        self.routes_failed_transformation += 1
        self.failures_by_code[code] = self.failures_by_code.get(code, 0) + 1
        if target_id is not None:
            target_failures = self.failures_by_target.setdefault(target_id, {})
            target_failures[code] = target_failures.get(code, 0) + 1

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

    def to_manifest_dict(self) -> dict[str, int | float | dict[str, int]]:
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
            "failures_by_code": dict(sorted(self.failures_by_code.items())),
        }
