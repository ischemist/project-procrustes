from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import BenchmarkValidationError
from retrocast.metrics.solvability import is_route_solved
from retrocast.models.chem import Route
from retrocast.typing import InchiKeyStr, SmilesStr


class ExecutionStats(BaseModel):
    wall_time: dict[str, float] = Field(default_factory=dict)
    cpu_time: dict[str, float] = Field(default_factory=dict)


class BenchmarkTarget(BaseModel):
    """
    Atomic unit of a benchmark.
    Represents a specific retrosynthesis problem: "Make this molecule."
    """

    id: str = Field(..., description="Unique identifier within the benchmark (e.g., 'n5-00123').")
    smiles: SmilesStr = Field(..., description="The canonical SMILES of the target.")
    inchi_key: InchiKeyStr = Field(..., description="The InChIKey of the target molecule.")

    # Bucket for anything else (e.g. "source_patent_id", "reaction_classes", "original_index")
    metadata: dict[str, Any] = Field(default_factory=dict)

    # List of acceptable routes for this target (first = primary route)
    # Empty list = pure prediction task with no reference routes
    acceptable_routes: list[Route] = Field(default_factory=list)

    @property
    def primary_route(self) -> Route | None:
        """Returns the primary (first) acceptable route, if any."""
        return self.acceptable_routes[0] if self.acceptable_routes else None

    @property
    def route_length(self) -> int | None:
        """Length of the primary acceptable route (used for filtering/subset creation)."""
        return self.primary_route.length if self.primary_route else None

    @property
    def is_convergent(self) -> bool | None:
        """Whether the primary acceptable route is convergent (used for filtering)."""
        if not self.primary_route:
            return None
        return self.primary_route.has_convergent_reaction


class BenchmarkSet(BaseModel):
    """
    The container for an evaluation set.
    This object defines the 'exam' that models will take.
    """

    name: str = Field(..., description="Unique name of this benchmark set (e.g., 'stratified-linear-600').")
    description: str = Field(default="", description="Human-readable description of provenance.")

    # The stock definition is part of the benchmark contract.
    # We store the name, not the path, to keep this portable across machines.
    stock_name: str | None = Field(default=None, description="Name of the stock file required for this benchmark.")

    # The core data: Map of ID -> Target.
    # Using a dict enforces ID uniqueness automatically.
    targets: dict[str, BenchmarkTarget] = Field(default_factory=dict)

    def get_smiles_map(self) -> dict[str, str]:
        """
        Returns a mapping of {smiles: target_id}.

        Crucial for mapping model predictions (which are keyed by SMILES)
        to benchmark targets (which are keyed by ID). Each SMILES maps to
        exactly one target ID (enforced at construction time).
        """
        return {target.smiles: target.id for target in self.targets.values()}

    def get_inchikey_map(self) -> dict[str, str]:
        """
        Returns a mapping of {inchi_key: target_id}.

        Similar to get_smiles_map but uses InChIKeys as keys. Useful for
        canonical molecule identity lookups. Each InChIKey maps to exactly
        one target ID (enforced at construction time).
        """
        return {target.inchi_key: target.id for target in self.targets.values()}

    def get_target_ids(self) -> list[str]:
        """Returns a sorted list of all target IDs."""
        return sorted(self.targets.keys())

    def subset(self, ids: list[str], new_name_suffix: str) -> BenchmarkSet:
        """
        Creates a new BenchmarkSet containing only the specified IDs.
        """
        missing = [i for i in ids if i not in self.targets]
        if missing:
            raise ValueError(f"IDs not found in parent set: {missing[:3]}...")

        return BenchmarkSet(
            name=f"{self.name}-{new_name_suffix}",
            description=f"Subset of {self.name}",
            stock_name=self.stock_name,
            targets={i: self.targets[i] for i in ids},
        )


def validate_benchmark_targets(targets: dict[str, BenchmarkTarget]) -> None:
    """
    Validates that benchmark targets have unique SMILES and InChIKeys.

    Args:
        targets: Dictionary mapping target IDs to BenchmarkTarget objects

    Raises:
        BenchmarkValidationError: If duplicate SMILES or InChIKeys are found
    """
    smiles_to_ids: dict[str, list[str]] = {}
    inchikey_to_ids: dict[str, list[str]] = {}

    for target_id, target in targets.items():
        # Track SMILES duplicates
        if target.smiles not in smiles_to_ids:
            smiles_to_ids[target.smiles] = []
        smiles_to_ids[target.smiles].append(target_id)

        # Track InChIKey duplicates
        if target.inchi_key not in inchikey_to_ids:
            inchikey_to_ids[target.inchi_key] = []
        inchikey_to_ids[target.inchi_key].append(target_id)

    # Check for duplicates
    smiles_duplicates = {smiles: ids for smiles, ids in smiles_to_ids.items() if len(ids) > 1}
    inchikey_duplicates = {inchi: ids for inchi, ids in inchikey_to_ids.items() if len(ids) > 1}

    errors = []
    if smiles_duplicates:
        for smiles, ids in list(smiles_duplicates.items())[:3]:  # Show first 3
            errors.append(f"SMILES '{smiles}' appears in targets: {ids}")
        if len(smiles_duplicates) > 3:
            errors.append(f"... and {len(smiles_duplicates) - 3} more SMILES duplicates")

    if inchikey_duplicates:
        for inchi, ids in list(inchikey_duplicates.items())[:3]:  # Show first 3
            errors.append(f"InChIKey '{inchi}' appears in targets: {ids}")
        if len(inchikey_duplicates) > 3:
            errors.append(f"... and {len(inchikey_duplicates) - 3} more InChIKey duplicates")

    if errors:
        raise BenchmarkValidationError(
            "Benchmark contains duplicate molecules:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def validate_acceptable_routes_solvable(benchmark: BenchmarkSet, stock: set[InchiKeyStr]) -> None:
    """
    Validates that all acceptable routes are solvable with the given stock.

    This validation should be run after benchmark creation/subset construction
    to ensure all acceptable routes can actually be solved with the benchmark's stock.

    Args:
        benchmark: The benchmark to validate
        stock: Set of InChIKeys representing available stock molecules

    Raises:
        BenchmarkValidationError: If any acceptable route is not solvable
    """
    errors = []

    for target_id, target in benchmark.targets.items():
        for idx, route in enumerate(target.acceptable_routes):
            if not is_route_solved(route, stock):
                missing_leaves = [leaf.inchikey for leaf in route.leaves if leaf.inchikey not in stock]
                errors.append(
                    f"Target '{target_id}' acceptable_routes[{idx}]: "
                    f"{len(missing_leaves)}/{len(route.leaves)} leaves missing from stock"
                )

    if errors:
        error_summary = errors[:10]  # Show first 10
        if len(errors) > 10:
            error_summary.append(f"... and {len(errors) - 10} more unsolvable routes")

        raise BenchmarkValidationError(
            f"Found {len(errors)} unsolvable acceptable routes:\n" + "\n".join(f"  - {e}" for e in error_summary)
        )


def create_benchmark_target(
    id: str,
    smiles: str,
    acceptable_routes: list[Route] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkTarget:
    """
    Creates a BenchmarkTarget with canonicalized SMILES and computed InChIKey.

    This is the official constructor for creating new benchmark targets.
    It ensures SMILES are canonicalized and InChIKeys are computed consistently.

    Args:
        id: Unique identifier for the target
        smiles: SMILES string (will be canonicalized)
        acceptable_routes: List of acceptable routes (first is primary)
        metadata: Additional metadata

    Returns:
        A validated BenchmarkTarget with canonicalized SMILES and computed InChIKey

    Raises:
        InvalidSmilesError: If the SMILES string is invalid
    """
    canonical_smiles = canonicalize_smiles(smiles)
    inchi_key = get_inchi_key(canonical_smiles)

    return BenchmarkTarget(
        id=id,
        smiles=canonical_smiles,
        inchi_key=inchi_key,
        acceptable_routes=acceptable_routes or [],
        metadata=metadata or {},
    )


def create_benchmark(
    name: str,
    targets: dict[str, BenchmarkTarget],
    stock: set[InchiKeyStr],
    description: str = "",
    stock_name: str | None = None,
) -> BenchmarkSet:
    """
    Creates a BenchmarkSet with validation for unique SMILES and InChIKeys.

    This is the official constructor for creating new benchmarks.
    It validates that:
    1. All targets have unique SMILES and InChIKeys
    2. All acceptable routes are solvable with the provided stock

    Args:
        name: Name of the benchmark
        targets: Dictionary mapping target IDs to BenchmarkTarget objects
        stock: Set of InChIKeys representing available stock molecules
        description: Human-readable description
        stock_name: Name of the stock file for this benchmark

    Returns:
        A validated BenchmarkSet

    Raises:
        BenchmarkValidationError: If validation fails (duplicates or unsolvable routes)
    """
    # Validate unique SMILES and InChIKeys
    validate_benchmark_targets(targets)

    # Create the benchmark
    benchmark = BenchmarkSet(
        name=name,
        description=description,
        stock_name=stock_name,
        targets=targets,
    )

    # Validate that all acceptable routes are solvable with the provided stock
    # Only validate if there are any acceptable routes
    has_acceptable_routes = any(len(target.acceptable_routes) > 0 for target in targets.values())
    if has_acceptable_routes:
        validate_acceptable_routes_solvable(benchmark, stock)

    return benchmark
