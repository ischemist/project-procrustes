from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

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

    # The "Gold Standard" route from the literature/patent.
    # Optional, because some benchmarks might be pure prediction tasks.
    ground_truth: Route | None = None
    # First-class properties for stratification
    is_convergent: bool | None = Field(..., description="True if the ground truth route contains convergent steps.")
    route_length: int | None = Field(
        ..., description="The length of the longest linear path in the ground truth route."
    )


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

    def get_smiles_map(self) -> dict[str, list[str]]:
        """
        Returns a mapping of {smiles: [target_id_1, target_id_2, ...]}.

        Crucial for mapping model predictions (which are keyed by SMILES)
        to benchmark targets (which are keyed by ID). Handles the edge case
        where the same molecule appears multiple times with different ground truths.
        """
        mapping = defaultdict(list)
        for target in self.targets.values():
            mapping[target.smiles].append(target.id)
        return dict(mapping)

    def get_inchikey_map(self) -> dict[str, list[str]]:
        """
        Returns a mapping of {inchi_key: [target_id_1, target_id_2, ...]}.

        Similar to get_smiles_map but uses InChIKeys as keys. Useful for
        canonical molecule identity lookups and handling tautomers/stereoisomers.
        """
        mapping = defaultdict(list)
        for target in self.targets.values():
            mapping[target.inchi_key].append(target.id)
        return dict(mapping)

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
