from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from retrocast.models.chem import Route, TargetIdentity


class BaseAdapter(ABC):
    """
    Abstract base class for all model output adapters.

    An adapter's role is to transform a model's raw output format into the
    canonical `Route` schema.
    """

    @abstractmethod
    def cast(
        self, raw_target_data: Any, target: TargetIdentity, ignore_stereo: bool = False
    ) -> Generator[Route, None, None]:
        """
        Validates, transforms, and yields Routes from raw model data.

        This is the primary method for an adapter. It encapsulates all model-specific
        logic. It should be a generator that yields successful routes and handles its
        own exceptions internally by logging and continuing.

        Args:
            raw_target_data: The raw data blob from a file for a single target.
                This blob can follow one of two common patterns:

                1.  **Route-Centric**: The data is a list of route objects, where the
                    root of each route object contains the target SMILES (e.g.,
                    AiZynthFinder, DMS). `raw_target_data` is typically a `list`.

                2.  **Target-Centric**: The data is a single JSON object that contains
                    metadata (like a top-level `smiles` key) and a nested list of
                    routes (e.g., RetroChimera). `raw_target_data` is typically a `dict`.

                The adapter is responsible for handling the specific structure of its model.
            target: The identity of the target molecule (id and canonical SMILES).
            ignore_stereo: If True, stereochemistry is stripped during SMILES canonicalization.

        Yields:
            Successfully transformed Route objects.
        """
        raise NotImplementedError
