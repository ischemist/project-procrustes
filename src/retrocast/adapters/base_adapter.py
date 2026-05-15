import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from retrocast.exceptions import UnsupportedAdapterFeatureError
from retrocast.models.chem import Route, TargetIdentity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RawRouteEntry:
    """One route-like payload extracted from a raw artifact plus source context."""

    payload: Any
    source_key: str | None = None
    source_row_index: int | None = None
    source_record_id: str | None = None
    target_hint_id: str | None = None
    target_hint_smiles: str | None = None
    source_order: int | None = None


class BaseAdapter(ABC):
    """
    Abstract base class for all model output adapters.

    An adapter's role is to transform a model's raw output format into the
    canonical `Route` schema.
    """

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        """
        Route-first migration seam.

        New adapters should split raw artifacts into route-like payloads here,
        then turn each payload into a canonical Route via cast().
        """
        raise UnsupportedAdapterFeatureError(
            f"{self.__class__.__name__} does not yet expose raw-route iteration",
            context={"adapter": self.__class__.__name__, "feature": "iter_raw_entries"},
        )

    @abstractmethod
    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        """
        Route-first migration seam.

        New adapters should accept one route-like payload and return one Route.
        """
        raise NotImplementedError
