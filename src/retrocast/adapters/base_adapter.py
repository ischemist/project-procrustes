import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from retrocast.exceptions import AdapterError, ChemError, UnsupportedAdapterFeatureError
from retrocast.models.chem import Route, TargetIdentity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RawRouteEntry:
    """One route-like payload extracted from a raw artifact plus source context."""

    payload: Any
    source_key: str | None = None
    source_row_index: int | None = None
    source_record_id: str | None = None
    expected_target_id: str | None = None
    expected_target_smiles: str | None = None
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
        expected_target: TargetIdentity | None = None,
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

    def adapt_target_payload(
        self,
        raw_target_data: Any,
        target: TargetIdentity,
        *,
        ignore_stereo: bool = False,
        **cast_kwargs: Any,
    ) -> Iterator[Route]:
        """
        Convenience helper for adapting a target-local payload through the
        route-first seam.
        """
        for entry in self.iter_raw_entries(raw_target_data, expected_target=target):
            try:
                yield self.cast(
                    entry.payload,
                    ignore_stereo=ignore_stereo,
                    expected_target=target,
                    **cast_kwargs,
                )
            except ValidationError as exc:
                logger.warning(
                    "Adapter failed for raw entry %s: %s [adapter.schema_invalid]",
                    entry.expected_target_id or entry.source_key or entry.source_row_index,
                    exc,
                )
            except (AdapterError, ChemError) as exc:
                logger.warning(
                    "Adapter failed for raw entry %s: %s [%s]",
                    entry.expected_target_id or entry.source_key or entry.source_row_index,
                    exc,
                    exc.code,
                )
                continue
