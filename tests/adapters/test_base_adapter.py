import logging
from abc import ABC, abstractmethod
from typing import Any

import pytest

from retrocast.models.chem import Route

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


class BaseAdapterTest(ABC):
    """
    An abstract base class for adapter unit tests.

    Subclasses MUST provide the required fixtures. In return, they inherit a
    standard set of tests for common adapter failure modes and success cases.
    This ensures consistency and provides a clear template for adding new adapters.
    """

    @pytest.fixture
    @abstractmethod
    def adapter_instance(self) -> Any:
        """Yield the adapter instance to be tested."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def raw_valid_route_data(self) -> Any:
        """
        Provide a minimal, valid piece of raw data representing one or more successful routes.

        Note: This data should match the expected structure for the adapter's `adapt` method.
        See the docstring for `retrocast.adapters.base_adapter.BaseAdapter.cast` for a discussion
        of "Route-Centric" vs. "Target-Centric" data formats.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def raw_unsuccessful_run_data(self) -> Any:
        """Provide raw data representing a failed run (e.g., succ: false, or an empty list)."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def raw_invalid_schema_data(self) -> Any:
        """Provide raw data that should fail the adapter's pydantic validation."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def target_input(self) -> Any:
        """Provide the correct TargetIdentity for the `raw_valid_route_data` fixture."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def mismatched_target_input(self) -> Any:
        """Provide a TargetIdentity whose smiles does NOT match the root of `raw_valid_route_data`."""
        raise NotImplementedError

    # --- Common tests, inherited by all adapter tests ---

    def test_adapt_success(self, adapter_instance, raw_valid_route_data, target_input):
        """Tests that a valid raw route produces at least one Route."""
        routes = list(adapter_instance.cast(raw_valid_route_data, target_input))
        assert len(routes) >= 1
        route = routes[0]
        assert isinstance(route, Route)
        assert route.target.smiles == target_input.smiles
        assert route.target.inchikey  # Ensure InChIKey is populated
        assert route.rank >= 1

    def test_adapt_handles_unsuccessful_run(self, adapter_instance, raw_unsuccessful_run_data, target_input):
        """Tests that data for an unsuccessful run yields no routes."""
        routes = list(adapter_instance.cast(raw_unsuccessful_run_data, target_input))
        assert len(routes) == 0

    def test_adapt_handles_invalid_schema(self, adapter_instance, raw_invalid_schema_data, target_input, caplog):
        """Tests that data failing schema validation yields no routes and logs a warning."""
        routes = list(adapter_instance.cast(raw_invalid_schema_data, target_input))
        assert len(routes) == 0
        assert "failed" in caplog.text and "validation" in caplog.text

    def test_adapt_handles_mismatched_smiles(
        self, adapter_instance, raw_valid_route_data, mismatched_target_input, caplog
    ):
        """Tests that a SMILES mismatch between target and data yields no routes and logs a warning."""
        routes = list(adapter_instance.cast(raw_valid_route_data, mismatched_target_input))
        assert len(routes) == 0
        assert "mismatched smiles" in caplog.text.lower() or "does not match expected target" in caplog.text.lower()
