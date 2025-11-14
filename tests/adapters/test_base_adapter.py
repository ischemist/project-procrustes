from abc import ABC, abstractmethod
from typing import Any

import pytest

from retrocast.domain.schemas import BenchmarkTree


class BaseAdapterTest(ABC):
    """
    an abstract base class for adapter unit tests.

    subclasses MUST provide the required fixtures. in return, they inherit a
    standard set of tests for common adapter failure modes and success cases.
    this ensures consistency and provides a clear template for adding new adapters.
    """

    @pytest.fixture
    @abstractmethod
    def adapter_instance(self) -> Any:
        """yield the adapter instance to be tested."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def raw_valid_route_data(self) -> Any:
        """
        Provide a minimal, valid piece of raw data representing one or more successful routes.

        Note: This data should match the expected structure for the adapter's `adapt` method.
        See the docstring for `ursa.adapters.base_adapter.BaseAdapter.adapt` for a discussion
        of "Route-Centric" vs. "Target-Centric" data formats.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def raw_unsuccessful_run_data(self) -> Any:
        """provide raw data representing a failed run (e.g., succ: false, or an empty list)."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def raw_invalid_schema_data(self) -> Any:
        """provide raw data that should fail the adapter's pydantic validation."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def target_info(self) -> Any:
        """provide the correct targetinfo for the `raw_valid_route_data` fixture."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def mismatched_target_info(self) -> Any:
        """provide a targetinfo whose smiles does NOT match the root of `raw_valid_route_data`."""
        raise NotImplementedError

    # --- common tests, inherited by all adapter tests ---

    def test_adapt_success(self, adapter_instance, raw_valid_route_data, target_info):
        """tests that a valid raw route produces at least one benchmarktree."""
        trees = list(adapter_instance.adapt(raw_valid_route_data, target_info))
        assert len(trees) >= 1
        tree = trees[0]
        assert isinstance(tree, BenchmarkTree)
        assert tree.target.id == target_info.id
        assert tree.retrosynthetic_tree.smiles == target_info.smiles

    def test_adapt_handles_unsuccessful_run(self, adapter_instance, raw_unsuccessful_run_data, target_info):
        """tests that data for an unsuccessful run yields no trees."""
        trees = list(adapter_instance.adapt(raw_unsuccessful_run_data, target_info))
        assert len(trees) == 0

    def test_adapt_handles_invalid_schema(self, adapter_instance, raw_invalid_schema_data, target_info, caplog):
        """tests that data failing schema validation yields no trees and logs a warning."""
        trees = list(adapter_instance.adapt(raw_invalid_schema_data, target_info))
        assert len(trees) == 0
        assert "failed" in caplog.text and "validation" in caplog.text

    def test_adapt_handles_mismatched_smiles(
        self, adapter_instance, raw_valid_route_data, mismatched_target_info, caplog
    ):
        """tests that a smiles mismatch between target and data yields no trees and logs a warning."""
        trees = list(adapter_instance.adapt(raw_valid_route_data, mismatched_target_info))
        assert len(trees) == 0
        assert "mismatched smiles" in caplog.text.lower() or "does not match expected target" in caplog.text.lower()
