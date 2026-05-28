from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from retrocast.exceptions import AdapterError, AdapterLogicError, AdapterSchemaError, ChemError
from retrocast.v2.adapters.base import Adapter
from retrocast.v2.models.route import Route
from retrocast.v2.models.task import Target


@dataclass(frozen=True)
class RawExtractionContractCase:
    valid_payload: Any
    malformed_payload: Any
    source_key: str = "source-1"
    expected_entry_count: int | None = None
    expected_source_keys: list[str | None] | None = None
    expected_source_order: int | None = None


@dataclass(frozen=True)
class CastContractCase:
    valid_raw_route: Any
    malformed_raw_route: Any
    target: Target
    mismatched_target: Target
    expected_root_reactant_count: int | None = None


@dataclass(frozen=True)
class InvalidSmilesContractCase:
    invalid_leaf_raw_route: Any
    expected_pruned_root_reactants: list[str] | None = None
    supports_prune: bool = True


@dataclass(frozen=True)
class AdapterContractCase:
    adapter: Adapter
    extraction: RawExtractionContractCase
    casting: CastContractCase
    invalid_smiles: InvalidSmilesContractCase | None = None


def assert_adapter_accepts_valid_route(
    adapter: Adapter,
    raw_route: Any,
    target: Target,
) -> Route:
    route = adapter.cast(raw_route, target=target)

    assert isinstance(route, Route)
    assert route.schema_version == "2"
    assert route.target.smiles == target.smiles
    assert route.target.inchikey
    return route


class AdapterContractSuite:
    @pytest.fixture
    def adapter_contract_case(self) -> AdapterContractCase:
        raise NotImplementedError

    @pytest.mark.contract
    def test_iter_raw_routes_accepts_valid_payload(self, adapter_contract_case: AdapterContractCase) -> None:
        case = adapter_contract_case
        extraction = case.extraction

        entries = list(case.adapter.iter_raw_routes(extraction.valid_payload, source_key=extraction.source_key))

        assert entries
        assert entries[0].payload is not None
        if extraction.expected_entry_count is not None:
            assert len(entries) == extraction.expected_entry_count
        if extraction.expected_source_keys is not None:
            assert [entry.source_key for entry in entries] == extraction.expected_source_keys
        else:
            assert entries[0].source_key == extraction.source_key
        if extraction.expected_source_order is not None:
            assert entries[0].source_order == extraction.expected_source_order

    @pytest.mark.contract
    def test_iter_raw_routes_rejects_malformed_payload(self, adapter_contract_case: AdapterContractCase) -> None:
        case = adapter_contract_case
        extraction = case.extraction

        with pytest.raises(AdapterSchemaError) as exc_info:
            list(case.adapter.iter_raw_routes(extraction.malformed_payload, source_key=extraction.source_key))

        assert exc_info.value.code == "adapter.schema_invalid"

    @pytest.mark.contract
    def test_cast_accepts_valid_route(self, adapter_contract_case: AdapterContractCase) -> None:
        case = adapter_contract_case
        casting = case.casting

        route = assert_adapter_accepts_valid_route(case.adapter, casting.valid_raw_route, casting.target)

        if casting.expected_root_reactant_count is not None:
            assert route.target.product_of is not None
            assert len(route.target.product_of.reactants) == casting.expected_root_reactant_count

    @pytest.mark.contract
    def test_cast_rejects_malformed_route(self, adapter_contract_case: AdapterContractCase) -> None:
        case = adapter_contract_case
        casting = case.casting

        with pytest.raises(AdapterSchemaError) as exc_info:
            case.adapter.cast(casting.malformed_raw_route, target=casting.target)

        assert exc_info.value.code == "adapter.schema_invalid"

    @pytest.mark.contract
    def test_cast_rejects_target_mismatch(self, adapter_contract_case: AdapterContractCase) -> None:
        case = adapter_contract_case
        casting = case.casting

        with pytest.raises(AdapterLogicError) as exc_info:
            case.adapter.cast(casting.valid_raw_route, target=casting.mismatched_target)

        assert exc_info.value.code == "adapter.target_mismatch"

    @pytest.mark.contract
    def test_cast_strict_rejects_invalid_smiles(self, adapter_contract_case: AdapterContractCase) -> None:
        case = adapter_contract_case
        invalid_smiles = case.invalid_smiles
        if invalid_smiles is None:
            pytest.skip("raw format does not provide an invalid-leaf contract case")

        with pytest.raises((AdapterError, ChemError)) as exc_info:
            case.adapter.cast(invalid_smiles.invalid_leaf_raw_route, target=case.casting.target, mode="strict")

        assert exc_info.value.code == "chem.invalid_smiles" or exc_info.value.code.startswith("adapter.")

    @pytest.mark.contract
    def test_cast_prune_drops_invalid_leaf(self, adapter_contract_case: AdapterContractCase) -> None:
        case = adapter_contract_case
        invalid_smiles = case.invalid_smiles
        if invalid_smiles is None:
            pytest.skip("raw format does not provide an invalid-leaf contract case")
        if not invalid_smiles.supports_prune:
            with pytest.raises(AdapterError) as exc_info:
                case.adapter.cast(invalid_smiles.invalid_leaf_raw_route, target=case.casting.target, mode="prune")
            assert exc_info.value.code.startswith("adapter.")
            return
        if invalid_smiles.expected_pruned_root_reactants is None:
            pytest.skip("prune contract case does not specify expected root reactants")

        route = case.adapter.cast(invalid_smiles.invalid_leaf_raw_route, target=case.casting.target, mode="prune")

        root_reactants = [reactant.value.smiles for reactant in route.reaction_at("rc:r:/").reactants()]
        assert root_reactants == invalid_smiles.expected_pruned_root_reactants
