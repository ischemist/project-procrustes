from collections.abc import Iterator
from typing import Any

import pytest

from retrocast._warnings import RetroCastFutureWarning
from retrocast.adapters.base_adapter import BaseAdapter, RawRouteEntry
from retrocast.exceptions import AdapterLogicError, BenchmarkCollectionError
from retrocast.models.benchmark import BenchmarkSet, BenchmarkTarget
from retrocast.models.chem import Route, TargetIdentity
from retrocast.typing import SmilesStr
from retrocast.workflow.adapt import (
    adapt_benchmark_keyed_route_corpus,
    adapt_provider_output,
    adapt_route,
    adapt_route_corpus,
    adapt_target_keyed_provider_output,
)
from retrocast.workflow.collect import collect_benchmark_predictions
from tests.helpers import _make_simple_route, _make_two_step_route, _synthetic_inchikey


class RouteFirstSyntheticAdapter(BaseAdapter):
    """Minimal route-first adapter for workflow tests."""

    def iter_raw_entries(
        self,
        raw_data: Any,
        *,
        source_key: str | None = None,
    ) -> Iterator[RawRouteEntry]:
        if not isinstance(raw_data, list):
            raise ValueError("expected list payload")

        for row_index, item in enumerate(raw_data, start=1):
            yield RawRouteEntry(
                payload=item,
                source_key=source_key,
                source_row_index=row_index,
                target_hint_id=None,
                target_hint_smiles=None,
                source_order=row_index,
            )

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        route = raw_route if isinstance(raw_route, Route) else Route.model_validate(raw_route)
        if expected_target is not None and route.target.smiles != expected_target.smiles:
            raise AdapterLogicError(
                "route target does not match expected target",
                code="adapter.target_mismatch",
                context={"expected_target": expected_target.id, "actual_target": route.target.smiles},
            )
        return route


class SingleRouteSyntheticAdapter(BaseAdapter):
    """Minimal adapter for one raw route-like payload."""

    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
        expected_target: TargetIdentity | None = None,
    ) -> Route:
        return raw_route if isinstance(raw_route, Route) else Route.model_validate(raw_route)


@pytest.fixture
def synthetic_benchmark() -> BenchmarkSet:
    return BenchmarkSet(
        name="route-corpus-benchmark",
        targets={
            "target_1": BenchmarkTarget(
                id="target_1",
                smiles=SmilesStr("CC"),
                inchi_key=_synthetic_inchikey("CC"),
                acceptable_routes=[],
            ),
            "target_2": BenchmarkTarget(
                id="target_2",
                smiles=SmilesStr("CCC"),
                inchi_key=_synthetic_inchikey("CCC"),
                acceptable_routes=[],
            ),
        },
    )


def _first_leaf_smiles(route: Route) -> str:
    return sorted(leaf.smiles for leaf in route.leaves)[0]


@pytest.mark.unit
class TestAdaptProviderOutput:
    def test_adapt_provider_output_supports_unkeyed_route_lists(self):
        raw_data = [
            _make_simple_route("CC", "C", rank=9).model_dump(mode="json"),
            _make_two_step_route("CCC", "CC", "C", rank=4).model_dump(mode="json"),
        ]

        routes = adapt_provider_output(raw_data, RouteFirstSyntheticAdapter())

        assert len(routes) == 2
        assert [route.target.smiles for route in routes] == ["CC", "CCC"]

    def test_adapt_route_accepts_one_route_like_payload_without_target_context(self):
        raw_data = _make_simple_route("CC", "C", rank=9).model_dump(mode="json")

        route = adapt_route(raw_data, SingleRouteSyntheticAdapter())

        assert route is not None
        assert route.target.smiles == "CC"

    def test_adapt_target_keyed_provider_output_filters_to_benchmark_targets(self, synthetic_benchmark):
        raw_data = {
            "target_1": [_make_simple_route("CC", "C", rank=7).model_dump(mode="json")],
            "CCC": [_make_two_step_route("CCC", "CC", "C", rank=3).model_dump(mode="json")],
            "unused": [_make_simple_route("CCCC", "CC", rank=1).model_dump(mode="json")],
        }

        routes = adapt_target_keyed_provider_output(raw_data, synthetic_benchmark, RouteFirstSyntheticAdapter())

        assert len(routes) == 2
        assert [route.target.smiles for route in routes] == ["CC", "CCC"]

    def test_legacy_adapt_names_warn(self, synthetic_benchmark):
        provider_output = [_make_simple_route("CC", "C").model_dump(mode="json")]
        target_keyed_provider_output = {"target_1": provider_output}

        with pytest.warns(RetroCastFutureWarning, match="adapt_route_corpus"):
            routes = adapt_route_corpus(provider_output, RouteFirstSyntheticAdapter())
        with pytest.warns(RetroCastFutureWarning, match="adapt_benchmark_keyed_route_corpus"):
            keyed_routes = adapt_benchmark_keyed_route_corpus(
                target_keyed_provider_output,
                synthetic_benchmark,
                RouteFirstSyntheticAdapter(),
            )

        assert len(routes) == 1
        assert len(keyed_routes) == 1


@pytest.mark.unit
class TestCollectBenchmarkPredictions:
    def test_collect_benchmark_predictions_orders_and_deduplicates(self, synthetic_benchmark):
        route_o = _make_simple_route("CC", "O", rank=9)
        route_c = _make_simple_route("CC", "C", rank=5)
        duplicate_route_o = Route.model_validate(route_o.model_dump(mode="json"))

        collected = collect_benchmark_predictions(
            [route_o, route_c, duplicate_route_o],
            synthetic_benchmark,
        )

        assert collected.stats.matched_by_canonical_smiles == 3
        assert collected.stats.duplicate_routes_dropped == 1
        assert len(collected.routes_by_target["target_1"]) == 2
        assert _first_leaf_smiles(collected.routes_by_target["target_1"][0]) == "O"
        assert _first_leaf_smiles(collected.routes_by_target["target_1"][1]) == "C"

    def test_collect_benchmark_predictions_reports_unmatched_routes(self, synthetic_benchmark):
        unmatched_route = _make_simple_route("CCCC", "CC", rank=1)

        collected = collect_benchmark_predictions([unmatched_route], synthetic_benchmark)

        assert collected.stats.unmatched_routes == 1
        assert collected.stats.final_unique_routes_saved == 0

    def test_collect_benchmark_predictions_warns_for_legacy_report_policy(self, synthetic_benchmark):
        unmatched_route = _make_simple_route("CCCC", "CC", rank=1)

        with pytest.warns(RetroCastFutureWarning, match="on_unmatched='ignore'"):
            collected = collect_benchmark_predictions(
                [unmatched_route],
                synthetic_benchmark,
                on_unmatched="report",
            )

        assert collected.stats.unmatched_routes == 1

    def test_collect_benchmark_predictions_raises_on_ambiguous_smiles(self):
        ambiguous_benchmark = BenchmarkSet(
            name="ambiguous-benchmark",
            targets={
                "a": BenchmarkTarget(
                    id="a",
                    smiles=SmilesStr("CC"),
                    inchi_key=_synthetic_inchikey("CC"),
                    acceptable_routes=[],
                ),
                "b": BenchmarkTarget(
                    id="b",
                    smiles=SmilesStr("CC"),
                    inchi_key=_synthetic_inchikey("CC"),
                    acceptable_routes=[],
                ),
            },
        )
        route = _make_simple_route("CC", "C", rank=1)

        with pytest.raises(BenchmarkCollectionError) as exc_info:
            collect_benchmark_predictions([route], ambiguous_benchmark)

        assert exc_info.value.code == "collection.ambiguous_smiles_match"
