import pytest

from retrocast._warnings import RetroCastFutureWarning
from retrocast.models.evaluation import ScoredCandidate, ScoredRoute, TargetEvaluation
from retrocast.models.validity import FailureRecord


@pytest.mark.unit
class TestEvaluationDeprecations:
    def test_target_is_solvable_warns_on_access(self):
        target = TargetEvaluation(target_id="t1", is_solvable=True)

        with pytest.warns(RetroCastFutureWarning, match="TargetEvaluation.is_solvable"):
            assert target.is_solvable is True

    def test_target_legacy_alias_backfills_from_stock_termination(self):
        target = TargetEvaluation(target_id="t1", has_stock_terminated_route=True)

        with pytest.warns(RetroCastFutureWarning, match="TargetEvaluation.is_solvable"):
            assert target.is_solvable is True

    def test_target_rejects_conflicting_stock_aliases(self):
        with pytest.raises(ValueError, match="is_solvable and has_stock_terminated_route disagree"):
            TargetEvaluation(target_id="t1", is_solvable=False, has_stock_terminated_route=True)

    def test_scored_route_is_solved_warns_on_access(self):
        route = ScoredRoute(rank=1, is_solved=True, matches_acceptable=False)

        with pytest.warns(RetroCastFutureWarning, match="ScoredRoute.is_solved"):
            assert route.is_solved is True

    def test_routes_property_is_not_serialized(self):
        target = TargetEvaluation(
            target_id="t1",
            candidates=[
                ScoredCandidate(
                    rank=1,
                    adapter_failure=FailureRecord(code="legacy.test", message="synthetic failure"),
                )
            ],
        )

        dumped = target.model_dump()

        assert "routes" not in dumped
        assert "legacy_routes" not in dumped

    def test_legacy_routes_load_and_warn_on_access(self):
        target = TargetEvaluation.model_validate(
            {
                "target_id": "t1",
                "routes": [{"rank": 1, "is_solved": True, "matches_acceptable": True}],
            }
        )

        with pytest.warns(RetroCastFutureWarning, match="TargetEvaluation.routes"):
            routes = target.routes

        assert routes[0].rank == 1
        assert routes[0].is_stock_terminated is True
