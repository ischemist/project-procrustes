import pytest

from retrocast._warnings import RetroCastFutureWarning
from retrocast.models.evaluation import ScoredRoute, TargetEvaluation


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
