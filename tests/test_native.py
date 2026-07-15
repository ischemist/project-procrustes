from __future__ import annotations

import pytest

import retrocast
from retrocast import native
from retrocast.adapters.aizynth import AiZynthFinderAdapter
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.io import (
    load_collected_candidates,
    load_evaluation,
    save_benchmark,
    save_collected_candidates,
    save_evaluation,
)
from retrocast.metrics.constraints import RequiredLeavesChecker, RouteDepthChecker, StockTerminationChecker
from retrocast.models.evaluation import AcceptableRouteMatch, Evaluation
from retrocast.models.route import InChIKeyLevel
from retrocast.models.task import Benchmark, StockTerminationConstraint, Target
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.workflow.stats import collected_candidate_statistics, evaluation_statistics

pytestmark = pytest.mark.skipif(not native.available(), reason="native extension is not built")


def _raw_route() -> dict:
    return {
        "type": "mol",
        "smiles": "OCC",
        "scores": {"state score": 0.75},
        "children": [
            {
                "type": "reaction",
                "smiles": "CCO",
                "metadata": {"template": "tmpl-1"},
                "children": [
                    {"type": "mol", "smiles": "C", "in_stock": True},
                    {"type": "mol", "smiles": "CC", "in_stock": True},
                ],
            }
        ],
    }


def _benchmark() -> Benchmark:
    smiles = SmilesStr(canonicalize_smiles("CCO"))
    target = Target(id="ethanol", smiles=smiles, inchikey=InChIKeyStr(get_inchi_key(smiles)))
    return Benchmark(
        name="native-test",
        targets={target.id: target},
        default_constraints=[StockTerminationConstraint(stock="test-stock")],
    )


def test_python_adapt_facade_uses_native_engine() -> None:
    adapter = AiZynthFinderAdapter()

    public_candidates = retrocast.adapt([_raw_route()], adapter, workers=2)
    native_candidates = native.adapt([_raw_route()], adapter, workers=2)

    assert public_candidates == native_candidates


def test_python_pipeline_facade_uses_native_engine() -> None:
    adapter = AiZynthFinderAdapter()
    task = _benchmark()
    raw = {"ethanol": [_raw_route()]}
    candidates = retrocast.ingest_candidates(raw, adapter, task, workers=2)
    assert candidates == native.ingest(raw, adapter, task, workers=2)

    stock = {InChIKeyStr(get_inchi_key("C")), InChIKeyStr(get_inchi_key("CC"))}
    checkers = [
        StockTerminationChecker(stocks={"test-stock": stock}),
        RequiredLeavesChecker(),
        RouteDepthChecker(),
    ]
    score_kwargs = {
        "constraint_checkers": checkers,
        "acceptable_match_level": InChIKeyLevel.FULL,
        "acceptable_route_match": AcceptableRouteMatch.PREFIX,
    }
    evaluation = retrocast.score(candidates, task, workers=2, **score_kwargs)
    native_evaluation = native.score(candidates, task, workers=2, execution_stats=None, **score_kwargs)
    assert evaluation.model_dump(mode="json") == native_evaluation.model_dump(mode="json")

    report = retrocast.analyze(evaluation, workers=2, n_boot=100)
    native_report = native.analyze(
        evaluation, ks=(1, 5, 10, 50), prefix_depths=(1, 2, 3), workers=2, n_boot=100, seed=42
    )
    assert report == native_report


def test_native_evaluation_equality_normalizes_empty_pydantic_extras() -> None:
    native_evaluation = native.NativeEvaluation.model_construct()
    plain_evaluation = Evaluation.model_construct()
    object.__setattr__(native_evaluation, "__pydantic_extra__", {})
    object.__setattr__(plain_evaluation, "__pydantic_extra__", None)

    assert native_evaluation == plain_evaluation
    assert plain_evaluation == native_evaluation


def test_python_pipeline_keeps_rust_values_between_untouched_stages(monkeypatch) -> None:
    adapter = AiZynthFinderAdapter()
    task = _benchmark()
    predictions = retrocast.ingest_candidates({"ethanol": [_raw_route()]}, adapter, task)
    assert predictions.native_handle() is not None

    def reject_json_fallback(*args, **kwargs):
        raise AssertionError("pipeline materialized an intermediate value")

    assert native._native is not None
    monkeypatch.setattr(native._native, "score_json", reject_json_fallback)
    monkeypatch.setattr(native._native, "analyze_json", reject_json_fallback)

    stock = {InChIKeyStr(get_inchi_key("C")), InChIKeyStr(get_inchi_key("CC"))}
    evaluation = retrocast.score(
        predictions,
        task,
        constraint_checkers=[
            StockTerminationChecker(stocks={"test-stock": stock}),
            RequiredLeavesChecker(),
            RouteDepthChecker(),
        ],
        acceptable_match_level=InChIKeyLevel.FULL,
        acceptable_route_match=AcceptableRouteMatch.PREFIX,
    )
    assert evaluation.native_handle() is not None
    assert retrocast.analyze(evaluation, n_boot=10).metrics


def test_access_materializes_python_values_and_invalidates_native_handoff() -> None:
    adapter = AiZynthFinderAdapter()
    task = _benchmark()
    predictions = retrocast.ingest_candidates({"ethanol": [_raw_route()]}, adapter, task)

    assert predictions["ethanol"][0].rank == 1
    assert predictions.native_handle() is None

    stock = {InChIKeyStr(get_inchi_key("C")), InChIKeyStr(get_inchi_key("CC"))}
    evaluation = retrocast.score(
        predictions,
        task,
        constraint_checkers=[
            StockTerminationChecker(stocks={"test-stock": stock}),
            RequiredLeavesChecker(),
            RouteDepthChecker(),
        ],
        acceptable_match_level=InChIKeyLevel.FULL,
        acceptable_route_match=AcceptableRouteMatch.PREFIX,
    )
    assert evaluation.targets["ethanol"].candidates[0].rank == 1
    assert not isinstance(evaluation, native.NativeEvaluation)


def test_workflow_statistics_preserve_opaque_rust_values(monkeypatch) -> None:
    adapter = AiZynthFinderAdapter()
    task = _benchmark()
    predictions = retrocast.ingest_candidates({"ethanol": [_raw_route()]}, adapter, task)

    def reject_json_fallback(*args, **kwargs):
        raise AssertionError("statistics materialized an opaque Rust value")

    assert native._native is not None
    monkeypatch.setattr(native._native, "collected_candidate_statistics_json", reject_json_fallback)
    prediction_stats = collected_candidate_statistics(predictions)
    assert prediction_stats.to_manifest_dict()["successful_candidates"] == 1
    assert predictions.native_handle() is not None

    stock = {InChIKeyStr(get_inchi_key("C")), InChIKeyStr(get_inchi_key("CC"))}
    evaluation = retrocast.score(
        predictions,
        task,
        constraint_checkers=[
            StockTerminationChecker(stocks={"test-stock": stock}),
            RequiredLeavesChecker(),
            RouteDepthChecker(),
        ],
        acceptable_match_level=InChIKeyLevel.FULL,
        acceptable_route_match=AcceptableRouteMatch.PREFIX,
    )
    monkeypatch.setattr(native._native, "evaluation_statistics_json", reject_json_fallback)
    assert evaluation_statistics(evaluation) == {
        "n_targets": 1,
        "n_candidates": 1,
        "n_failed_candidates": 0,
        "n_solv_0": 1,
    }
    assert evaluation.native_handle() is not None


def test_file_artifacts_round_trip_without_materializing_native_graphs(tmp_path, monkeypatch) -> None:
    adapter = AiZynthFinderAdapter()
    task = _benchmark()
    predictions_path = tmp_path / "candidates.json.gz"
    evaluation_path = tmp_path / "evaluation.json.gz"

    predictions = retrocast.ingest_candidates({"ethanol": [_raw_route()]}, adapter, task)
    save_collected_candidates(predictions, predictions_path)
    loaded_predictions = load_collected_candidates(predictions_path)
    assert loaded_predictions.native_handle() is not None

    def reject_materialization(*args, **kwargs):
        raise AssertionError("file-native artifact was serialized through Python")

    monkeypatch.setattr(type(loaded_predictions), "_ensure_materialized", reject_materialization)
    stock = {InChIKeyStr(get_inchi_key("C")), InChIKeyStr(get_inchi_key("CC"))}
    evaluation = retrocast.score(
        loaded_predictions,
        task,
        constraint_checkers=[
            StockTerminationChecker(stocks={"test-stock": stock}),
            RequiredLeavesChecker(),
            RouteDepthChecker(),
        ],
        acceptable_match_level=InChIKeyLevel.FULL,
        acceptable_route_match=AcceptableRouteMatch.PREFIX,
    )
    save_evaluation(evaluation, evaluation_path)
    loaded_evaluation = load_evaluation(evaluation_path)
    assert loaded_evaluation.native_handle() is not None
    monkeypatch.setattr(type(loaded_evaluation), "_ensure_materialized", reject_materialization)
    assert retrocast.analyze(loaded_evaluation, n_boot=10).metrics


def test_project_scoring_rejects_unsafe_stock_name_before_path_join(tmp_path) -> None:
    task = _benchmark().model_copy(update={"default_constraints": [StockTerminationConstraint(stock="../outside")]})
    predictions = retrocast.ingest_candidates({"ethanol": [_raw_route()]}, AiZynthFinderAdapter(), task)
    predictions_path = tmp_path / "candidates.json.gz"
    task_path = tmp_path / "benchmark.json.gz"
    save_collected_candidates(predictions, predictions_path)
    save_benchmark(task, task_path)

    with pytest.raises(ValueError, match="unsafe stock name path component"):
        native.score_project_files(
            predictions_path,
            task_path,
            tmp_path / "stocks",
            execution_stats_path=None,
            acceptable_match_level=InChIKeyLevel.FULL,
            acceptable_route_match=AcceptableRouteMatch.PREFIX,
        )
