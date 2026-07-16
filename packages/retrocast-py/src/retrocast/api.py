from collections.abc import Sequence

from retrocast.adapters import Adapter, get_adapter
from retrocast.metrics.constraints import TaskConstraintChecker
from retrocast.models import AnalysisReport, Benchmark, Evaluation
from retrocast.workflow import analyze, ingest_candidates, score
from retrocast.workflow.collect import CollectedCandidates


def ingest_with_adapter(
    raw_payload: object,
    adapter: Adapter | str,
    task: Benchmark,
    *,
    max_candidates: int | None = None,
) -> CollectedCandidates:
    resolved = get_adapter(adapter) if isinstance(adapter, str) else adapter
    return ingest_candidates(raw_payload, resolved, task, max_candidates=max_candidates)


def score_predictions(
    predictions: CollectedCandidates,
    task: Benchmark,
    *,
    constraint_checkers: Sequence[TaskConstraintChecker] = (),
) -> Evaluation:
    return score(
        predictions,
        task,
        constraint_checkers=constraint_checkers,
    )


def analyze_evaluation(evaluation: Evaluation, *, n_boot: int = 10000) -> AnalysisReport:
    return analyze(evaluation, n_boot=n_boot)
