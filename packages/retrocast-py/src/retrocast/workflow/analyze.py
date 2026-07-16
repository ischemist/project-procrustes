"""Python-facing analysis orchestration backed by the Rust engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from retrocast.models.analysis import AnalysisReport
from retrocast.models.evaluation import Evaluation, TargetResult


def analyze(
    evaluation: Evaluation,
    *,
    ks: Sequence[int] = (1, 5, 10, 50),
    prefix_depths: Sequence[int] = (1, 2, 3),
    stratify_by: Callable[[TargetResult], str | None] | None = None,
    n_boot: int = 10000,
    seed: int = 42,
    workers: int = 1,
) -> AnalysisReport:
    """Analyze an evaluation in Rust, optionally grouping through a Python callback.

    The callback only assigns target results to named groups. Metrics,
    bootstrapping, runtime summaries, and default route-depth strata remain
    Rust-owned for both the overall report and every custom group.
    """
    from retrocast import native

    report = native.analyze(
        evaluation,
        ks=ks,
        prefix_depths=prefix_depths,
        n_boot=n_boot,
        seed=seed,
        workers=workers,
    )
    if stratify_by is None:
        return report

    groups: dict[str, dict[str, TargetResult]] = {}
    for target_id, target in evaluation.targets.items():
        label = stratify_by(target)
        if label is not None:
            groups.setdefault(label, {})[target_id] = target

    by_stratum = {}
    for label, targets in groups.items():
        subset = evaluation.model_copy(update={"targets": targets})
        by_stratum[label] = native.analyze(
            subset,
            ks=ks,
            prefix_depths=prefix_depths,
            n_boot=n_boot,
            seed=seed,
            workers=workers,
        ).metrics

    return report.model_copy(update={"by_stratum": by_stratum})
