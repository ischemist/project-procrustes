import json
from collections.abc import Sequence

from retrocast.chem import InChIKeyLevel
from retrocast.models.analysis import MetricSummary
from retrocast.models.evaluation import TargetResult, Tier


def summarize_targets(
    targets: Sequence[TargetResult],
    *,
    tiers: Sequence[Tier],
    ks: Sequence[int],
    prefix_depths: Sequence[int] = (1, 2, 3),
    metric_label: str = "task",
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, MetricSummary]:
    from retrocast import _native

    payload = _native.summarize_targets_json(
        json.dumps([target.model_dump(mode="json") for target in targets], separators=(",", ":")),
        json.dumps([int(tier) for tier in tiers], separators=(",", ":")),
        json.dumps(list(ks), separators=(",", ":")),
        json.dumps(list(prefix_depths), separators=(",", ":")),
        metric_label=metric_label,
        match_level=acceptable_match_level.value,
        n_boot=n_boot,
        seed=seed,
    )
    return {name: MetricSummary.model_validate(summary) for name, summary in json.loads(payload).items()}
