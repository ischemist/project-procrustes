from __future__ import annotations

# ruff: noqa: E402,I001

import shutil
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from pprint import pprint

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from retrocast._warnings import RetroCastFutureWarning
from retrocast.adapters import adapt_routes, adapt_single_route, get_adapter
from retrocast.io.blob import load_json_artifact, save_jsonl_gz
from retrocast.io.data import load_benchmark, load_route_corpus, load_routes
from retrocast.models.chem import TargetInput
from retrocast.workflow.adapt import (
    adapt_benchmark_keyed_route_corpus,
    adapt_provider_output,
    adapt_route_corpus,
    adapt_target_keyed_provider_output,
)
from retrocast.workflow.collect import collect_benchmark_predictions
from retrocast.workflow.ingest import ingest_model_predictions

# this file is intentionally notebook-ish. edit run_only or comment out sections.
# it writes only to data/tmp/showcase.

data_dir = root / "data/retrocast"
benchmark_path = data_dir / "1-benchmarks/definitions/mkt-cnv-160.json.gz"
raw_dir = data_dir / "2-raw/aizynthfinder-mcts/mkt-cnv-160"
raw_results_path = raw_dir / "results.json.gz"
processed_dir = data_dir / "3-processed/mkt-cnv-160/aizynthfinder-mcts"
processed_routes_path = processed_dir / "routes.json.gz"
showcase_output_dir = root / "data/tmp/showcase"

# leave empty to run everything. otherwise list section names, e.g. {"load_legacy_routes"}.
run_only: set[str] = set()


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def count_routes(routes_by_target: dict[str, list]) -> int:
    return sum(len(routes) for routes in routes_by_target.values())


def sample_target_key(raw_data: dict) -> str:
    return next(iter(raw_data))


def section_paths() -> None:
    banner("paths")
    for label, path in {
        "benchmark": benchmark_path,
        "raw results": raw_results_path,
        "processed routes": processed_routes_path,
        "scratch output": showcase_output_dir,
    }.items():
        print(f"{label:>16}: {path} exists={path.exists()}")


def section_load_legacy_routes() -> None:
    banner("load already converted routes")
    routes_by_target = load_routes(processed_routes_path)
    non_empty_targets = [target_id for target_id, routes in routes_by_target.items() if routes]
    first_target_id = non_empty_targets[0]
    first_route = routes_by_target[first_target_id][0]

    print(f"targets: {len(routes_by_target)}")
    print(f"total routes: {count_routes(routes_by_target)}")
    print(f"first non-empty target: {first_target_id}")
    print(f"first route target smiles: {first_route.target.smiles}")
    print(f"first route length: {first_route.length}")
    print(f"legacy route.rank present: {hasattr(first_route, 'rank')}")


def section_load_synthetic_legacy_rank() -> None:
    banner("load synthetic legacy route with rank field")
    routes_by_target = load_routes(processed_routes_path)
    non_empty_targets = [target_id for target_id, routes in routes_by_target.items() if routes]
    first_route = routes_by_target[non_empty_targets[0]][0]

    legacy_row = first_route.model_dump(mode="json")
    legacy_row["rank"] = 7
    legacy_path = showcase_output_dir / "legacy-rank-route-corpus.jsonl.gz"
    save_jsonl_gz([legacy_row], legacy_path)

    loaded_route = load_route_corpus(legacy_path)[0]

    print(f"wrote synthetic legacy row with rank=7 to: {legacy_path}")
    print(f"rank survived as Route attribute: {hasattr(loaded_route, 'rank')}")
    print("warning emitted: no")
    print("compat behavior: legacy rank is ignored during validation")


def section_adapt_target_keyed_provider_output() -> None:
    banner("adapt target-keyed provider output")
    benchmark = load_benchmark(benchmark_path)
    raw_data = load_json_artifact(raw_results_path)
    adapter = get_adapter("aizynth")

    routes = adapt_target_keyed_provider_output(raw_data, benchmark, adapter)

    print(f"adapted routes: {len(routes)}")
    print(f"first route target smiles: {routes[0].target.smiles}")
    print(f"first route length: {routes[0].length}")


def section_adapt_single_provider_output() -> None:
    banner("adapt one provider output bucket")
    raw_data = load_json_artifact(raw_results_path)
    adapter = get_adapter("aizynth")
    target_id = sample_target_key(raw_data)
    provider_output = raw_data[target_id]

    routes = adapt_provider_output(provider_output, adapter)

    print(f"source target key: {target_id}")
    print(f"routes from that provider output: {len(routes)}")
    print(f"first route target smiles: {routes[0].target.smiles}")


def section_adapt_target_local_convenience_wrappers() -> None:
    banner("target-local convenience wrappers")
    benchmark = load_benchmark(benchmark_path)
    raw_data = load_json_artifact(raw_results_path)
    target_id = sample_target_key(raw_data)
    target = benchmark.targets[target_id]
    target_input = TargetInput(id=target.id, smiles=target.smiles)
    raw_target_payload = raw_data[target_id]

    one_route = adapt_single_route(raw_target_payload, target_input, "aizynth")
    routes = adapt_routes(raw_target_payload, target_input, "aizynth", max_routes=3)

    print(f"target id: {target_id}")
    print(f"adapt_single_route returned route: {one_route is not None}")
    print(f"adapt_routes max_routes=3 returned: {len(routes)}")


def section_collect_routes() -> None:
    banner("collect routes onto benchmark")
    benchmark = load_benchmark(benchmark_path)
    raw_data = load_json_artifact(raw_results_path)
    adapter = get_adapter("aizynth")

    routes = adapt_target_keyed_provider_output(raw_data, benchmark, adapter)
    collected = collect_benchmark_predictions(routes, benchmark)

    print("collection stats:")
    pprint(collected.stats.model_dump(mode="json"))
    print(f"routes by target: {len(collected.routes_by_target)}")
    print(f"total collected routes: {count_routes(collected.routes_by_target)}")


def section_ingest_target_keyed_provider_output() -> None:
    banner("ingest target-keyed provider output")
    benchmark = load_benchmark(benchmark_path)
    raw_data = load_json_artifact(raw_results_path)
    adapter = get_adapter("aizynth")

    output_dir = showcase_output_dir / "ingest-target-keyed"
    shutil.rmtree(output_dir, ignore_errors=True)

    routes_by_target, save_path, stats = ingest_model_predictions(
        model_name="aizynthfinder-mcts",
        benchmark=benchmark,
        raw_data=raw_data,
        adapter=adapter,
        output_dir=output_dir,
        provider_output_kind="target_keyed_provider_output",
    )

    print(f"save path: {save_path}")
    print(f"targets: {len(routes_by_target)}")
    print(f"total routes: {count_routes(routes_by_target)}")
    print("run stats:")
    pprint(stats.to_manifest_dict())


def section_deprecated_aliases() -> None:
    banner("deprecated aliases")
    benchmark = load_benchmark(benchmark_path)
    raw_data = load_json_artifact(raw_results_path)
    adapter = get_adapter("aizynth")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RetroCastFutureWarning)
        routes_from_old_provider_name = adapt_route_corpus(raw_data[sample_target_key(raw_data)], adapter)
        routes_from_old_target_keyed_name = adapt_benchmark_keyed_route_corpus(raw_data, benchmark, adapter)

    print(f"old provider alias routes: {len(routes_from_old_provider_name)}")
    print(f"old target-keyed alias routes: {len(routes_from_old_target_keyed_name)}")
    print("warnings:")
    for warning in caught:
        print(f"- {warning.message}")


sections: dict[str, Callable[[], None]] = {
    "paths": section_paths,
    # "load_legacy_routes": section_load_legacy_routes,
    # "load_synthetic_legacy_rank": section_load_synthetic_legacy_rank,
    # "adapt_target_keyed_provider_output": section_adapt_target_keyed_provider_output,
    # "adapt_single_provider_output": section_adapt_single_provider_output,
    # "adapt_target_local_convenience_wrappers": section_adapt_target_local_convenience_wrappers,
    # "collect_routes": section_collect_routes,
    # "ingest_target_keyed_provider_output": section_ingest_target_keyed_provider_output,
    "deprecated_aliases": section_deprecated_aliases,
}


def main() -> None:
    selected = run_only or set(sections)
    unknown = selected - set(sections)
    if unknown:
        raise ValueError(f"unknown showcase sections: {sorted(unknown)}")

    for name, section in sections.items():
        if name in selected:
            section()


if __name__ == "__main__":
    main()
