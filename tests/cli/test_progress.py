from types import SimpleNamespace

from retrocast.cli.progress import estimate_raw_route_entries


def test_estimate_raw_route_entries_counts_target_keyed_route_lists():
    raw_data = {"target-1": [{"route": 1}, {"route": 2}], "CCO": [{"route": 3}]}
    benchmark_targets = {
        "target-1": SimpleNamespace(smiles="CC"),
        "target-2": SimpleNamespace(smiles="CCO"),
    }

    assert (
        estimate_raw_route_entries(
            raw_data,
            input_kind="target-keyed-provider-output",
            benchmark_targets=benchmark_targets,
        )
        == 3
    )


def test_estimate_raw_route_entries_rejects_non_route_sized_payloads():
    raw_data = {"target-1": {"not": "a route collection"}}
    benchmark_targets = {"target-1": SimpleNamespace(smiles="CC")}

    assert (
        estimate_raw_route_entries(
            raw_data,
            input_kind="target-keyed-provider-output",
            benchmark_targets=benchmark_targets,
        )
        is None
    )


def test_estimate_raw_route_entries_counts_payload_routes_attribute():
    raw_data = {"target-1": SimpleNamespace(routes=[{"route": 1}, {"route": 2}])}
    benchmark_targets = {"target-1": SimpleNamespace(smiles="CC")}

    assert (
        estimate_raw_route_entries(
            raw_data,
            input_kind="target-keyed-provider-output",
            benchmark_targets=benchmark_targets,
        )
        == 2
    )
