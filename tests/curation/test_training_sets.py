from __future__ import annotations

import gzip
import json

import pytest

from retrocast.curation.training_sets import (
    AdaptedTrainingRoute,
    RawRouteSource,
    TrainingSetBuildConfig,
    assign_train_val_splits,
    build_training_records_from_adapted,
    summarize_records,
    write_jsonl_gz,
)
from retrocast.models.chem import Molecule, ReactionStep, Route
from retrocast.typing import InchiKeyStr, SmilesStr


def make_leaf(name: str, buyable: bool = True) -> Molecule:
    suffix = "BUY" if buyable else "MISS"
    return Molecule(
        smiles=SmilesStr(name),
        inchikey=InchiKeyStr(f"INCHI-{name}-{suffix}"),
    )


def make_route(name: str, depth: int, buyable: bool = True) -> Route:
    current = make_leaf(f"{name}-0", buyable=buyable)
    for idx in range(1, depth + 1):
        current = Molecule(
            smiles=SmilesStr(f"{name}-{idx}"),
            inchikey=InchiKeyStr(f"INCHI-{name}-{idx}"),
            synthesis_step=ReactionStep(reactants=[current]),
        )
    return Route(target=current, rank=1)


def make_adapted_route(name: str, route: Route) -> AdaptedTrainingRoute:
    return AdaptedTrainingRoute(
        route=route,
        route_signature=route.get_signature(),
        reaction_signatures=route.get_reaction_signatures(),
        source=RawRouteSource(dataset="all-routes", raw_index=0, raw_route_hash=f"hash-{name}"),
    )


@pytest.mark.unit
class TestTrainingSetSplits:
    def test_assign_train_val_splits_is_deterministic_and_stratified(self):
        routes = {f"route-{idx}": make_route(f"r{idx}", depth=2, buyable=True) for idx in range(20)}
        routes.update({f"missing-{idx}": make_route(f"m{idx}", depth=2, buyable=False) for idx in range(20)})
        stock = {InchiKeyStr(f"INCHI-r{idx}-0-BUY") for idx in range(20)}

        first = assign_train_val_splits(routes, buyables_stock=stock, val_fraction=0.1, seed=17)
        second = assign_train_val_splits(routes, buyables_stock=stock, val_fraction=0.1, seed=17)

        assert first == second
        assert sum(split == "val" for split in first.values()) == 4
        assert sum(first[f"route-{idx}"] == "val" for idx in range(20)) == 2
        assert sum(first[f"missing-{idx}"] == "val" for idx in range(20)) == 2

    def test_assign_train_val_splits_rejects_invalid_fraction(self):
        with pytest.raises(ValueError, match="val_fraction"):
            assign_train_val_splits({}, buyables_stock=set(), val_fraction=0, seed=1)

    def test_build_from_adapted_routes_reuses_adapted_inputs(self):
        route = make_route("keep", depth=1, buyable=True)
        heldout = make_route("heldout", depth=1, buyable=True)
        stock = {leaf.inchikey for leaf in route.leaves | heldout.leaves}

        result = build_training_records_from_adapted(
            all_routes=[make_adapted_route("keep", route), make_adapted_route("heldout", heldout)],
            raw_all_routes_count=2,
            skipped_adaptation=3,
            heldout_routes={"n1": [make_adapted_route("heldout", heldout)]},
            buyables_stock=stock,
            config=TrainingSetBuildConfig(holdout_mode="route", show_progress=False),
        )

        assert [record.route_signature for record in result.records] == [route.get_signature()]
        assert result.statistics["raw_all_routes"] == 2
        assert result.statistics["skipped_adaptation"] == 3
        assert result.statistics["skipped_route_holdout"] == 1


@pytest.mark.unit
class TestTrainingSetJsonl:
    def test_write_jsonl_gz_writes_record_dicts(self, tmp_path):
        route = make_route("a", depth=1, buyable=True)

        class Record:
            split = "train"
            buyables_solved = True

            def to_json_dict(self):
                return {"id": "example", "route": route.model_dump(mode="json")}

        out_path = tmp_path / "records.jsonl.gz"
        n_written = write_jsonl_gz([Record()], out_path)  # type: ignore[list-item]

        assert n_written == 1
        with gzip.open(out_path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        assert rows[0]["id"] == "example"
        assert rows[0]["route"]["target"]["inchikey"] == "INCHI-a-1"

    def test_summarize_records_counts_splits_and_buyables(self):
        class Record:
            def __init__(self, split: str, buyables_solved: bool) -> None:
                self.split = split
                self.buyables_solved = buyables_solved

        summary = summarize_records(
            [
                Record("train", True),
                Record("train", False),
                Record("val", True),
            ]  # type: ignore[list-item]
        )

        assert summary["n_records"] == 3
        assert summary["splits"] == {"train": 2, "val": 1}
        assert summary["buyables"] == {"train": 1, "val": 1, "all": 2}
