from __future__ import annotations

import pytest

from retrocast.chem import get_inchi_key
from retrocast.models.route import Molecule, Reaction, Route
from retrocast.typing import SmilesStr
from retrocast.visualization.depth import depth_group_sort_key, depth_group_value
from retrocast.visualization.routes import extract_route_stats, is_convergent_route


def molecule(smiles: str, product_of: Reaction | None = None) -> Molecule:
    return Molecule(smiles=SmilesStr(smiles), inchikey=get_inchi_key(smiles), product_of=product_of)


@pytest.mark.unit
def test_depth_group_keys_sort_numerically() -> None:
    keys = ["depth 10", "2", "depth 1", 3]

    assert sorted(keys, key=depth_group_sort_key) == ["depth 1", "2", 3, "depth 10"]
    assert [depth_group_value(key) for key in ["depth 1", "2", 3]] == [1, 2, 3]


@pytest.mark.unit
def test_route_stats_use_schema_v2_route_shape() -> None:
    intermediate = molecule("CO", product_of=Reaction(reactants=[molecule("C")]))
    route = Route(target=molecule("CCO", product_of=Reaction(reactants=[intermediate, molecule("N")])))

    stats = extract_route_stats({"target": route})

    assert len(stats) == 1
    assert stats[0].depth == 2
    assert stats[0].target_hac == 3
    assert stats[0].target_mw > 0
    assert not stats[0].is_convergent


@pytest.mark.unit
def test_convergent_route_requires_multiple_expanded_children_at_one_step() -> None:
    left = molecule("CO", product_of=Reaction(reactants=[molecule("C")]))
    right = molecule("CN", product_of=Reaction(reactants=[molecule("N")]))
    route = Route(target=molecule("CCN", product_of=Reaction(reactants=[left, right, molecule("O")])))

    assert is_convergent_route(route)
