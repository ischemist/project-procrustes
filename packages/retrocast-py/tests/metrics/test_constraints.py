from __future__ import annotations

import pytest

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.exceptions import UnsupportedTaskConstraintError
from retrocast.metrics.constraints import (
    RequiredLeavesChecker,
    RouteDepthChecker,
    StockTerminationChecker,
    check_task_constraints,
)
from retrocast.models import (
    CheckStatus,
    Molecule,
    Reaction,
    RequiredLeavesConstraint,
    Route,
    RouteDepthConstraint,
    StockTerminationConstraint,
    TaskConstraint,
)
from retrocast.typing import InChIKeyStr, SmilesStr


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(
        smiles=SmilesStr(canonical),
        inchikey=InChIKeyStr(get_inchi_key(canonical)),
        product_of=product_of,
    )


def route() -> Route:
    return Route(target=molecule("CCO", product_of=Reaction(reactants=[molecule("C"), molecule("CO")])))


def route_with_depth(depth: int) -> Route:
    value = molecule("C")
    for _ in range(depth):
        value = molecule("C", product_of=Reaction(reactants=[value]))
    return Route(target=value)


def stock_for(*smiles_values: str) -> set[InChIKeyStr]:
    return {InChIKeyStr(get_inchi_key(canonicalize_smiles(smiles))) for smiles in smiles_values}


def inchikey_for(smiles: str) -> InChIKeyStr:
    return InChIKeyStr(get_inchi_key(canonicalize_smiles(smiles)))


def test_stock_constraint_passes_when_all_leaves_are_in_stock() -> None:
    result = check_task_constraints(
        route(),
        [StockTerminationConstraint(stock="stock-a")],
        [StockTerminationChecker(stocks={"stock-a": stock_for("C", "CO")})],
    )

    assert result.status == CheckStatus.PASS


def test_stock_constraint_fails_when_leaf_is_missing() -> None:
    result = check_task_constraints(
        route(),
        [StockTerminationConstraint(stock="stock-a")],
        [StockTerminationChecker(stocks={"stock-a": stock_for("C")})],
    )

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.stock_termination.missing_leaf"


def test_stock_constraint_selects_stock_by_task_constraint_name() -> None:
    stocks = {
        "stock-a": stock_for("C"),
        "stock-b": stock_for("C", "CO"),
    }

    checkers = [StockTerminationChecker(stocks=stocks)]

    assert (
        check_task_constraints(route(), [StockTerminationConstraint(stock="stock-a")], checkers).status
        == CheckStatus.FAIL
    )
    assert (
        check_task_constraints(route(), [StockTerminationConstraint(stock="stock-b")], checkers).status
        == CheckStatus.PASS
    )


def test_stock_constraint_fails_clearly_when_named_stock_is_not_registered() -> None:
    result = check_task_constraints(
        route(),
        [StockTerminationConstraint(stock="stock-b")],
        [StockTerminationChecker(stocks={"stock-a": stock_for("C", "CO")})],
    )

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.stock_termination.unregistered_stock"
    assert result.checks[0].details == {"stock": "stock-b"}


def test_stock_constraint_fails_clearly_when_registered_stock_is_empty() -> None:
    result = check_task_constraints(
        route(),
        [StockTerminationConstraint(stock="stock-a")],
        [StockTerminationChecker(stocks={"stock-a": set()})],
    )

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.stock_termination.empty_stock"
    assert result.checks[0].details == {"stock": "stock-a"}


def test_required_leaf_constraint_fails_when_required_leaf_is_missing() -> None:
    result = check_task_constraints(
        route(),
        [RequiredLeavesConstraint(smiles=[SmilesStr("CC")])],
        [RequiredLeavesChecker()],
    )

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.required_leaf.missing"
    assert result.checks[0].details == {"missing_leaf_inchikeys": [inchikey_for("CC")]}


def test_route_depth_constraint_fails_when_route_is_too_deep() -> None:
    deep_route = Route(
        target=molecule(
            "CCO",
            product_of=Reaction(
                reactants=[molecule("C", product_of=Reaction(reactants=[molecule("[H][H]")])), molecule("CO")]
            ),
        )
    )

    result = check_task_constraints(deep_route, [RouteDepthConstraint(max_depth=1)], [RouteDepthChecker()])

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.route_depth.exceeded"


def test_route_depth_named_constraints_use_explicit_ranges() -> None:
    checkers = [RouteDepthChecker()]

    assert (
        check_task_constraints(route_with_depth(3), [RouteDepthConstraint(max_depth="short")], checkers).status
        == CheckStatus.PASS
    )
    assert (
        check_task_constraints(route_with_depth(4), [RouteDepthConstraint(max_depth="short")], checkers).status
        == CheckStatus.FAIL
    )
    assert (
        check_task_constraints(route_with_depth(4), [RouteDepthConstraint(max_depth="medium")], checkers).status
        == CheckStatus.PASS
    )
    assert (
        check_task_constraints(route_with_depth(7), [RouteDepthConstraint(max_depth="medium")], checkers).status
        == CheckStatus.FAIL
    )
    assert (
        check_task_constraints(route_with_depth(7), [RouteDepthConstraint(max_depth="long")], checkers).status
        == CheckStatus.PASS
    )
    assert (
        check_task_constraints(route_with_depth(6), [RouteDepthConstraint(max_depth="long")], checkers).status
        == CheckStatus.FAIL
    )


def test_unknown_constraint_kind_fails_closed() -> None:
    with pytest.raises(UnsupportedTaskConstraintError) as exc_info:
        check_task_constraints(
            route(), [TaskConstraint(kind="ariadne.reaction_count", max_count=5)], [RouteDepthChecker()]
        )

    assert exc_info.value.code == "constraint.unsupported"
    assert exc_info.value.context == {"kind": "ariadne.reaction_count"}
