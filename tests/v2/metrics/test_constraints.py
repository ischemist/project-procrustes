from __future__ import annotations

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.v2.metrics.constraints import TaskConstraintChecker
from retrocast.v2.models import CheckStatus, Molecule, Reaction, Route, TaskConstraints


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
    result = TaskConstraintChecker(stock=stock_for("C", "CO"), stock_name="stock-a").check_route(
        route(), TaskConstraints(stock="stock-a")
    )

    assert result.status == CheckStatus.PASS


def test_stock_constraint_fails_when_leaf_is_missing() -> None:
    result = TaskConstraintChecker(stock=stock_for("C"), stock_name="stock-a").check_route(
        route(), TaskConstraints(stock="stock-a")
    )

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.stock_termination.missing_leaf"


def test_stock_constraint_fails_clearly_without_stock_keys() -> None:
    result = TaskConstraintChecker(stock_name="stock-a").check_route(route(), TaskConstraints(stock="stock-a"))

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.stock_termination.no_stock_keys"


def test_required_leaf_constraint_fails_when_required_leaf_is_missing() -> None:
    result = TaskConstraintChecker().check_route(route(), TaskConstraints(required_leaves_smiles=[SmilesStr("CC")]))

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

    result = TaskConstraintChecker().check_route(deep_route, TaskConstraints(route_depth=1))

    assert result.status == CheckStatus.FAIL
    assert result.checks[0].code == "constraint.route_depth.exceeded"


def test_route_depth_named_constraints_use_explicit_ranges() -> None:
    checker = TaskConstraintChecker()

    assert checker.check_route(route_with_depth(3), TaskConstraints(route_depth="short")).status == CheckStatus.PASS
    assert checker.check_route(route_with_depth(4), TaskConstraints(route_depth="short")).status == CheckStatus.FAIL
    assert checker.check_route(route_with_depth(4), TaskConstraints(route_depth="medium")).status == CheckStatus.PASS
    assert checker.check_route(route_with_depth(7), TaskConstraints(route_depth="medium")).status == CheckStatus.FAIL
    assert checker.check_route(route_with_depth(7), TaskConstraints(route_depth="long")).status == CheckStatus.PASS
    assert checker.check_route(route_with_depth(6), TaskConstraints(route_depth="long")).status == CheckStatus.FAIL


def test_stock_only_checker_ignores_other_task_constraints() -> None:
    result = TaskConstraintChecker.stock_termination(stock=stock_for("C", "CO"), stock_name="stock-a").check_route(
        route(),
        TaskConstraints(stock="stock-a", required_leaves_smiles=[SmilesStr("CC")], route_depth=0),
    )

    assert result.status == CheckStatus.PASS
