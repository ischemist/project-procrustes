from __future__ import annotations

from collections.abc import Sequence

from retrocast.chem import canonicalize_smiles, get_inchi_key, reduce_inchikey
from retrocast.typing import InChIKeyStr
from retrocast.v2.models.evaluation import CheckResult, CheckStatus, ConstraintResult
from retrocast.v2.models.route import InChIKeyLevel, Molecule, Route
from retrocast.v2.models.task import TaskConstraints


class TaskConstraintChecker:
    name = "task-constraints"

    def __init__(
        self,
        *,
        stock: set[InChIKeyStr] | None = None,
        stock_name: str | None = None,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    ) -> None:
        self.stock = stock or set()
        self.stock_name = stock_name
        self.match_level = match_level
        self._stock_keys = _reduce_inchikeys(self.stock, match_level)

    def check_route(self, route: Route, constraints: TaskConstraints) -> ConstraintResult:
        checks = []

        if constraints.stock is not None:
            if self.stock_name is not None and constraints.stock != self.stock_name:
                checks.append(
                    CheckResult(
                        code="constraint.stock_termination.stock_mismatch",
                        status=CheckStatus.FAIL,
                        details={"expected_stock": constraints.stock, "checker_stock": self.stock_name},
                    )
                )
            missing_leaves = _missing_stock_leaves(route, self._stock_keys, self.match_level)
            if missing_leaves:
                checks.append(
                    CheckResult(
                        code="constraint.stock_termination.missing_leaf",
                        status=CheckStatus.FAIL,
                        details={"missing_leaf_inchikeys": missing_leaves},
                    )
                )

        if constraints.required_leaves_smiles:
            missing_required_leaves = _missing_required_leaves(
                route, constraints.required_leaves_smiles, self.match_level
            )
            if missing_required_leaves:
                checks.append(
                    CheckResult(
                        code="constraint.required_leaf.missing",
                        status=CheckStatus.FAIL,
                        details={"missing_leaf_smiles": missing_required_leaves},
                    )
                )

        if isinstance(constraints.route_depth, int):
            route_depth = _route_depth(route)
            if route_depth > constraints.route_depth:
                checks.append(
                    CheckResult(
                        code="constraint.route_depth.exceeded",
                        status=CheckStatus.FAIL,
                        details={"route_depth": route_depth, "max_depth": constraints.route_depth},
                    )
                )

        return ConstraintResult(status=_checks_status(checks), checks=checks)


def _checks_status(checks: Sequence[CheckResult]) -> CheckStatus:
    return CheckStatus.FAIL if checks else CheckStatus.PASS


def _reduce_inchikeys(inchikeys: set[InChIKeyStr], match_level: InChIKeyLevel) -> set[str]:
    if match_level == InChIKeyLevel.FULL:
        return set(inchikeys)
    return {str(reduce_inchikey(inchikey, match_level)) for inchikey in inchikeys}


def _missing_stock_leaves(route: Route, stock_keys: set[str], match_level: InChIKeyLevel) -> list[str]:
    missing = []
    for leaf in _leaf_molecules(route.target):
        leaf_key = leaf.key(match_level)
        if leaf_key not in stock_keys:
            missing.append(leaf.inchikey)
    return sorted(set(missing))


def _missing_required_leaves(
    route: Route,
    required_leaves_smiles: Sequence[str],
    match_level: InChIKeyLevel,
) -> list[str]:
    leaf_keys = {leaf.key(match_level) for leaf in _leaf_molecules(route.target)}
    missing = []
    for smiles in required_leaves_smiles:
        required_key = get_inchi_key(canonicalize_smiles(smiles), level=match_level)
        if required_key not in leaf_keys:
            missing.append(smiles)
    return missing


def _route_depth(route: Route) -> int:
    return _molecule_depth(route.target)


def _leaf_molecules(molecule: Molecule) -> list[Molecule]:
    reaction = molecule.product_of
    if reaction is None:
        return [molecule]

    leaves = []
    for reactant in reaction.reactants:
        leaves.extend(_leaf_molecules(reactant))
    return leaves


def _molecule_depth(molecule: Molecule) -> int:
    reaction = molecule.product_of
    if reaction is None:
        return 0
    if not reaction.reactants:
        return 1
    return 1 + max(_molecule_depth(reactant) for reactant in reaction.reactants)
