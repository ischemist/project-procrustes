from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from retrocast.chem import canonicalize_smiles, get_inchi_key, reduce_inchikey
from retrocast.typing import InChIKeyStr
from retrocast.v2.models.evaluation import CheckResult, CheckStatus, ConstraintResult
from retrocast.v2.models.route import InChIKeyLevel, Molecule, Route
from retrocast.v2.models.task import TaskConstraints


class ConstraintCheck(Protocol):
    name: str

    def check_route(self, route: Route, constraints: TaskConstraints) -> list[CheckResult]: ...


class TaskConstraintChecker:
    name = "task-constraints"

    def __init__(
        self,
        *,
        stock: set[InChIKeyStr] | None = None,
        stock_name: str | None = None,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        checks: Sequence[ConstraintCheck] | None = None,
    ) -> None:
        self.checks = (
            list(checks)
            if checks is not None
            else [
                StockTerminationCheck(stock=stock, stock_name=stock_name, match_level=match_level),
                RequiredLeavesCheck(match_level=match_level),
                RouteDepthCheck(),
            ]
        )

    @classmethod
    def stock_termination(
        cls,
        *,
        stock: set[InChIKeyStr] | None = None,
        stock_name: str | None = None,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    ) -> TaskConstraintChecker:
        return cls(checks=[StockTerminationCheck(stock=stock, stock_name=stock_name, match_level=match_level)])

    def check_route(self, route: Route, constraints: TaskConstraints) -> ConstraintResult:
        checks = []
        for check in self.checks:
            checks.extend(check.check_route(route, constraints))
        return ConstraintResult(status=_checks_status(checks), checks=checks)


class StockTerminationCheck:
    name = "stock-termination"

    def __init__(
        self,
        *,
        stock: set[InChIKeyStr] | None = None,
        stock_name: str | None = None,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    ) -> None:
        self.stock_name = stock_name
        self.match_level = match_level
        self._stock_keys = _reduce_inchikeys(stock or set(), match_level)

    def check_route(self, route: Route, constraints: TaskConstraints) -> list[CheckResult]:
        if constraints.stock is None:
            return []
        return self._check_stock(route, constraints.stock)

    def _check_stock(self, route: Route, expected_stock: str) -> list[CheckResult]:
        checks = []
        if self.stock_name is not None and expected_stock != self.stock_name:
            checks.append(
                CheckResult(
                    code="constraint.stock_termination.stock_mismatch",
                    status=CheckStatus.FAIL,
                    details={"expected_stock": expected_stock, "checker_stock": self.stock_name},
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
        return checks


class RequiredLeavesCheck:
    name = "required-leaves"

    def __init__(self, *, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> None:
        self.match_level = match_level

    def check_route(self, route: Route, constraints: TaskConstraints) -> list[CheckResult]:
        if not constraints.required_leaves_smiles:
            return []
        return self._check_required_leaves(route, constraints.required_leaves_smiles)

    def _check_required_leaves(self, route: Route, required_leaves_smiles: Sequence[str]) -> list[CheckResult]:
        missing_required_leaves = _missing_required_leaves(route, required_leaves_smiles, self.match_level)
        if not missing_required_leaves:
            return []
        return [
            CheckResult(
                code="constraint.required_leaf.missing",
                status=CheckStatus.FAIL,
                details={"missing_leaf_smiles": missing_required_leaves},
            )
        ]


class RouteDepthCheck:
    name = "route-depth"

    def check_route(self, route: Route, constraints: TaskConstraints) -> list[CheckResult]:
        if not isinstance(constraints.route_depth, int):
            return []
        return self._check_route_depth(route, constraints.route_depth)

    def _check_route_depth(self, route: Route, max_depth: int) -> list[CheckResult]:
        route_depth = _route_depth(route)
        if route_depth <= max_depth:
            return []
        return [
            CheckResult(
                code="constraint.route_depth.exceeded",
                status=CheckStatus.FAIL,
                details={"route_depth": route_depth, "max_depth": max_depth},
            )
        ]


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
