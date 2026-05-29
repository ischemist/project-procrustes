from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from retrocast.chem import reduce_inchikey
from retrocast.typing import InChIKeyStr
from retrocast.v2.models.evaluation import CheckResult, CheckStatus, ConstraintResult
from retrocast.v2.models.route import InChIKeyLevel, Route
from retrocast.v2.models.task import TaskConstraints

ROUTE_DEPTH_RANGES = {
    "short": (1, 3),
    "medium": (4, 6),
    "long": (7, None),
}


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
        status = CheckStatus.FAIL if checks else CheckStatus.PASS
        return ConstraintResult(status=status, checks=checks)


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
        self._stock_keys = {str(reduce_inchikey(inchikey, match_level)) for inchikey in stock or set()}

    def check_route(self, route: Route, constraints: TaskConstraints) -> list[CheckResult]:
        if constraints.stock is None:
            return []
        if not self._stock_keys:
            return [
                CheckResult(
                    code="constraint.stock_termination.no_stock_keys",
                    status=CheckStatus.FAIL,
                    details={"stock": constraints.stock},
                )
            ]
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

        missing_leaves = []
        for leaf in route.iter_leaves():
            if leaf.key(self.match_level) not in self._stock_keys:
                missing_leaves.append(leaf.value.inchikey)
        missing_leaves = sorted(set(missing_leaves))
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
        if not constraints.required_leaf_inchikeys:
            return []
        return self._check_required_leaves(route, constraints.required_leaf_inchikeys)

    def _check_required_leaves(self, route: Route, required_leaves: Sequence[InChIKeyStr]) -> list[CheckResult]:
        leaf_keys = {leaf.key(self.match_level) for leaf in route.iter_leaves()}
        missing_required_leaves = []
        for inchikey in required_leaves:
            required_key = str(reduce_inchikey(inchikey, self.match_level))
            if required_key not in leaf_keys:
                missing_required_leaves.append(inchikey)
        if not missing_required_leaves:
            return []
        return [
            CheckResult(
                code="constraint.required_leaf.missing",
                status=CheckStatus.FAIL,
                details={"missing_leaf_inchikeys": missing_required_leaves},
            )
        ]


class RouteDepthCheck:
    name = "route-depth"

    def check_route(self, route: Route, constraints: TaskConstraints) -> list[CheckResult]:
        if constraints.route_depth is None:
            return []
        if isinstance(constraints.route_depth, int):
            return self._check_max_depth(route, constraints.route_depth)
        min_depth, max_depth = ROUTE_DEPTH_RANGES[constraints.route_depth]
        return self._check_depth_range(route, min_depth=min_depth, max_depth=max_depth)

    def _check_max_depth(self, route: Route, max_depth: int) -> list[CheckResult]:
        route_depth = route.depth()
        if route_depth <= max_depth:
            return []
        return [
            CheckResult(
                code="constraint.route_depth.exceeded",
                status=CheckStatus.FAIL,
                details={"route_depth": route_depth, "max_depth": max_depth},
            )
        ]

    def _check_depth_range(self, route: Route, *, min_depth: int, max_depth: int | None) -> list[CheckResult]:
        route_depth = route.depth()
        if route_depth >= min_depth and (max_depth is None or route_depth <= max_depth):
            return []
        return [
            CheckResult(
                code="constraint.route_depth.out_of_range",
                status=CheckStatus.FAIL,
                details={"route_depth": route_depth, "min_depth": min_depth, "max_depth": max_depth},
            )
        ]
