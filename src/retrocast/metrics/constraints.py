from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from retrocast.chem import get_inchi_key, reduce_inchikey
from retrocast.exceptions import UnsupportedTaskConstraintError
from retrocast.models.evaluation import CheckResult, CheckStatus, ConstraintResult
from retrocast.models.route import InChIKeyLevel, Route
from retrocast.models.task import (
    REQUIRED_LEAVES,
    ROUTE_DEPTH,
    STOCK_TERMINATION,
    RequiredLeavesConstraint,
    RouteDepthConstraint,
    StockTerminationConstraint,
    TaskConstraint,
)
from retrocast.typing import InChIKeyStr

ROUTE_DEPTH_RANGES = {
    "short": (1, 3),
    "medium": (4, 6),
    "long": (7, None),
}


class TaskConstraintChecker(Protocol):
    kind: str

    def check_route(self, route: Route, constraint: TaskConstraint) -> list[CheckResult]: ...


class StockTerminationChecker:
    kind = STOCK_TERMINATION

    def __init__(
        self,
        *,
        stocks: Mapping[str, set[InChIKeyStr]],
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    ) -> None:
        self.match_level = match_level
        self._stock_keys_by_name = {
            stock_name: {str(reduce_inchikey(inchikey, match_level)) for inchikey in stock_values}
            for stock_name, stock_values in stocks.items()
        }

    def check_route(self, route: Route, constraint: TaskConstraint) -> list[CheckResult]:
        parsed = cast(StockTerminationConstraint, constraint)
        stock_keys = self._stock_keys_by_name.get(parsed.stock, set())
        if not stock_keys:
            return [
                CheckResult(
                    code="constraint.stock_termination.unregistered_stock",
                    status=CheckStatus.FAIL,
                    details={"stock": parsed.stock},
                )
            ]

        missing_leaves = sorted(
            {leaf.value.inchikey for leaf in route.iter_leaves() if leaf.key(self.match_level) not in stock_keys}
        )
        if not missing_leaves:
            return []
        return [
            CheckResult(
                code="constraint.stock_termination.missing_leaf",
                status=CheckStatus.FAIL,
                details={"missing_leaf_inchikeys": missing_leaves},
            )
        ]


class RequiredLeavesChecker:
    kind = REQUIRED_LEAVES

    def __init__(self, *, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> None:
        self.match_level = match_level

    def check_route(self, route: Route, constraint: TaskConstraint) -> list[CheckResult]:
        parsed = cast(RequiredLeavesConstraint, constraint)
        required_leaf_inchikeys = [InChIKeyStr(get_inchi_key(smiles)) for smiles in parsed.smiles]
        leaf_keys = {leaf.key(self.match_level) for leaf in route.iter_leaves()}
        missing_required_leaves = [
            inchikey
            for inchikey in required_leaf_inchikeys
            if str(reduce_inchikey(inchikey, self.match_level)) not in leaf_keys
        ]
        if not missing_required_leaves:
            return []
        return [
            CheckResult(
                code="constraint.required_leaf.missing",
                status=CheckStatus.FAIL,
                details={"missing_leaf_inchikeys": missing_required_leaves},
            )
        ]


class RouteDepthChecker:
    kind = ROUTE_DEPTH

    def check_route(self, route: Route, constraint: TaskConstraint) -> list[CheckResult]:
        parsed = cast(RouteDepthConstraint, constraint)
        route_depth = route.depth()
        if isinstance(parsed.max_depth, int):
            if route_depth <= parsed.max_depth:
                return []
            return [
                CheckResult(
                    code="constraint.route_depth.exceeded",
                    status=CheckStatus.FAIL,
                    details={"route_depth": route_depth, "max_depth": parsed.max_depth},
                )
            ]
        min_depth, max_depth = ROUTE_DEPTH_RANGES[parsed.max_depth]
        if route_depth >= min_depth and (max_depth is None or route_depth <= max_depth):
            return []
        return [
            CheckResult(
                code="constraint.route_depth.out_of_range",
                status=CheckStatus.FAIL,
                details={"route_depth": route_depth, "min_depth": min_depth, "max_depth": max_depth},
            )
        ]


def check_task_constraints(
    route: Route,
    constraints: Sequence[TaskConstraint],
    checkers: Sequence[TaskConstraintChecker],
) -> ConstraintResult:
    registry = {checker.kind: checker for checker in checkers}
    checks = []

    for constraint in constraints:
        checker = registry.get(constraint.kind)
        if checker is None:
            raise UnsupportedTaskConstraintError(
                f"No checker registered for task constraint kind '{constraint.kind}'.",
                context={"kind": constraint.kind},
            )
        checks.extend(checker.check_route(route, constraint))

    status = CheckStatus.FAIL if checks else CheckStatus.PASS
    return ConstraintResult(status=status, checks=checks)
