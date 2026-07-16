import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from retrocast.chem import InChIKeyLevel
from retrocast.exceptions import UnsupportedTaskConstraintError
from retrocast.models.evaluation import CheckResult, CheckStatus, ConstraintResult
from retrocast.models.route import Route
from retrocast.models.task import (
    REQUIRED_LEAVES,
    ROUTE_DEPTH,
    STOCK_TERMINATION,
    TaskConstraint,
)
from retrocast.typing import InChIKeyStr


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
            stock_name: {str(inchikey) for inchikey in stock_values} for stock_name, stock_values in stocks.items()
        }

    @property
    def stock_keys_by_name(self) -> Mapping[str, set[str]]:
        return self._stock_keys_by_name

    def check_route(self, route: Route, constraint: TaskConstraint) -> list[CheckResult]:
        return _native_check(
            route,
            constraint,
            stocks=self._stock_keys_by_name,
            match_level=self.match_level,
        ).checks


class RequiredLeavesChecker:
    kind = REQUIRED_LEAVES

    def __init__(self, *, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> None:
        self.match_level = match_level

    def check_route(self, route: Route, constraint: TaskConstraint) -> list[CheckResult]:
        return _native_check(route, constraint, match_level=self.match_level).checks


class RouteDepthChecker:
    kind = ROUTE_DEPTH

    def check_route(self, route: Route, constraint: TaskConstraint) -> list[CheckResult]:
        return _native_check(route, constraint).checks


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
    return ConstraintResult(status=CheckStatus.FAIL if checks else CheckStatus.PASS, checks=checks)


def _native_check(
    route: Route,
    constraint: TaskConstraint,
    *,
    stocks: Mapping[str, set[str]] | None = None,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> ConstraintResult:
    from retrocast import _native

    payload = _native.check_task_constraints_json(
        route.model_dump_json(exclude_none=True),
        json.dumps([constraint.model_dump(mode="json")], separators=(",", ":")),
        json.dumps(
            {name: sorted(values) for name, values in (stocks or {}).items()},
            separators=(",", ":"),
        ),
        match_level=match_level.value,
    )
    return ConstraintResult.model_validate_json(payload)
