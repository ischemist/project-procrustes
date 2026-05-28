from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from retrocast.chem import canonicalize_smiles, get_inchi_key, reduce_inchikey
from retrocast.exceptions import ChemError
from retrocast.typing import InChIKeyStr
from retrocast.v2.models.candidates import Candidate
from retrocast.v2.models.evaluation import (
    CheckResult,
    CheckStatus,
    ConstraintResult,
    Evaluation,
    ReactionValidity,
    RouteValidity,
    ScoredCandidate,
    TargetResult,
    Tier,
    TierResult,
)
from retrocast.v2.models.route import InChIKeyLevel, Molecule, ReactionView, Route, RoutePath
from retrocast.v2.models.task import Target, Task, TaskConstraints


class TierChecker(Protocol):
    tier: Tier
    name: str

    def check_route(self, route: Route) -> RouteValidity: ...


class ConstraintChecker(Protocol):
    name: str

    def check_route(self, route: Route, constraints: TaskConstraints) -> ConstraintResult: ...


class TierZeroChecker:
    tier = Tier.ZERO
    name = "tier-zero"

    def check_route(self, route: Route) -> RouteValidity:
        route_checks: list[CheckResult] = []
        reaction_validity = []

        for molecule in _iter_molecules(route.target):
            try:
                expected_inchikey = get_inchi_key(molecule.smiles)
            except ChemError as exc:
                route_checks.append(
                    CheckResult(
                        code="tier0.invalid_smiles",
                        status=CheckStatus.FAIL,
                        message=str(exc),
                        details={"smiles": molecule.smiles},
                    )
                )
                continue
            if molecule.inchikey != expected_inchikey:
                route_checks.append(
                    CheckResult(
                        code="tier0.inchikey_mismatch",
                        status=CheckStatus.FAIL,
                        details={
                            "smiles": molecule.smiles,
                            "actual_inchikey": molecule.inchikey,
                            "expected_inchikey": expected_inchikey,
                        },
                    )
                )

        for reaction in _iter_reactions(route):
            checks = []
            if not reaction.value.reactants:
                checks.append(CheckResult(code="tier0.empty_reactants", status=CheckStatus.FAIL))
            reaction_validity.append(
                ReactionValidity(
                    reaction_id=reaction.id(),
                    tiers={Tier.ZERO: TierResult(status=_checks_status(checks), checks=checks)},
                )
            )
            route_checks.extend(checks)

        return RouteValidity(
            tiers={Tier.ZERO: TierResult(status=_checks_status(route_checks), checks=route_checks)},
            reactions=reaction_validity,
        )


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


def score_candidate(
    candidate: Candidate,
    *,
    target: Target,
    constraints: TaskConstraints,
    tier_checkers: Sequence[TierChecker],
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> ScoredCandidate:
    """Score one candidate while preserving failed adaptation slots."""
    if candidate.failure is not None:
        tiers = {checker.tier: _failed_tier_result(candidate) for checker in tier_checkers}
        return ScoredCandidate(
            rank=candidate.rank,
            failure=candidate.failure,
            validity=RouteValidity(tiers=tiers),
            constraints=ConstraintResult(status=CheckStatus.NOT_EVALUATED),
        )

    route = candidate.route
    if route is None:
        raise ValueError("Candidate requires route or failure.")

    validity = _check_route_validity(route, tier_checkers)
    constraints_result = constraint_checker.check_route(route, constraints)
    matched_index = _acceptable_match_index(route, target.acceptable_routes, acceptable_match_level)
    return ScoredCandidate(
        rank=candidate.rank,
        route=route,
        validity=validity,
        constraints=constraints_result,
        matches_acceptable=matched_index is not None,
        matched_acceptable_index=matched_index,
    )


def score_target(
    candidates: Sequence[Candidate],
    *,
    target: Target,
    constraints: TaskConstraints,
    tier_checkers: Sequence[TierChecker],
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> TargetResult:
    scored_candidates = []
    for candidate in candidates:
        scored_candidates.append(
            score_candidate(
                candidate,
                target=target,
                constraints=constraints,
                tier_checkers=tier_checkers,
                constraint_checker=constraint_checker,
                acceptable_match_level=acceptable_match_level,
            )
        )

    return TargetResult(
        target=target,
        effective_constraints=constraints,
        candidates=scored_candidates,
    )


def score(
    predictions: Mapping[str, Sequence[Candidate]],
    task: Task,
    *,
    tier_checkers: Sequence[TierChecker],
    constraint_checker: ConstraintChecker,
    acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> Evaluation:
    tiers = sorted({checker.tier for checker in tier_checkers})
    return Evaluation(
        task=task,
        tiers=tiers,
        targets={
            target_id: score_target(
                predictions.get(target_id, []),
                target=target,
                constraints=task.constraints.get(target_id, task.default_constraints),
                tier_checkers=tier_checkers,
                constraint_checker=constraint_checker,
                acceptable_match_level=acceptable_match_level,
            )
            for target_id, target in task.targets.items()
        },
    )


def _failed_tier_result(candidate: Candidate) -> TierResult:
    failure = candidate.failure
    return TierResult(
        status=CheckStatus.FAIL,
        checks=[
            CheckResult(
                code=failure.code if failure is not None else "adapter.unknown_failure",
                status=CheckStatus.FAIL,
                message=failure.message if failure is not None else None,
                details=failure.context if failure is not None else {},
            )
        ],
    )


def _check_route_validity(route: Route, tier_checkers: Sequence[TierChecker]) -> RouteValidity:
    validity = RouteValidity()
    reactions_by_id: dict[str, ReactionValidity] = {}
    for checker in tier_checkers:
        result = checker.check_route(route)
        validity.tiers.update(result.tiers)
        for reaction in result.reactions:
            existing = reactions_by_id.get(reaction.reaction_id)
            if existing is None:
                reactions_by_id[reaction.reaction_id] = reaction
            else:
                existing.tiers.update(reaction.tiers)
    validity.reactions = list(reactions_by_id.values())
    return validity


def _acceptable_match_index(
    route: Route,
    acceptable_routes: Sequence[Route],
    match_level: InChIKeyLevel,
) -> int | None:
    route_signature = route.signature(match_level)
    for index, acceptable_route in enumerate(acceptable_routes):
        if route_signature == acceptable_route.signature(match_level):
            return index
    return None


def _checks_status(checks: Sequence[CheckResult]) -> CheckStatus:
    return CheckStatus.FAIL if checks else CheckStatus.PASS


def _iter_molecules(molecule: Molecule) -> Sequence[Molecule]:
    molecules = [molecule]
    if molecule.product_of is not None:
        for reactant in molecule.product_of.reactants:
            molecules.extend(_iter_molecules(reactant))
    return molecules


def _iter_reactions(route: Route) -> Sequence[ReactionView]:
    reactions = []

    def visit(path: RoutePath) -> None:
        molecule = route.molecule_at(path)
        reaction = molecule.produced_by()
        if reaction is None:
            return
        reactions.append(reaction)
        for index, _ in enumerate(reaction.value.reactants):
            visit(reaction.path.reactant(index))

    visit(RoutePath.target())
    return reactions


def _leaf_molecules(route: Route) -> Sequence[Molecule]:
    return [molecule for molecule in _iter_molecules(route.target) if molecule.product_of is None]


def _reduce_inchikeys(inchikeys: set[InChIKeyStr], match_level: InChIKeyLevel) -> set[str]:
    if match_level == InChIKeyLevel.FULL:
        return set(inchikeys)
    return {str(reduce_inchikey(inchikey, match_level)) for inchikey in inchikeys}


def _missing_stock_leaves(route: Route, stock_keys: set[str], match_level: InChIKeyLevel) -> list[str]:
    missing = []
    for leaf in _leaf_molecules(route):
        leaf_key = leaf.key(match_level)
        if leaf_key not in stock_keys:
            missing.append(leaf.inchikey)
    return sorted(set(missing))


def _missing_required_leaves(
    route: Route,
    required_leaves_smiles: Sequence[str],
    match_level: InChIKeyLevel,
) -> list[str]:
    leaf_keys = {leaf.key(match_level) for leaf in _leaf_molecules(route)}
    missing = []
    for smiles in required_leaves_smiles:
        required_key = get_inchi_key(canonicalize_smiles(smiles), level=match_level)
        if required_key not in leaf_keys:
            missing.append(smiles)
    return missing


def _route_depth(route: Route) -> int:
    reactions = _iter_reactions(route)
    if not reactions:
        return 0
    return max(reaction.path.depth() + 1 for reaction in reactions)
