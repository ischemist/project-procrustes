from __future__ import annotations

from collections.abc import Sequence

from retrocast.chem import get_inchi_key
from retrocast.exceptions import ChemError
from retrocast.v2.metrics._route import iter_molecules, iter_reactions
from retrocast.v2.models.evaluation import CheckResult, CheckStatus, ReactionValidity, RouteValidity, Tier, TierResult
from retrocast.v2.models.route import Route


class TierZeroChecker:
    tier = Tier.ZERO
    name = "tier-zero"

    def check_route(self, route: Route) -> RouteValidity:
        route_checks: list[CheckResult] = []
        reaction_validity = []

        for molecule in iter_molecules(route.target):
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

        for reaction in iter_reactions(route):
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


def _checks_status(checks: Sequence[CheckResult]) -> CheckStatus:
    return CheckStatus.FAIL if checks else CheckStatus.PASS
