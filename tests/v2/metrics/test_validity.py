from __future__ import annotations

from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.typing import InChIKeyStr, SmilesStr
from retrocast.v2.metrics.validity import TierZeroChecker
from retrocast.v2.models import CheckStatus, Molecule, Reaction, Route, Tier


def molecule(smiles: str, *, product_of: Reaction | None = None) -> Molecule:
    canonical = canonicalize_smiles(smiles)
    return Molecule(
        smiles=SmilesStr(canonical),
        inchikey=InChIKeyStr(get_inchi_key(canonical)),
        product_of=product_of,
    )


def route() -> Route:
    return Route(target=molecule("CCO", product_of=Reaction(reactants=[molecule("C"), molecule("CO")])))


def test_tier_zero_passes_valid_route() -> None:
    validity = TierZeroChecker().check_route(route())

    assert validity.tiers[Tier.ZERO].status == CheckStatus.PASS
    assert validity.reactions[0].reaction_id == "rc:r:/"
    assert validity.reactions[0].tiers[Tier.ZERO].status == CheckStatus.PASS


def test_empty_reaction_fails_tier_zero() -> None:
    invalid_route = Route(target=molecule("CCO", product_of=Reaction(reactants=[])))

    validity = TierZeroChecker().check_route(invalid_route)

    assert validity.tiers[Tier.ZERO].status == CheckStatus.FAIL
    assert validity.reactions[0].tiers[Tier.ZERO].status == CheckStatus.FAIL
    assert validity.reactions[0].tiers[Tier.ZERO].checks[0].code == "tier0.empty_reactants"


def test_inchikey_mismatch_fails_tier_zero() -> None:
    invalid_route = Route(target=Molecule(smiles=SmilesStr("CCO"), inchikey=InChIKeyStr(get_inchi_key("C"))))

    validity = TierZeroChecker().check_route(invalid_route)

    assert validity.tiers[Tier.ZERO].status == CheckStatus.FAIL
    assert validity.tiers[Tier.ZERO].checks[0].code == "tier0.inchikey_mismatch"
