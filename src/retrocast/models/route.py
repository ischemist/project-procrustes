from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator

from retrocast.chem import InChIKeyLevel, reduce_inchikey
from retrocast.hashing import hash_json
from retrocast.typing import InChIKeyStr, ReactionSmilesStr, SmilesStr

RouteNodeKind = Literal["m", "r"]


def _stable_hash(value: Any) -> str:
    return hash_json(value)


def _validate_depth(depth: int | None) -> None:
    if depth is not None and depth < 0:
        raise ValueError("depth must be non-negative")


@dataclass(frozen=True, slots=True)
class RoutePath:
    kind: RouteNodeKind
    indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("m", "r"):
            raise ValueError("route path kind must be 'm' or 'r'")
        if any(index < 0 for index in self.indices):
            raise ValueError("route path indices must be non-negative")

    @classmethod
    def parse(cls, value: str) -> RoutePath:
        prefix = "rc:"
        if not value.startswith(prefix):
            raise ValueError("route path must start with 'rc:'")

        try:
            kind, rest = value[len(prefix) :].split(":", 1)
        except ValueError as exc:
            raise ValueError("route path must have form 'rc:<kind>:/...'") from exc

        if kind == "m":
            route_kind: RouteNodeKind = "m"
        elif kind == "r":
            route_kind = "r"
        else:
            raise ValueError("route path kind must be 'm' or 'r'")
        if not rest.startswith("/"):
            raise ValueError("route path indices must start with '/'")

        tail = rest[1:]
        if tail == "":
            return cls(kind=route_kind, indices=())

        parts = tail.split("/")
        try:
            indices = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("route path indices must be integers") from exc
        if any(index < 0 for index in indices):
            raise ValueError("route path indices must be non-negative")
        if any(str(index) != part for index, part in zip(indices, parts, strict=True)):
            raise ValueError("route path indices must be canonical non-negative integers")
        return cls(kind=route_kind, indices=indices)

    @classmethod
    @cache
    def target(cls) -> RoutePath:
        return cls(kind="m")

    @classmethod
    @cache
    def root_reaction(cls) -> RoutePath:
        return cls(kind="r")

    def id(self) -> str:
        suffix = "/".join(str(index) for index in self.indices)
        return f"rc:{self.kind}:/{suffix}" if suffix else f"rc:{self.kind}:/"

    def depth(self) -> int:
        return len(self.indices)

    def is_molecule(self) -> bool:
        return self.kind == "m"

    def is_reaction(self) -> bool:
        return self.kind == "r"

    def produced_by(self) -> RoutePath:
        if not self.is_molecule():
            raise ValueError("only molecule paths have a producing reaction")
        return RoutePath(kind="r", indices=self.indices)

    def product(self) -> RoutePath:
        if not self.is_reaction():
            raise ValueError("only reaction paths have a product molecule")
        return RoutePath(kind="m", indices=self.indices)

    def reactant(self, index: int) -> RoutePath:
        if not self.is_reaction():
            raise ValueError("only reaction paths have reactants")
        if index < 0:
            raise ValueError("reactant index must be non-negative")
        return RoutePath(kind="m", indices=(*self.indices, index))


def validate_reaction_id(value: str) -> str:
    path = RoutePath.parse(value)
    if not path.is_reaction():
        raise ValueError("reaction id must identify a reaction node, e.g. 'rc:r:/1/0'")
    return path.id()


def validate_molecule_id(value: str) -> str:
    path = RoutePath.parse(value)
    if not path.is_molecule():
        raise ValueError("molecule id must identify a molecule node, e.g. 'rc:m:/1/0'")
    return path.id()


ReactionId = Annotated[str, AfterValidator(validate_reaction_id)]
MoleculeId = Annotated[str, AfterValidator(validate_molecule_id)]


class Reaction(BaseModel):
    reactants: list[Molecule]
    mapped_reaction_smiles: ReactionSmilesStr | None = None
    template: str | None = None
    reagents: list[SmilesStr] | None = None
    solvents: list[SmilesStr] | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_reactant_order(self) -> Reaction:
        self.reactants = sorted(self.reactants, key=_reactant_order_key)
        return self


class Molecule(BaseModel):
    smiles: SmilesStr
    inchikey: InChIKeyStr
    product_of: Reaction | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)

    def key(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> str:
        return str(reduce_inchikey(self.inchikey, match_level))

    def signature(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> str:
        return _stable_hash(self.key(match_level))


def _reactant_order_key(molecule: Molecule) -> tuple[tuple[Any, ...], str]:
    return (
        _molecule_subtree_key(molecule),
        _reactant_order_tiebreaker(molecule),
    )


def _reactant_order_tiebreaker(molecule: Molecule) -> str:
    return json.dumps(molecule.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _molecule_subtree_key(
    molecule: Molecule,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    *,
    depth: int | None = None,
) -> tuple[Any, ...]:
    """Compute subtree identity from raw molecules before a route-bound view exists."""
    _validate_depth(depth)
    reaction = molecule.product_of
    if reaction is None or depth == 0:
        return ("mol", molecule.key(match_level))

    next_depth = None if depth is None else depth - 1
    child_signatures = sorted(
        _stable_hash(_molecule_subtree_key(reactant, match_level, depth=next_depth)) for reactant in reaction.reactants
    )
    return (
        "mol",
        molecule.key(match_level),
        _reaction_key(molecule, reaction, match_level),
        tuple(child_signatures),
    )


def _reaction_key(
    product: Molecule,
    reaction: Reaction,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> tuple[str, str, tuple[str, ...]]:
    return (
        "rxn",
        product.key(match_level),
        tuple(sorted(reactant.key(match_level) for reactant in reaction.reactants)),
    )


class Route(BaseModel):
    target: Molecule
    annotations: dict[str, Any] = Field(default_factory=dict)
    schema_version: Literal["2"] = "2"

    def molecule_at(self, path: RoutePath | str) -> MoleculeView:
        route_path = RoutePath.parse(path) if isinstance(path, str) else path
        if not route_path.is_molecule():
            raise ValueError("path must identify a molecule")

        molecule = self.target
        for index in route_path.indices:
            reaction = molecule.product_of
            if reaction is None:
                raise KeyError(route_path.id())
            try:
                molecule = reaction.reactants[index]
            except IndexError as exc:
                raise KeyError(route_path.id()) from exc
        return MoleculeView(route=self, path=route_path, value=molecule)

    def reaction_at(self, path: RoutePath | str) -> ReactionView:
        route_path = RoutePath.parse(path) if isinstance(path, str) else path
        if not route_path.is_reaction():
            raise ValueError("path must identify a reaction")

        try:
            product = self.molecule_at(route_path.product())
        except KeyError as exc:
            raise KeyError(route_path.id()) from exc
        if product.value.product_of is None:
            raise KeyError(route_path.id())
        return ReactionView(route=self, path=route_path, value=product.value.product_of)

    def key(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        depth: int | None = None,
    ) -> tuple[Any, ...]:
        _validate_depth(depth)
        return self.molecule_at(RoutePath.target()).subtree_key(match_level, depth=depth)

    def signature(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        depth: int | None = None,
    ) -> str:
        return _stable_hash(self.key(match_level, depth=depth))

    def leaves(self) -> list[MoleculeView]:
        return list(self.iter_leaves())

    def iter_leaves(self) -> Iterator[MoleculeView]:
        yield from self.molecule_at(RoutePath.target()).iter_leaves()

    def iter_molecules(self) -> Iterator[MoleculeView]:
        yield from self.molecule_at(RoutePath.target()).iter_molecules()

    def find_molecules(
        self,
        molecule: Molecule | MoleculeView,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    ) -> tuple[MoleculeView, ...]:
        molecule_key = molecule.key(match_level)
        return tuple(candidate for candidate in self.iter_molecules() if candidate.key(match_level) == molecule_key)

    def contains_molecule(
        self,
        molecule: Molecule | MoleculeView,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    ) -> bool:
        molecule_key = molecule.key(match_level)
        return any(candidate.key(match_level) == molecule_key for candidate in self.iter_molecules())

    def reactions(self) -> list[ReactionView]:
        return list(self.iter_reactions())

    def iter_reactions(self) -> Iterator[ReactionView]:
        stack = [RoutePath.root_reaction()]
        while stack:
            path = stack.pop()
            try:
                reaction = self.reaction_at(path)
            except KeyError:
                continue
            yield reaction
            stack.extend(
                reactant.path.produced_by()
                for reactant in reaction.reactants()
                if reactant.value.product_of is not None
            )

    def reaction_signatures(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> set[str]:
        return {reaction.signature(match_level) for reaction in self.iter_reactions()}

    def depth(self) -> int:
        return self.molecule_at(RoutePath.target()).depth()

    def is_convergent(self) -> bool:
        """Return whether any reaction joins multiple synthesized branches."""
        stack = [self.target]
        while stack:
            molecule = stack.pop()
            reaction = molecule.product_of
            if reaction is None:
                continue
            synthesized_reactants = [reactant for reactant in reaction.reactants if reactant.product_of is not None]
            if len(synthesized_reactants) > 1:
                return True
            stack.extend(reaction.reactants)
        return False


class ReactionView(BaseModel):
    route: Route
    path: RoutePath
    value: Reaction

    def id(self) -> ReactionId:
        return self.path.id()

    def product(self) -> MoleculeView:
        return self.route.molecule_at(self.path.product())

    def reactants(self) -> list[MoleculeView]:
        return [
            MoleculeView(route=self.route, path=self.path.reactant(index), value=reactant)
            for index, reactant in enumerate(self.value.reactants)
        ]

    def key(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> tuple[str, str, tuple[str, ...]]:
        return _reaction_key(self.product().value, self.value, match_level)

    def signature(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> str:
        return _stable_hash(self.key(match_level))


class MoleculeView(BaseModel):
    route: Route
    path: RoutePath
    value: Molecule

    def id(self) -> MoleculeId:
        return self.path.id()

    def key(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> str:
        return self.value.key(match_level)

    def produced_by(self) -> ReactionView | None:
        if self.value.product_of is None:
            return None
        return ReactionView(route=self.route, path=self.path.produced_by(), value=self.value.product_of)

    def leaves(self) -> list[MoleculeView]:
        return list(self.iter_leaves())

    def iter_leaves(self) -> Iterator[MoleculeView]:
        reaction = self.produced_by()
        if reaction is None:
            yield self
            return

        for reactant in reaction.reactants():
            yield from reactant.iter_leaves()

    def iter_molecules(self) -> Iterator[MoleculeView]:
        yield self
        reaction = self.produced_by()
        if reaction is None:
            return
        for reactant in reaction.reactants():
            yield from reactant.iter_molecules()

    def depth(self) -> int:
        max_depth = 0
        stack: list[tuple[MoleculeView, int]] = [(self, 0)]
        while stack:
            molecule, depth = stack.pop()
            reaction = molecule.produced_by()
            if reaction is None:
                max_depth = max(max_depth, depth)
                continue
            if not reaction.value.reactants:
                max_depth = max(max_depth, depth + 1)
                continue
            for reactant in reaction.reactants():
                stack.append((reactant, depth + 1))
        return max_depth

    def subtree_key(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        depth: int | None = None,
    ) -> tuple[Any, ...]:
        return _molecule_subtree_key(self.value, match_level, depth=depth)

    def subtree_signature(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        depth: int | None = None,
    ) -> str:
        return _stable_hash(self.subtree_key(match_level, depth=depth))
