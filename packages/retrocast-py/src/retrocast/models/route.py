from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, Literal, cast

from pydantic import AfterValidator, BaseModel, Field, model_validator

from retrocast.chem import InChIKeyLevel, reduce_inchikey
from retrocast.hashing import hash_json
from retrocast.typing import InChIKeyStr, ReactionSmilesStr, SmilesStr

# section: public route types

RouteNodeKind = Literal["m", "r"]
ReactionContentField = Literal["mapped_reaction_smiles", "template", "reagents", "solvents"]
REACTION_CONTENT_FIELDS: tuple[ReactionContentField, ...] = (
    "mapped_reaction_smiles",
    "template",
    "reagents",
    "solvents",
)


# section: shared validation and hashing


def _stable_hash(value: Any) -> str:
    return hash_json(value)


# section: route paths


@dataclass(frozen=True, slots=True)
class RoutePath:
    kind: RouteNodeKind
    indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _parse_route_path(self.id())

    @classmethod
    def parse(cls, value: str) -> RoutePath:
        kind, indices = _parse_route_path(value)
        return cls(kind=kind, indices=tuple(indices))

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
        return RoutePath.parse(_transform_route_path(self.id(), "produced_by"))

    def product(self) -> RoutePath:
        return RoutePath.parse(_transform_route_path(self.id(), "product"))

    def reactant(self, index: int) -> RoutePath:
        return RoutePath.parse(_transform_route_path(self.id(), "reactant", index=index))


def _parse_route_path(value: str) -> tuple[RouteNodeKind, list[int]]:
    from retrocast import _native

    kind, indices = _native.route_path_parse(value)
    return cast(RouteNodeKind, kind), indices


def _transform_route_path(value: str, operation: str, *, index: int | None = None) -> str:
    from retrocast import _native

    return _native.route_path_transform(value, operation, index=index)


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


# section: core schema


class Reaction(BaseModel):
    reactants: list[Molecule]
    mapped_reaction_smiles: ReactionSmilesStr | None = None
    template: str | None = None
    reagents: list[SmilesStr] | None = None
    solvents: list[SmilesStr] | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_reactant_order(self) -> Reaction:
        self.reactants = _normalize_reactants(self.reactants)
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


# section: structural and content identity


def _normalize_reactants(reactants: list[Molecule]) -> list[Molecule]:
    from retrocast import _native

    payload = json.dumps([_molecule_wire(reactant) for reactant in reactants], separators=(",", ":"))
    order: list[int] = json.loads(_native.reactant_order_json(payload))
    return [reactants[index] for index in order]


def _molecule_subtree_key(
    molecule: Molecule,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    *,
    depth: int | None = None,
) -> tuple[Any, ...]:
    return _native_identity(
        _route_wire(molecule),
        "molecule",
        match_level,
        path="rc:m:/",
        depth=depth,
    )


def _reaction_key(
    product: Molecule,
    reaction: Reaction,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
) -> tuple[str, str, tuple[str, ...]]:
    return _native_identity(
        _route_wire(product, reaction=reaction),
        "reaction",
        match_level,
        path="rc:r:/",
    )


def _normalize_content_fields(fields: Iterable[ReactionContentField]) -> tuple[ReactionContentField, ...]:
    # str is iterable, but a field name should be treated as one requested field.
    requested = {fields} if isinstance(fields, str) else set(fields)
    unknown = sorted(requested - set(REACTION_CONTENT_FIELDS))
    if unknown:
        raise ValueError(f"unknown reaction content fields: {', '.join(unknown)}")
    return tuple(field for field in REACTION_CONTENT_FIELDS if field in requested)


def _reaction_content_key(
    product: Molecule,
    reaction: Reaction,
    match_level: InChIKeyLevel,
    *,
    fields: tuple[ReactionContentField, ...],
) -> tuple[Any, ...]:
    return _native_identity(
        _route_wire(product, reaction=reaction),
        "reaction",
        match_level,
        path="rc:r:/",
        fields=fields,
    )


def _molecule_content_subtree_key(
    molecule: Molecule,
    match_level: InChIKeyLevel,
    *,
    fields: tuple[ReactionContentField, ...],
    depth: int | None = None,
) -> tuple[Any, ...]:
    return _native_identity(
        _route_wire(molecule),
        "molecule",
        match_level,
        path="rc:m:/",
        depth=depth,
        fields=fields,
    )


def _native_identity(
    route: dict[str, Any],
    node_kind: str,
    match_level: InChIKeyLevel,
    *,
    path: str | None = None,
    depth: int | None = None,
    fields: Iterable[ReactionContentField] = (),
) -> tuple[Any, ...]:
    from retrocast import _native

    result = json.loads(
        _native.route_identity_json(
            json.dumps(route, separators=(",", ":")),
            node_kind,
            level=match_level.value,
            path=path,
            depth=depth,
            fields_json=json.dumps(list(fields), separators=(",", ":")),
        )
    )
    return _tuples(result["key"])


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def _route_wire(product: Molecule, *, reaction: Reaction | None = None) -> dict[str, Any]:
    target = _molecule_wire(product)
    if reaction is not None:
        target["product_of"] = _reaction_wire(reaction)
    return {"target": target, "schema_version": "2"}


def _molecule_wire(molecule: Molecule) -> dict[str, Any]:
    return {
        "smiles": str(molecule.smiles),
        "inchikey": str(molecule.inchikey),
        "product_of": _reaction_wire(molecule.product_of) if molecule.product_of is not None else None,
    }


def _reaction_wire(reaction: Reaction) -> dict[str, Any]:
    return {
        "reactants": [_molecule_wire(reactant) for reactant in reaction.reactants],
        "mapped_reaction_smiles": reaction.mapped_reaction_smiles,
        "template": reaction.template,
        "reagents": reaction.reagents,
        "solvents": reaction.solvents,
    }


def _native_structure(route: Route, *, molecule_path: str | None = None) -> dict[str, Any]:
    from retrocast import _native

    return json.loads(
        _native.route_structure_json(
            json.dumps(_route_wire(route.target), separators=(",", ":")),
            molecule_path=molecule_path,
        )
    )


# section: route model


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
        for path in _native_structure(self)["leaf_paths"]:
            yield self.molecule_at(path)

    def iter_molecules(self) -> Iterator[MoleculeView]:
        for path in _native_structure(self)["molecule_paths"]:
            yield self.molecule_at(path)

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
        for path in _native_structure(self)["reaction_paths"]:
            yield self.reaction_at(path)

    def reaction_signatures(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> set[str]:
        return {reaction.signature(match_level) for reaction in self.iter_reactions()}

    def content_key(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        fields: Iterable[ReactionContentField],
        depth: int | None = None,
    ) -> tuple[Any, ...]:
        content_fields = _normalize_content_fields(fields)
        return self.molecule_at(RoutePath.target()).content_subtree_key(match_level, fields=content_fields, depth=depth)

    def content_signature(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        fields: Iterable[ReactionContentField],
        depth: int | None = None,
    ) -> str:
        return _stable_hash(self.content_key(match_level, fields=fields, depth=depth))

    def reaction_content_signatures(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        fields: Iterable[ReactionContentField],
    ) -> set[str]:
        content_fields = _normalize_content_fields(fields)
        return {reaction.content_signature(match_level, fields=content_fields) for reaction in self.iter_reactions()}

    def depth(self) -> int:
        return int(_native_structure(self)["depth"])

    def is_convergent(self) -> bool:
        return bool(_native_structure(self)["is_convergent"])


# section: route-bound views


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

    def content_key(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        fields: Iterable[ReactionContentField],
    ) -> tuple[Any, ...]:
        content_fields = _normalize_content_fields(fields)
        return _reaction_content_key(self.product().value, self.value, match_level, fields=content_fields)

    def content_signature(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        fields: Iterable[ReactionContentField],
    ) -> str:
        return _stable_hash(self.content_key(match_level, fields=fields))


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
        for path in _native_structure(self.route, molecule_path=self.path.id())["leaf_paths"]:
            yield self.route.molecule_at(path)

    def iter_molecules(self) -> Iterator[MoleculeView]:
        for path in _native_structure(self.route, molecule_path=self.path.id())["molecule_paths"]:
            yield self.route.molecule_at(path)

    def depth(self) -> int:
        return int(_native_structure(self.route, molecule_path=self.path.id())["depth"])

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

    def content_subtree_key(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        fields: Iterable[ReactionContentField],
        depth: int | None = None,
    ) -> tuple[Any, ...]:
        content_fields = _normalize_content_fields(fields)
        return _molecule_content_subtree_key(self.value, match_level, fields=content_fields, depth=depth)

    def content_subtree_signature(
        self,
        match_level: InChIKeyLevel = InChIKeyLevel.FULL,
        *,
        fields: Iterable[ReactionContentField],
        depth: int | None = None,
    ) -> str:
        return _stable_hash(self.content_subtree_key(match_level, fields=fields, depth=depth))
