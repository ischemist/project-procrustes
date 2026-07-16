"""Python-facing route embedding results backed by the Rust engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from retrocast.chem import InChIKeyLevel
from retrocast.exceptions import InvalidRouteEmbeddingQueryError
from retrocast.models.route import MoleculeView, Route, RoutePath


@dataclass(frozen=True, slots=True)
class LeafExtension:
    """Evidence that the container continues below a matched query leaf."""

    query_leaf_path: RoutePath
    container_path: RoutePath


@dataclass(frozen=True, slots=True)
class EmbeddingMatch:
    """Trace for one place where a query route embeds in a container route."""

    query_path: RoutePath
    container_path: RoutePath
    matched_reactions: int
    leaf_extensions: tuple[LeafExtension, ...] = ()

    @property
    def leaf_extended(self) -> bool:
        return bool(self.leaf_extensions)

    @property
    def root_shifted(self) -> bool:
        return self.container_path != RoutePath.target()


def find_route_embeddings(
    query: Route,
    container: Route,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    *,
    allow_leaf_extension: bool = True,
) -> tuple[EmbeddingMatch, ...]:
    """Return every Rust-computed embedding of ``query`` in ``container``."""
    from retrocast import native

    try:
        matches = native.find_route_embeddings(
            query,
            container,
            match_level=match_level,
            allow_leaf_extension=allow_leaf_extension,
        )
    except RuntimeError as error:
        if query.target.product_of is not None:
            raise
        raise InvalidRouteEmbeddingQueryError(
            "route embedding query must contain at least one reaction; use Route.contains_molecule() "
            "for molecule membership checks",
            context={
                "query_target_smiles": str(query.target.smiles),
                "query_target_inchikey": str(query.target.inchikey),
                "query_reactions": 0,
                "suggested_operation": "Route.contains_molecule",
            },
        ) from error
    return tuple(_embedding_from_json(match) for match in matches)


def route_embeds_at(
    query: MoleculeView,
    container: MoleculeView,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    *,
    allow_leaf_extension: bool = True,
) -> EmbeddingMatch | None:
    """Return the Rust-computed match for two selected molecule roots."""
    from retrocast import native

    matched = native.route_embeds_at(
        query,
        container,
        match_level=match_level,
        allow_leaf_extension=allow_leaf_extension,
    )
    return None if matched is None else _embedding_from_json(matched)


def subtree_reaction_count(molecule: MoleculeView) -> int:
    """Count reactions in the rooted subtree below ``molecule`` in Rust."""
    from retrocast import native

    return native.subtree_reaction_count(molecule)


def _embedding_from_json(value: dict[str, object]) -> EmbeddingMatch:
    raw_extensions = value.get("leaf_extensions", [])
    assert isinstance(raw_extensions, list)
    extensions = []
    for raw_extension in raw_extensions:
        assert isinstance(raw_extension, dict)
        extension = cast("dict[str, object]", raw_extension)
        extensions.append(
            LeafExtension(
                query_leaf_path=RoutePath.parse(str(extension["query_leaf_path"])),
                container_path=RoutePath.parse(str(extension["container_path"])),
            )
        )
    return EmbeddingMatch(
        query_path=RoutePath.parse(str(value["query_path"])),
        container_path=RoutePath.parse(str(value["container_path"])),
        matched_reactions=int(str(value["matched_reactions"])),
        leaf_extensions=tuple(extensions),
    )
