"""route embedding checks whether one route occurs inside another route.

the matcher treats routes as unordered rooted trees. exact subtree signatures
give a fast proof when roots already line up; recursive matching handles shifted
roots, leaf extension, and duplicate same-key reactant assignment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from retrocast.chem import InChIKeyLevel
from retrocast.exceptions import InvalidRouteEmbeddingQueryError
from retrocast.models.route import MoleculeView, Route, RoutePath


@dataclass(frozen=True, slots=True)
class LeafExtension:
    """evidence that the container continues below a matched query leaf.

    ``query_leaf_path`` names the query boundary that was exceeded.
    ``container_path`` names the corresponding molecule in the container route;
    that molecule has a producing reaction, but the query stops there.
    """

    query_leaf_path: RoutePath
    container_path: RoutePath


@dataclass(frozen=True, slots=True)
class EmbeddingMatch:
    """trace for one place where a query route embeds in a container route.

    ``query_path`` and ``container_path`` are the two roots that matched.
    ``matched_reactions`` counts only reactions present in the query-side
    pattern, not reactions below extended container leaves. ``leaf_extensions``
    records those extended query leaves for audit ledgers and explanations.
    """

    query_path: RoutePath
    container_path: RoutePath
    matched_reactions: int
    leaf_extensions: tuple[LeafExtension, ...] = ()

    @property
    def leaf_extended(self) -> bool:
        """return whether the container continues below any matched query leaf."""
        return bool(self.leaf_extensions)

    @property
    def root_shifted(self) -> bool:
        """return whether the query root matched an internal container molecule."""
        return self.container_path != RoutePath.target()


@dataclass(frozen=True, slots=True)
class _Trace:
    """internal match trace for the recursive matcher.

    ``EmbeddingMatch`` stores the selected roots. ``_Trace`` does not; it only
    carries the facts accumulated below the current recursive pair. parent calls
    add their own matched reaction and merge traces from sibling branches.
    """

    matched_reactions: int
    leaf_extensions: tuple[LeafExtension, ...] = ()


def find_route_embeddings(
    query: Route,
    container: Route,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    *,
    allow_leaf_extension: bool = True,
) -> tuple[EmbeddingMatch, ...]:
    """scan the container for every place where the query target embeds.

    the query is always rooted at its target. the container root is allowed to
    shift, so this finds both exact/full-route matches and internal matches like
    ``b <- c`` inside ``a <- b <- c``.
    """
    _validate_embedding_query(query)
    query_root = query.molecule_at(RoutePath.target())
    matches = []
    for container_root in container.iter_molecules():
        match = route_embeds_at(query_root, container_root, match_level, allow_leaf_extension=allow_leaf_extension)
        if match is not None:
            matches.append(match)
    return tuple(matches)


def route_embeds_at(
    query: MoleculeView,
    container: MoleculeView,
    match_level: InChIKeyLevel = InChIKeyLevel.FULL,
    *,
    allow_leaf_extension: bool = True,
) -> EmbeddingMatch | None:
    """check one selected query root against one selected container root.

    exact subtree signatures are accepted as a fast proof. all other candidates
    are checked by the recursive matcher, which is authoritative for leaf
    extension and duplicate same-key reactant assignment.
    """
    if not _can_match_root(query, container, match_level, allow_leaf_extension=allow_leaf_extension):
        return None

    if query.subtree_signature(match_level) == container.subtree_signature(match_level):
        return EmbeddingMatch(
            query_path=query.path,
            container_path=container.path,
            matched_reactions=subtree_reaction_count(query),
        )

    memo: dict[tuple[RoutePath, RoutePath], _Trace | None] = {}
    trace = _match_molecule(query, container, match_level, allow_leaf_extension=allow_leaf_extension, memo=memo)
    if trace is None:
        return None
    return EmbeddingMatch(
        query_path=query.path,
        container_path=container.path,
        matched_reactions=trace.matched_reactions,
        leaf_extensions=trace.leaf_extensions,
    )


def subtree_reaction_count(molecule: MoleculeView) -> int:
    """count reactions in the rooted subtree below ``molecule``."""
    return sum(1 for candidate in molecule.iter_molecules() if candidate.produced_by() is not None)


def _validate_embedding_query(query: Route) -> None:
    if query.target.product_of is not None:
        return
    raise InvalidRouteEmbeddingQueryError(
        "route embedding query must contain at least one reaction; use Route.contains_molecule() "
        "for molecule membership checks",
        context={
            "query_target_smiles": str(query.target.smiles),
            "query_target_inchikey": str(query.target.inchikey),
            "query_reactions": 0,
            "suggested_operation": "Route.contains_molecule",
        },
    )


def _can_match_root(
    query: MoleculeView,
    container: MoleculeView,
    match_level: InChIKeyLevel,
    *,
    allow_leaf_extension: bool,
) -> bool:
    """return whether a root pair can possibly embed.

    this is a necessary condition, not proof. molecule keys must match. non-leaf
    query roots also require the same local reaction signature. query leaves may
    match non-leaf container roots only when leaf extension is enabled.
    """
    if query.key(match_level) != container.key(match_level):
        return False

    query_reaction = query.produced_by()
    container_reaction = container.produced_by()
    if query_reaction is None:
        return allow_leaf_extension or container_reaction is None
    if container_reaction is None:
        return False
    return query_reaction.signature(match_level) == container_reaction.signature(match_level)


def _match_molecule(
    query: MoleculeView,
    container: MoleculeView,
    match_level: InChIKeyLevel,
    *,
    allow_leaf_extension: bool,
    memo: dict[tuple[RoutePath, RoutePath], _Trace | None],
) -> _Trace | None:
    """recursively match two molecule roots after root-level checks pass.

    leaves are the only asymmetric case: a query leaf can match a container
    non-leaf when leaf extension is enabled. otherwise both sides must expose
    the same producing reaction, and their reactants are matched recursively as
    an unordered multiset.
    """
    key = (query.path, container.path)
    if key in memo:
        return memo[key]

    if query.key(match_level) != container.key(match_level):
        memo[key] = None
        return None

    query_reaction = query.produced_by()
    container_reaction = container.produced_by()
    if query_reaction is None:
        if container_reaction is None:
            trace = _Trace(matched_reactions=0)
        elif allow_leaf_extension:
            trace = _Trace(
                matched_reactions=0,
                leaf_extensions=(LeafExtension(query_leaf_path=query.path, container_path=container.path),),
            )
        else:
            trace = None
        memo[key] = trace
        return trace

    if container_reaction is None or query_reaction.signature(match_level) != container_reaction.signature(match_level):
        memo[key] = None
        return None

    child_trace = _match_reactants(
        query_reaction.reactants(),
        container_reaction.reactants(),
        match_level,
        allow_leaf_extension=allow_leaf_extension,
        memo=memo,
    )
    trace = (
        None
        if child_trace is None
        else _Trace(
            matched_reactions=child_trace.matched_reactions + 1,
            leaf_extensions=child_trace.leaf_extensions,
        )
    )
    memo[key] = trace
    return trace


def _match_reactants(
    query_reactants: Sequence[MoleculeView],
    container_reactants: Sequence[MoleculeView],
    match_level: InChIKeyLevel,
    *,
    allow_leaf_extension: bool,
    memo: dict[tuple[RoutePath, RoutePath], _Trace | None],
) -> _Trace | None:
    """match two reaction reactant lists as unordered duplicate-aware multisets.

    reactants are first grouped by molecule key. group keys and group sizes must
    match, which preserves duplicate multiplicity. same-key groups are then
    matched recursively because same molecule identity does not guarantee same
    downstream subtree.
    """
    if len(query_reactants) != len(container_reactants):
        return None

    query_groups = _group_reactants_by_molecule_key(query_reactants, match_level)
    container_groups = _group_reactants_by_molecule_key(container_reactants, match_level)
    if query_groups.keys() != container_groups.keys():
        return None
    if any(len(query_groups[molecule_key]) != len(container_groups[molecule_key]) for molecule_key in query_groups):
        return None

    trace = _Trace(matched_reactions=0)
    for molecule_key in sorted(query_groups):
        group_trace = _match_same_key_reactants(
            query_groups[molecule_key],
            container_groups[molecule_key],
            match_level,
            allow_leaf_extension=allow_leaf_extension,
            memo=memo,
        )
        if group_trace is None:
            return None
        trace = _merge_traces(trace, group_trace)
    return trace


def _group_reactants_by_molecule_key(
    molecules: Sequence[MoleculeView],
    match_level: InChIKeyLevel,
) -> dict[str, list[MoleculeView]]:
    """group sibling reactants so matching is order-invariant but duplicate-aware."""
    grouped: dict[str, list[MoleculeView]] = defaultdict(list)
    for molecule in molecules:
        grouped[molecule.key(match_level)].append(molecule)
    return grouped


def _match_same_key_reactants(
    query_reactants: Sequence[MoleculeView],
    container_reactants: Sequence[MoleculeView],
    match_level: InChIKeyLevel,
    *,
    allow_leaf_extension: bool,
    memo: dict[tuple[RoutePath, RoutePath], _Trace | None],
) -> _Trace | None:
    """find a recursive assignment inside one same-key sibling group.

    same-key siblings cannot be paired by position because reactant order is not
    semantic. this backtracks over container assignments and keeps the valid
    assignment with the fewest leaf extensions, using path ids only to break ties.
    """
    used = [False] * len(container_reactants)

    def assign(query_index: int) -> _Trace | None:
        if query_index == len(query_reactants):
            return _Trace(matched_reactions=0)

        best: _Trace | None = None
        query = query_reactants[query_index]
        for container_index, container in enumerate(container_reactants):
            if used[container_index]:
                continue
            trace = _match_molecule(
                query,
                container,
                match_level,
                allow_leaf_extension=allow_leaf_extension,
                memo=memo,
            )
            if trace is None:
                continue

            used[container_index] = True
            rest = assign(query_index + 1)
            used[container_index] = False
            if rest is not None:
                candidate = _merge_traces(trace, rest)
                # prefer fewer leaf extensions; path ids only make ties deterministic.
                candidate_rank = (
                    len(candidate.leaf_extensions),
                    tuple(extension.query_leaf_path.id() for extension in candidate.leaf_extensions),
                    tuple(extension.container_path.id() for extension in candidate.leaf_extensions),
                )
                if best is None:
                    best = candidate
                    continue

                best_rank = (
                    len(best.leaf_extensions),
                    tuple(extension.query_leaf_path.id() for extension in best.leaf_extensions),
                    tuple(extension.container_path.id() for extension in best.leaf_extensions),
                )
                if candidate_rank < best_rank:
                    best = candidate
        return best

    return assign(0)


def _merge_traces(left: _Trace, right: _Trace) -> _Trace:
    """combine matched-reaction counts and leaf-extension evidence from two branches."""
    return _Trace(
        matched_reactions=left.matched_reactions + right.matched_reactions,
        leaf_extensions=left.leaf_extensions + right.leaf_extensions,
    )
