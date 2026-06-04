from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm.auto import tqdm

from retrocast._version import __version__
from retrocast.adapters.paroutes import ConditionSlotParseStatistics, PaRoutesAdapter, analyze_condition_slots
from retrocast.chem import canonicalize_smiles, get_inchi_key
from retrocast.cli.progress import step_progress
from retrocast.curation.filtering import excise_reactions_from_route
from retrocast.curation.training.records import (
    AdaptationStatistics,
    AdaptedTrainingRoute,
    PreparedTrainingRoute,
    RawRouteSource,
    TrainingRouteAdaptation,
    TrainingRouteRecord,
    TrainingSetBuildConfig,
    TrainingSetBuildResult,
)
from retrocast.exceptions import AdapterError, ChemError, TrainingReleaseError
from retrocast.hashing import hash_file, hash_json
from retrocast.io import ContentType, create_manifest, load_json_artifact, save_jsonl_gz
from retrocast.io.cache import json_type_cache, local_cache
from retrocast.models.route import Route, RoutePath
from retrocast.models.task import Target
from retrocast.typing import InChIKeyStr, SmilesStr

TRAINING_RELEASE_ACTION = "scripts/paroutes/training-set-prep/create-training-release"
PAROUTES_TRAINING_ROUTE_ID_WIDTH = 6


def _phase(step, description: str):
    if step is None:
        return nullcontext()
    return step(description)


def stable_raw_route_hash(raw_route: dict[str, Any]) -> str:
    return hash_json(raw_route)


@local_cache(
    namespace="paroutes-adapted-routes",
    key=lambda source_path, *, dataset, **_: {
        "retrocast_version": __version__,
        "function": "adapt_training_routes",
        "source_sha256": hash_file(source_path),
        "dataset": dataset,
    },
    codec=json_type_cache(TrainingRouteAdaptation),
)
def adapt_training_routes(
    source_path: Path,
    *,
    dataset: str,
    show_progress: bool = True,
) -> TrainingRouteAdaptation:
    adapter = PaRoutesAdapter()
    entries = adapter.iter_raw_routes(load_json_artifact(source_path))
    parse_stats = ConditionSlotParseStatistics()
    adapted = []
    failures: dict[str, int] = defaultdict(int)
    reaction_hash_signature_pairs: set[tuple[str, str]] = set()
    if show_progress:
        entries = tqdm(entries, desc=f"adapt {dataset}", unit="route", leave=False)
    raw_count = 0
    for entry in entries:
        assert entry.source_order is not None
        raw_count += 1
        try:
            target_smiles = canonicalize_smiles(entry.payload["smiles"])
            target = Target(
                id=f"{dataset}-{entry.source_order:0{PAROUTES_TRAINING_ROUTE_ID_WIDTH}d}",
                smiles=SmilesStr(target_smiles),
                inchikey=InChIKeyStr(get_inchi_key(target_smiles)),
            )
        except ChemError as exc:
            failures[exc.code] += 1
            continue

        analyze_condition_slots(entry.payload, stats=parse_stats)
        try:
            route = adapter.cast(entry.payload, target=target)
        except (AdapterError, ChemError) as exc:
            failures[exc.code] += 1
            continue
        collect_reaction_hash_sanity(route, entry.payload, reaction_hash_signature_pairs)
        patent_id = route.annotations.get("patent_id")
        adapted.append(
            AdaptedTrainingRoute(
                route=route,
                source=RawRouteSource(
                    dataset=dataset,
                    raw_index=entry.source_order - 1,
                    raw_route_hash=stable_raw_route_hash(entry.payload),
                    patent_id=patent_id if isinstance(patent_id, str) else None,
                ),
            )
        )
    skipped = raw_count - len(adapted)
    assert_paroutes_reaction_hash_matches_retrocast_signature(reaction_hash_signature_pairs)
    skipped_without_error_code = skipped - sum(failures.values())
    return TrainingRouteAdaptation(
        routes=adapted,
        stats=AdaptationStatistics(
            raw_routes=raw_count,
            adapted_routes=len(adapted),
            skipped_routes=skipped,
            skipped_without_error_code=skipped_without_error_code,
            failures_by_code=dict(failures),
            non_fatal_condition_slot_parse=parse_stats,
        ),
    )


class TrainingRouteReleaseBuilder:
    def __init__(
        self,
        *,
        all_routes: Sequence[AdaptedTrainingRoute],
        all_adaptation: AdaptationStatistics,
        holdout_routes: Mapping[str, Sequence[AdaptedTrainingRoute]],
        holdout_adaptation: Mapping[str, AdaptationStatistics],
        config: TrainingSetBuildConfig,
    ) -> None:
        self.all_routes = all_routes
        self.all_adaptation = all_adaptation
        self.holdout_routes = holdout_routes
        self.holdout_adaptation = holdout_adaptation
        self.config = config
        self._started = False

    def build(self) -> TrainingSetBuildResult:
        if self._started:
            raise RuntimeError("TrainingRouteReleaseBuilder instances are single-use")
        self._started = True

        if not self.config.show_progress:
            return self._build(step=None)

        with step_progress(console=Console(), total=self._build_step_count(), transient=True) as step:
            return self._build(step=step)

    def _build_step_count(self) -> int:
        return 6 if self.config.holdout_mode == "reaction" else 5

    def _build(self, *, step) -> TrainingSetBuildResult:
        with _phase(step, f"{self.config.holdout_mode}: collect holdout routes"):
            holdout_routes = [route for routes in self.holdout_routes.values() for route in routes]
            holdout_route_signatures = {route.route.signature() for route in holdout_routes}

        holdout_reaction_signatures = set()
        if self.config.holdout_mode == "reaction":
            with _phase(step, "reaction: collect holdout reactions"):
                for route in holdout_routes:
                    holdout_reaction_signatures.update(route.route.reaction_signatures())

        with _phase(step, f"{self.config.holdout_mode}: prepare candidates"):
            candidates = []
            skipped_route_holdout = 0
            routes_with_overlapping_reactions = 0
            reaction_excision_fragments = 0
            routes_fully_removed_after_excision = 0
            for route in self.all_routes:
                structural_signature = route.route.signature()
                if structural_signature in holdout_route_signatures:
                    skipped_route_holdout += 1
                    continue
                if (
                    self.config.holdout_mode == "reaction"
                    and route.route.reaction_signatures() & holdout_reaction_signatures
                ):
                    routes_with_overlapping_reactions += 1
                    fragments = excise_reactions_from_route(route.route, holdout_reaction_signatures)
                    if not fragments:
                        routes_fully_removed_after_excision += 1
                        continue
                    reaction_excision_fragments += len(fragments)
                    for fragment in fragments:
                        candidates.append(
                            PreparedTrainingRoute(
                                route=fragment,
                                structural_signature=fragment.signature(),
                                sources=[route.source],
                            )
                        )
                    continue
                candidates.append(
                    PreparedTrainingRoute(
                        route=route.route,
                        structural_signature=structural_signature,
                        sources=[route.source],
                    )
                )

        with _phase(step, f"{self.config.holdout_mode}: merge exact duplicates"):
            exact_unique, exact_duplicates_removed = _merge_exact_route_duplicates(candidates)
        with _phase(step, f"{self.config.holdout_mode}: collapse mapped variants"):
            transform_unique, mapped_variants_collapsed = _merge_transform_route_variants(exact_unique)
        with _phase(step, f"{self.config.holdout_mode}: assign splits"):
            records = self._build_records(transform_unique)
        summary = {
            "input": {"all_routes": len(self.all_routes)},
            "adaptation": {
                "all_routes": self.all_adaptation.to_manifest_dict(),
                **{name: stats.to_manifest_dict() for name, stats in sorted(self.holdout_adaptation.items())},
            },
            "postprocessing": {
                "exact_route_matches_removed": skipped_route_holdout,
                "duplicate_routes_removed": exact_duplicates_removed + mapped_variants_collapsed,
                "chemical_duplicates_removed": exact_duplicates_removed,
                "mapped_smiles_variants_collapsed": mapped_variants_collapsed,
                "reaction_overlap": {
                    "unique_reference_reaction_signatures": len(holdout_reaction_signatures),
                    "routes_with_overlapping_reactions": routes_with_overlapping_reactions,
                    "fragments_kept_after_excision": reaction_excision_fragments,
                    "routes_fully_removed_after_excision": routes_fully_removed_after_excision,
                }
                if self.config.holdout_mode == "reaction"
                else None,
            },
            "output": summarize_records(records),
        }
        return TrainingSetBuildResult(release_name=self.config.release_name, records=records, summary=summary)

    def _build_records(
        self,
        routes: Sequence[PreparedTrainingRoute],
    ) -> list[TrainingRouteRecord]:
        validation_ids = _validation_indices(routes, val_fraction=self.config.val_fraction, seed=self.config.seed)
        width = max(6, len(str(len(routes))))
        records = []
        for index, route in enumerate(routes, start=1):
            split = "validation" if index - 1 in validation_ids else "training"
            records.append(
                TrainingRouteRecord(
                    id=f"{self.config.route_prefix}-{self.config.release_name}-{index:0{width}d}",
                    split=split,
                    route=route.route,
                    sources=route.sources,
                )
            )
        return records


def write_training_release(
    *,
    result: TrainingSetBuildResult,
    output_dir: Path,
    source_paths: list[Path],
    source_root: Path,
    config: TrainingSetBuildConfig,
) -> None:
    release_dir = output_dir / result.release_name
    all_path = release_dir / "all.jsonl.gz"
    training_path = release_dir / "training.jsonl.gz"
    validation_path = release_dir / "validation.jsonl.gz"
    manifest_path = release_dir / "manifest.json"
    training = [record for record in result.records if record.split == "training"]
    validation = [record for record in result.records if record.split == "validation"]
    save_jsonl_gz(result.records, all_path)
    save_jsonl_gz(training, training_path)
    save_jsonl_gz(validation, validation_path)
    manifest = create_manifest(
        action=TRAINING_RELEASE_ACTION,
        sources=source_paths,
        outputs=[
            ("all", all_path, result.records, ContentType.UNKNOWN, route_records_content_hash(result.records)),
            ("training", training_path, training, ContentType.UNKNOWN, route_records_content_hash(training)),
            ("validation", validation_path, validation, ContentType.UNKNOWN, route_records_content_hash(validation)),
        ],
        root_dir=source_root,
        parameters=config.to_manifest_dict(),
        statistics=result.summary.get("output", {}),
        summary=result.summary,
        release_name=result.release_name,
        keyed_output_files=True,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def summarize_records(records: Sequence[TrainingRouteRecord]) -> dict[str, Any]:
    training = sum(1 for record in records if record.split == "training")
    validation = sum(1 for record in records if record.split == "validation")
    depths = Counter(record.route.depth() for record in records)
    return {
        "all_records": {"total": len(records), "training": training, "validation": validation},
        "by_depth": {str(depth): depths[depth] for depth in sorted(depths)},
    }


def route_records_content_hash(records: Sequence[TrainingRouteRecord]) -> str:
    signatures = sorted(record.route.signature() for record in records)
    return hash_json(signatures)


def collect_reaction_hash_sanity(
    route: Route,
    raw_route: Mapping[str, Any],
    reaction_hash_signature_pairs: set[tuple[str, str]],
) -> None:
    reaction_hash_by_source_id = extract_paroutes_reaction_hash_by_source_id(raw_route)
    for reaction in route.iter_reactions():
        source_id = reaction.value.annotations.get("source_id")
        reaction_hash = reaction_hash_by_source_id.get(source_id) if isinstance(source_id, str) else None
        if reaction_hash is not None:
            reaction_hash_signature_pairs.add((reaction_hash, reaction.signature()))


def assert_paroutes_reaction_hash_matches_retrocast_signature(
    reaction_hash_signature_pairs: set[tuple[str, str]],
) -> None:
    n_pairs = len(reaction_hash_signature_pairs)
    n_hashes = len({reaction_hash for reaction_hash, _signature in reaction_hash_signature_pairs})
    n_signatures = len({signature for _reaction_hash, signature in reaction_hash_signature_pairs})
    if n_pairs != n_hashes or n_pairs != n_signatures:
        raise TrainingReleaseError(
            "PaRoutes reaction_hash is not equivalent to RetroCast reaction signatures: "
            f"{n_pairs} unique pairs, {n_hashes} unique hashes, {n_signatures} unique signatures",
            code="workflow.paroutes_reaction_hash_mismatch",
            context={"unique_pairs": n_pairs, "unique_hashes": n_hashes, "unique_signatures": n_signatures},
        )


def extract_paroutes_reaction_hash_by_source_id(raw_route: Mapping[str, Any]) -> dict[str, str]:
    reaction_hash_by_source_id = {}
    stack: list[Mapping[str, Any]] = [raw_route]
    while stack:
        node = stack.pop()
        metadata = node.get("metadata")
        if isinstance(metadata, Mapping):
            source_id = metadata.get("ID")
            reaction_hash = metadata.get("reaction_hash")
            if isinstance(source_id, str) and isinstance(reaction_hash, str) and reaction_hash:
                reaction_hash_by_source_id[source_id] = reaction_hash
        children = node.get("children")
        if isinstance(children, list):
            stack.extend(child for child in children if isinstance(child, Mapping))
    return reaction_hash_by_source_id


def _merge_exact_route_duplicates(
    routes: Sequence[PreparedTrainingRoute],
) -> tuple[list[PreparedTrainingRoute], int]:
    grouped: dict[tuple[Any, ...], PreparedTrainingRoute] = {}
    duplicates_removed = 0
    for route in routes:
        key = _route_exact_key(route.route)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = PreparedTrainingRoute(
                route=route.route, structural_signature=route.structural_signature, sources=list(route.sources)
            )
            continue
        existing.sources.extend(route.sources)
        duplicates_removed += 1
    return list(grouped.values()), duplicates_removed


def _merge_transform_route_variants(
    routes: Sequence[PreparedTrainingRoute],
) -> tuple[list[PreparedTrainingRoute], int]:
    groups: dict[tuple[Any, ...], list[PreparedTrainingRoute]] = defaultdict(list)
    for route in routes:
        groups[_route_transform_key(route.route)].append(route)

    output = []
    collapsed = 0
    for group in groups.values():
        collapsed += max(0, len(group) - 1)
        canonical = group[0] if len(group) == 1 else _choose_canonical_route_variant(group)
        output.append(
            PreparedTrainingRoute(
                route=_route_with_merged_source_annotations(canonical.route, group),
                structural_signature=canonical.route.signature(),
                sources=[source for route in group for source in route.sources],
            )
        )
    return output, collapsed


def _route_exact_key(route: Route) -> tuple[Any, ...]:
    return (route.signature(), tuple(sorted(_reaction_profiles(route))))


def _route_transform_key(route: Route) -> tuple[Any, ...]:
    return (
        route.signature(),
        tuple(sorted((signature, condition) for signature, _mapped, condition in _reaction_profiles(route))),
    )


def _reaction_profiles(route: Route) -> list[tuple[str, str | None, tuple[str, ...] | str | None]]:
    profiles = []
    for reaction in route.iter_reactions():
        profiles.append(
            (
                reaction.signature(),
                reaction.value.mapped_reaction_smiles,
                _condition_identity(reaction.value.annotations),
            )
        )
    return profiles


def _condition_identity(annotations: Mapping[str, Any]) -> tuple[str, ...] | str | None:
    condition_slot_smiles = annotations.get("condition_slot_smiles")
    if isinstance(condition_slot_smiles, Sequence) and not isinstance(condition_slot_smiles, (str, bytes)):
        values = tuple(str(value) for value in condition_slot_smiles if isinstance(value, str))
        if values:
            return tuple(sorted(values))
    condition_slot = annotations.get("condition_slot")
    return condition_slot if isinstance(condition_slot, str) and condition_slot else None


def _choose_canonical_route_variant(group: Sequence[PreparedTrainingRoute]) -> PreparedTrainingRoute:
    support_by_profile: dict[tuple[Any, ...], int] = defaultdict(int)
    for route in group:
        support_by_profile[tuple(sorted(_reaction_profiles(route.route)))] += len(route.sources)

    def key(route: PreparedTrainingRoute) -> tuple[int, tuple[Any, ...], str]:
        profile = tuple(sorted(_reaction_profiles(route.route)))
        first_hash = min((source.raw_route_hash for source in route.sources), default="")
        return (-support_by_profile[profile], profile, first_hash)

    return min(group, key=key)


def _route_with_merged_source_annotations(route: Route, group: Sequence[PreparedTrainingRoute]) -> Route:
    alternative_mapped_by_signature: dict[str, set[str]] = defaultdict(set)
    for variant in group:
        for signature, mapped_smiles, _condition in _reaction_profiles(variant.route):
            if mapped_smiles is not None:
                alternative_mapped_by_signature[signature].add(str(mapped_smiles))
    patents = sorted({source.patent_id for variant in group for source in variant.sources if source.patent_id})
    target = _annotate_route_molecule(route.target, RoutePath.target(), route, alternative_mapped_by_signature)
    annotations = {key: value for key, value in route.annotations.items() if key != "patent_id"}
    if patents:
        annotations["source_patent_ids"] = patents
    return Route(target=target, annotations=annotations)


def _annotate_route_molecule(
    molecule,
    path: RoutePath,
    route: Route,
    alternative_mapped_by_signature: Mapping[str, set[str]],
):
    reaction = molecule.product_of
    if reaction is None:
        return molecule.model_copy(deep=True)
    reaction_view = route.reaction_at(path.produced_by())
    reactants = [
        _annotate_route_molecule(reactant, path.produced_by().reactant(index), route, alternative_mapped_by_signature)
        for index, reactant in enumerate(reaction.reactants)
    ]
    annotations = reaction.annotations.copy()
    mapped_smiles = reaction.mapped_reaction_smiles
    alternatives = sorted(
        value
        for value in alternative_mapped_by_signature.get(reaction_view.signature(), set())
        if value != mapped_smiles
    )
    if alternatives:
        annotations["alternative_mapped_smiles"] = alternatives
    return molecule.model_copy(
        update={
            "product_of": reaction.model_copy(update={"reactants": reactants, "annotations": annotations}, deep=True)
        },
        deep=True,
    )


def _validation_indices(routes: Sequence[PreparedTrainingRoute], *, val_fraction: float, seed: int) -> set[int]:
    grouped: dict[tuple[int, bool], list[int]] = defaultdict(list)
    for index, route in enumerate(routes):
        grouped[(route.route.depth(), route.route.is_convergent())].append(index)

    rng = random.Random(seed)
    validation = set()
    for indices in grouped.values():
        count = int(round(len(indices) * val_fraction))
        if count > 0:
            validation.update(rng.sample(indices, count))
    return validation
