# backlog for library redesign.

retrocast should have a clean separation between:

1. reading whatever raw artifact a user has
2. adapting route-like payloads into canonical `Route` objects
3. projecting those canonical routes into benchmark or training workflows

the core promise of the library should be literal:

> give retrocast whatever route-ish artifact you have, and retrocast will give you canonical `Route` objects.

benchmark alignment is a separate workflow.
training-set preparation is a separate workflow.
neither should define the core adapter boundary.

### the core concepts are explicit

the library should keep the noun set intentionally small.

`raw artifact`

- the untrusted thing produced by a planner, model, service, notebook, dataset dump, or script.
- it may be json, jsonl, gzipped, keyed, unkeyed, per-target, multi-target, ranked, unranked, or messy.

`RawRouteEntry`

- a workflow-level unit yielded by artifact readers.
- it wraps one route-like raw payload plus source context.
- it is not the chemistry boundary.
- it is not required in every public api.
- it exists so corpus traversal, source ids, row indices, and target hints do not get smeared into `Route`.

`Route`

- the canonical chemistry representation.
- this is the core output of route adaptation.
- it should be reusable for prediction evaluation, training-set prep, reference-route curation, and ad hoc analysis.

`route corpus`

- an iterable or `jsonl.gz` artifact of canonical `Route` rows.
- it is benchmark-agnostic.
- it is training-agnostic.
- row order is meaningful.

`benchmark collection`

- the workflow that derives benchmark-keyed predictions from a route corpus.
- it owns matching policy, ambiguity handling, unmatched-route policy, deduplication, and final per-target ordering.

`training release projection`

- the workflow that derives training route and reaction records from canonical routes plus source information.
- it owns split logic, holdout policy, duplicate collapse, and release-specific provenance.

once these nouns exist, the architecture stops smearing unrelated jobs together.

### `Route` is chemistry-only

the `Route` model should be deliberately boring.

it represents chemistry and route-local annotations only:

- route topology
- molecules
- reaction metadata
- route metadata that is intrinsic to the route itself
- retrocast provenance fields that truly belong on the route object

it should not carry collection-specific ordering fields anymore.

specifically:

- ordering is not fake chemistry metadata.
- benchmark target ids do not belong on `Route`.
- provider ordering does not belong on `Route`.
- collection policy does not belong on `Route`.

this makes `Route` reusable in contexts where ranking is unknown, irrelevant, or multiple orderings coexist.

ordering no longer lives on `Route`.
the canonical ordering signal is list order in route corpora and benchmark-collected outputs, while scoring derives explicit ranks at evaluation time.

### raw artifact reading is separate from chemistry adaptation

traditional planners nudged the original design toward target-keyed blobs because they are often called once per target.
that was a useful implementation convenience, but it is not the right universal boundary for a general library.

the redesign separates two jobs:

1. unpacking a raw artifact into route-like units
2. turning one route-like unit into one canonical `Route`

the first job belongs to readers and workflow helpers.
the second job belongs to adapters.

the reader-facing shape looks roughly like this:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RawRouteEntry:
    payload: Any
    source_key: str | None = None
    source_row_index: int | None = None
    source_record_id: str | None = None
    expected_target_id: str | None = None
    expected_target_smiles: str | None = None
```

important ownership rules:

- readers know how to walk keyed and unkeyed corpora.
- readers know how to split one large raw container into individual route-like units.
- adapters do not need to understand corpus traversal policy.
- optional target hints can exist at the workflow layer without becoming part of `Route`.

### adapters are route-first

the core adapter boundary should be route-first, not benchmark-first.

the primary contract looks roughly like this:

```python
class RouteAdapter(Protocol):
    def cast(
        self,
        raw_route: Any,
        *,
        ignore_stereo: bool = False,
    ) -> Route: ...
```

hard rules for this contract:

- the adapter validates raw route payloads at its own boundary.
- one route-like raw payload maps to one `Route`.
- if one provider artifact contains many routes, that container is split before `cast()` is called.
- the adapter does not require a benchmark object in order to do useful work.
- the adapter may use workflow-supplied target hints for optional validation, but that validation is not the core chemistry contract.
- the adapter returns canonical chemistry, not benchmark grouping.

that keeps the word "adapter" honest.

### route corpora are the canonical intermediate shape

the redesign treats streaming as a system-level requirement, not a nice-to-have.

- adaptation is iterator-first.
- collection is iterator-first.
- training release prep can also be iterator-first.
- in-memory dictionaries are still useful, but they are secondary convenience shapes.
- a user should be able to process a large multi-target corpus without first materializing a giant benchmark-keyed map.

the ideal dataflow is mechanical and inspectable:

`raw artifact -> iterator[RawRouteEntry] -> adapter.cast(entry.payload) -> iterator[Route]`

every higher-level workflow should sit on top of that spine.

### the artifact formats match the workflow

the library should have one canonical intermediate artifact and one derived benchmark artifact.

`route-corpus.jsonl.gz`

- one json row per canonical `Route`
- streamable
- resumable
- easy to inspect
- easy to filter or repartition
- appropriate for multi-target raw corpora
- preserves encounter order by row order

`routes.json.gz`

- benchmark-keyed `dict[target_id, list[Route]]`
- still useful for scoring and legacy analysis flows
- list order is the benchmark ranking contract

this split is important.

the route corpus artifact is for normalization.
the benchmark-keyed artifact is for scoring and benchmark-level reporting.

they are related, but they are not the same object wearing two hats.

if a workflow needs richer source provenance than plain `Route` rows carry, that provenance should live in:

- the in-memory `RawRouteEntry` stream during the run
- manifests
- workflow-specific sidecars
- workflow-specific output models

not in the core `Route` schema.

### benchmark collection is a separate library feature

retrocast should expose a dedicated collector layer instead of hiding collection logic inside `ingest`.

the core collector contract looks roughly like this:

```python
def collect_benchmark_predictions(
    routes: Iterable[Route],
    benchmark: BenchmarkSet,
    *,
    match_by: tuple[str, ...] = ("canonical_smiles",),
    on_unmatched: Literal["skip", "error", "report"] = "report",
    on_ambiguous: Literal["error", "skip", "report"] = "error",
    deduplicate: bool = True,
) -> CollectedBenchmarkPredictions: ...
```

this function owns the benchmark-specific policy surface:

- matching by canonical smiles
- optional matching by workflow-supplied target hints when a higher-level wrapper has them
- ambiguity handling when multiple benchmark targets share the same smiles
- unmatched route accounting
- deduplication within each target bucket
- final ordering of route lists for scoring

the collector does not parse raw provider data.
the collector does not normalize chemistry.
the collector does not define the adapter contract.

the core collector should be able to operate on plain `Route` streams.
higher-level wrappers like `ingest` may use `RawRouteEntry` hints to enrich matching when explicit target ids are available.

### training-set preparation is another projection, not a special adapter mode

the training pipeline already proves an important point:
not every adapted route is a prediction.

training-set prep should therefore reuse the same adaptation spine:

`raw artifact -> RawRouteEntry -> Route`

and then apply training-specific logic afterward:

- holdout selection
- reaction excision
- route deduplication
- split assignment
- release provenance

training-specific output records should remain training-specific.
they should not force prediction terminology onto the core adaptation layer.

the existing training models are a good hint here:

- `RawRouteSource`
- `AdaptedTrainingRoute`
- `TrainingRouteRecord`
- `TrainingReactionRecord`

the long-term goal is not to make those disappear.
the goal is to make them sit on top of shared adaptation and raw-entry infrastructure instead of reinventing it locally.

### the cli matches the mental model

the command line should mirror the architecture instead of smuggling multiple jobs into one verb.
it should also lead with the best default path rather than the most abstract one.

`retrocast ingest`

- the flagship command for most users
- input: raw route-ish artifact plus benchmark
- output: benchmark-keyed `routes.json.gz`
- runs `adapt` and `collect` internally
- preserves the one-command happy path for users who do not care about the intermediate corpus
- emits summaries that show both adaptation and collection outcomes

example:

```bash
uv run retrocast ingest \
  --adapter ursa-llm \
  --input completions.jsonl.gz \
  --benchmark data/retrocast/1-benchmarks/definitions/example.json.gz \
  --output-dir data/retrocast/3-processed/example/ursa-llm
```

`retrocast adapt`

- input: raw route-ish artifact
- output: `route-corpus.jsonl.gz`
- no benchmark required
- errors talk about raw records, row indices, source ids, and adapter contracts
- power-user primitive for users who want canonical routes directly

example:

```bash
uv run retrocast adapt \
  --adapter ursa-llm \
  --input completions.jsonl.gz \
  --output route-corpus.jsonl.gz
```

`retrocast collect`

- input: route corpus
- output: benchmark-keyed `routes.json.gz`
- benchmark required
- errors talk about matching, ambiguity, missing targets, and benchmark alignment
- power-user primitive for users who want explicit benchmark collection as a separate step

example:

```bash
uv run retrocast collect \
  --input route-corpus.jsonl.gz \
  --benchmark data/retrocast/1-benchmarks/definitions/example.json.gz \
  --output routes.json.gz
```

the one-command path stays beautiful.
the lower-level path stays explicit.

### the dx is aggressively explicit

the library should feel fast, sharp, and unsurprising.

that means:

- readers advertise the raw shapes and file formats they support.
- adapters advertise the route payload shapes they support.
- the cli tells the user exactly what object is being produced at each stage.
- errors use stable machine-readable codes and human-readable context.
- manifests record which stage produced which artifact.
- summaries are deterministic and compact.
- ambiguity is never resolved by hidden guesses.
- if collection falls back from explicit target hints to smiles matching, it says so in the summary.

the output from a successful run should be the kind of thing a user can trust at a glance:

- entries read
- route-like payloads extracted
- routes adapted
- entries dropped by schema failure
- entries dropped by chemistry failure
- matched by explicit target hint
- matched by canonical smiles
- unmatched routes
- ambiguous routes
- routes deduplicated
- final routes saved

the output from a failed run should be equally crisp:

- which stage failed
- which reader failed
- which adapter failed
- which raw artifact failed
- which row or source record failed
- whether the failure was adaptation or collection
- whether the run is retry-safe

### ordering semantics are finally sane

retrocast should recognize that there are multiple kinds of ordering.

`encounter order`

- always meaningful
- owned by the raw-entry stream and preserved by `jsonl.gz` row order
- this is the default ordering for generic route corpora

`provider order`

- optional
- may be available from the raw artifact
- belongs in workflow-local source context, manifests, or reader helpers when needed
- does not belong on `Route`

`benchmark order`

- owned by collection
- represented by the order of `list[Route]` inside the benchmark-keyed output
- derived after matching, deduplication, and filtering

this prevents the old category error where one `rank` field tried to mean several different things at once.

### matching behavior is a product feature, not a side effect

benchmark collection should expose policy intentionally.

the defaults should be conservative:

- match by canonical smiles by default
- use explicit target-id hints only when a wrapper workflow has them
- treat many-to-one smiles matches as ambiguous by default
- report unmatched routes instead of silently disappearing them

advanced users can override those policies deliberately, but the override should be visible in code and in manifests.

the collector should produce a structured report that can be saved or logged:

- matched by explicit target hint
- matched by smiles
- ambiguous by smiles
- unmatched routes
- duplicate routes dropped

### full migration phases

the redesign should not require a flag day.
it should ship in phases that each leave the repo in a working state.

#### phase 0: freeze the nouns and boundaries

- adopt the vocabulary in this document.
- stop using `PredictionRecord` as the target architecture noun.
- treat any exploratory record-wrapper implementations as migration scaffolding, not the final design.

#### phase 1: separate corpus reading from route adaptation

- introduce shared raw-artifact reader helpers that yield `RawRouteEntry`.
- make keyed-vs-unkeyed traversal explicit and testable.
- move per-corpus unpacking logic out of benchmark-centric ingest paths.
- keep existing user-facing workflows working while the new reader seam lands.

#### phase 2: make the core adapter contract route-first

- introduce the route-first adapter boundary: one raw route payload in, one `Route` out.
- keep optional validation helpers for expected target smiles or ids outside the core chemistry return type.
- add compatibility wrappers for current target-bound adapters while they migrate.
- update adapter tests so they validate route transformation separately from corpus traversal.

#### phase 3: introduce the canonical route-corpus artifact

- make `retrocast adapt` read a raw artifact, iterate `RawRouteEntry`, adapt each payload into `Route`, and write `route-corpus.jsonl.gz`.
- make row order the canonical generic ordering.
- emit adaptation stats, manifests, and failure accounting at this stage.
- do not require a benchmark for this path.

#### phase 4: introduce explicit benchmark collection

- implement `retrocast collect` over route corpora.
- support smiles-based collection in the core collector.
- allow higher-level wrappers to pass explicit target hints when they have them.
- keep benchmark matching policy, ambiguity handling, and deduplication here rather than inside adaptation.

#### phase 5: reimplement `ingest` as orchestration

- make `retrocast ingest` a thin wrapper over `adapt + collect`.
- preserve the one-command happy path.
- expose summaries that clearly separate adaptation outcomes from collection outcomes.
- keep `routes.json.gz` as the scoring-facing derived artifact.

#### phase 6: converge training workflows on the shared spine

- refactor training-set prep to reuse shared raw-entry and route-adaptation utilities.
- keep `TrainingRouteRecord`, `TrainingReactionRecord`, and related release models as training-specific projections.
- remove duplicated adaptation logic that exists only because the shared seam was missing.

#### phase 7: retire legacy assumptions

- remove the assumption that adapters consume benchmark-target-keyed blobs directly.
- remove benchmark-centric wording from docs that describe adaptation.
- keep ordering in list order, collector output order, and evaluation-time derived ranks rather than on `Route`.
- delete temporary compatibility bridges once all adapters and workflows use the new seams.

### the docs should describe the system the way users experience it

the docs should no longer center the old implementation accident that adapters require target-keyed blobs.

the docs should start with one copy-paste `ingest` example that works in under a minute.
then they should peel back the layers for users who want `adapt` and `collect` separately.

the docs should center the user journey:

1. the default path is `ingest`.
2. under the hood, readers unpack raw artifacts into route-like entries.
3. adapters normalize those route-like payloads into canonical `Route` objects.
4. collectors align canonical routes to benchmarks when benchmark evaluation is desired.
5. scoring and analysis consume benchmark-keyed route artifacts.
6. training release workflows project the same canonical routes into training-specific records.

the examples should show both library and cli usage.
the docs should say which stage owns which failure mode.

the result should be a library that is easier to explain, easier to extend, and much harder to misuse.
