---
icon: lucide/package-open
---

# Training Set Releases

this page explains how retrocast creates the public paroutes training releases.

the point is to give a compact mental model of the pipeline:

- what artifacts we produce
- what problem each artifact solves
- which functions own each stage
- which tradeoffs the current design makes

this is about **training-set release prep**, not benchmark curation or model
evaluation.

## overview

retrocast currently produces three paroutes-derived training artifacts:

1. `route-heldout-n1-n5`
2. `reaction-heldout-n1-n5`
3. `single-step-reaction-heldout-n1-n5`

the first two are **route releases**. the third is a **flat reaction release**.

all three ultimately come from the same raw paroutes assets:

- `all-routes.json.gz`
- `n1-routes.json.gz`
- `n5-routes.json.gz`

the main question is not how to serialize them. the main question is **what we
want the released dataset to mean**.

### artifact semantics

#### `route-heldout-n1-n5`

this artifact solves a simple multistep leakage problem: if a route structure
already appears in the paroutes `n1 ∪ n5` reference union, it should not remain
in the released training set.

core rule:

- hold out by `Route.get_structural_signature()`

#### `reaction-heldout-n1-n5`

this artifact solves the stricter version of the same problem: route-level
holdout is not enough if a training route contains an embedded heldout
reaction.

core rule:

- remove exact heldout routes first
- then excise heldout reactions from surviving routes
- keep surviving route fragments

#### `single-step-reaction-heldout-n1-n5`

this artifact solves a product problem more than a chemistry problem: we want a
single-step release that is obviously derived from the released multistep
dataset, rather than a totally separate curation pipeline with its own split.

core rule:

- start from the released `reaction-heldout-n1-n5` route artifact
- preserve its `training` / `validation` split
- flatten each split separately
- deduplicate reactions within each split
- report cross-split overlap instead of silently inventing a new split

tradeoff:

- this preserves lineage and gives a coherent artifact family
- but it does **not** guarantee zero flat-reaction overlap across
  `training` and `validation`

that tradeoff is deliberate. for the shipped public artifact, preserving the
route-release partition was judged more important than creating a completely new
reaction-only split.

## core data shapes

the release code lives mainly in
`src/retrocast/curation/training_sets.py`.

the important in-memory / persisted record types are:

- `RawRouteSource`
  - provenance for one raw paroutes route
- `AdaptedTrainingRoute`
  - adapted route plus route/reaction signatures and raw transform sidecar
- `PreparedTrainingRoute`
  - post-holdout, post-dedup route ready for split assignment
- `TrainingRouteRecord`
  - persisted route-release row
- `TrainingReactionSource`
  - provenance for one flattened reaction row
- `PreparedTrainingReaction`
  - in-memory flat reaction candidate
- `TrainingReactionRecord`
  - persisted single-step release row

## adapter contract

some release behavior only makes sense if the paroutes adapter contract is
clear.

for paroutes reaction steps:

- `ReactionStep.mapped_smiles` stores the full mapped `rsmi`
- `ReactionStep.template` stays unset
- `ReactionStep.reagents` stays unset
- `ReactionStep.solvents` stays unset
- the ambiguous middle `rsmi` slot is stored in metadata:
  - `metadata["condition_slot"]`
  - `metadata["condition_slot_smiles"]`

this is solving a trust problem.

we want top-level schema fields to be reliable. paroutes does **not** tell us
which middle-slot molecules are solvents vs reagents, and it does **not**
provide a real generalized template. so that information stays in metadata
instead of being forced into misleading schema fields.

one more important detail:

- paroutes `reaction_hash` is not stored on the public `Route`
- it is used only as a temporary sidecar during raw route curation

that gives us a cleaner public artifact, but it also means some later
split-preserving reaction dedup has to work without access to raw
`reaction_hash`.

## mental model: running `01-create-training-release.py`

entrypoint:

- `scripts/paroutes/training-set-prep/01-create-training-release.py`

main functions involved:

- `load_raw_paroutes_list()`
- `adapt_training_routes()`
- `build_training_records_from_adapted()`
- `prepare_training_routes_from_adapted()`
- `materialize_training_route_records()`
- `write_training_release()`

you can think about `01` as a 6-step pipeline.

### step 1: load raw assets

the script loads:

- `all-routes.json.gz`
- `n1-routes.json.gz`
- `n5-routes.json.gz`

via `load_raw_paroutes_list()`.

`all` is the candidate training universe. `n1` and `n5` define the heldout
reference sets.

### step 2: adapt raw paroutes into retrocast routes

`adapt_training_routes()` converts raw paroutes dictionaries into
`AdaptedTrainingRoute`.

for each route it stores:

- the adapted `Route`
- `structural_signature`
- `reaction_signatures` when reaction holdout is needed
- `RawRouteSource`
- `transform_ids_by_source_id`

`transform_ids_by_source_id` comes from
`extract_paroutes_transform_ids_by_source_id()`, which extracts raw paroutes
`reaction_hash` values keyed by reaction `source_id`.

this is the compromise point in the pipeline:

- the public `Route` object stays schema-honest
- we still retain enough raw provenance to do stronger route-level dedup during
  release prep

### step 3: apply holdout

`prepare_training_routes_from_adapted()` first computes the heldout reference
signatures with `collect_heldout_signatures()`.

then it handles the two route-release modes differently.

#### if `holdout_mode="route"`

the rule is simple:

- drop any candidate route whose `structural_signature` is in the heldout route
  signature set

this produces `route-heldout-n1-n5`.

#### if `holdout_mode="reaction"`

the rule is stricter:

- drop exact heldout routes first
- then call `excise_heldout_reactions()`
- `excise_heldout_reactions()` delegates to
  `retrocast.curation.filtering.excise_reactions_from_route()`
- if a heldout reaction is inside a route, cut it out and keep any surviving
  fragments
- deduplicate those fragments with `deduplicate_routes()`

this produces the candidate pool for `reaction-heldout-n1-n5`.

tradeoff:

- reaction holdout is more useful for leakage control
- but the resulting route release is no longer just a filtered subset of raw
  paroutes; it can contain excision fragments

### step 4: deduplicate routes

after holdout, route dedup happens in two stages.

#### stage 4a: exact chemistry duplicates

`merge_exact_chemical_duplicates()` groups routes by
`get_exact_chemical_signature()`.

today that exact signature is:

- `route.get_annotated_signature(include_mapped_smiles=True)`

when duplicates are merged:

- raw provenance is preserved in `PreparedTrainingRoute.sources`
- singular `metadata["patent_id"]` is removed because it becomes misleading
  after merges
- honest aggregate provenance is written back onto the released route as
  `metadata["source_patent_ids"]` by `sync_route_source_metadata()`

that last point matters. after dedup, the released route is no longer “from
patent x”. it is “supported by patents x, y, z”.

#### stage 4b: transform-equivalent route collapse

`merge_transform_equivalent_routes()` groups routes by
`get_transform_dedup_key()`.

that key includes:

- route structural signature
- per-step condition identity from `get_step_condition_identity()`
- raw paroutes transform ids from `transform_ids_by_source_id`

canonical mapped reactions are chosen by:

1. most frequent mapped-smiles profile
2. lexicographic tie-break
3. raw-route-hash tie-break between equally weighted candidates

non-canonical mapped variants are preserved on the kept route step via
`merge_alternative_mapped_smiles()`, which writes:

- `ReactionStep.metadata["alternative_mapped_smiles"]`

### step 5: assign route split

`materialize_training_route_records()` assigns `training` / `validation` with
`assign_train_val_splits()`.

the stratification key is:

- `route.length`
- `route.has_convergent_reaction`

the point of assigning the split here, rather than earlier, is that we only
split the final released route population, not raw candidates that may later be
removed or merged.

### step 6: write the route release

`write_training_release()` writes:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `manifest.json`

the manifest comes from `build_training_manifest()`.

## mental model: running `02-create-single-step-release.py`

entrypoint:

- `scripts/paroutes/training-set-prep/02-create-single-step-release.py`

main functions involved:

- `load_training_route_records()`
- `build_training_reaction_records_from_route_records()`
- `flatten_training_route_records_to_reactions()`
- `merge_exact_reaction_duplicates()`
- `merge_transform_equivalent_reactions()`
- `summarize_split_preserving_reaction_overlap()`
- `write_training_reaction_release()`

the key point is that `02` does **not** re-adapt raw paroutes. it starts from
the already released `reaction-heldout-n1-n5` route artifact.

you can think about `02` as a 5-step pipeline.

### step 1: load the released route artifact

the script loads:

- `reaction-heldout-n1-n5/training.jsonl.gz`
- `reaction-heldout-n1-n5/validation.jsonl.gz`

using `load_training_route_records()`, which reconstructs
`TrainingRouteRecord` with:

- the `Route`
- the split
- route signature
- content hash
- route provenance

this keeps the single-step release visibly derived from the route release,
rather than from a second hidden curation pass over raw paroutes.

### step 2: flatten routes into reactions, split by split

`build_training_reaction_records_from_route_records()` first groups route
records by their existing split:

- `training`
- `validation`

then `flatten_training_route_records_to_reactions()` walks each route and emits
`PreparedTrainingReaction`.

each flattened reaction keeps:

- `reactants`
- `product`
- `mapped_smiles`
- `alternative_mapped_smiles`
- `condition_slot`
- `condition_slot_smiles`
- `TrainingReactionSource`

that provenance shape matters because every released reaction row can still be
traced back to the route record that produced it.

### step 3: deduplicate reactions within each split

each split is deduplicated independently in two stages.

#### stage 3a: exact flat reaction duplicates

`merge_exact_reaction_duplicates()` groups by
`get_exact_reaction_signature()`.

that exact signature includes:

- `mapped_smiles`
- `condition_slot_smiles`
- fallback `condition_slot`

#### stage 3b: mapping-drift collapse

`merge_transform_equivalent_reactions()` groups by
`get_transform_reaction_dedup_key()`.

for the split-preserving route-derived release, that key is effectively:

- `reactants`
- `product`
- condition identity

note the limitation here:

- raw paroutes `reaction_hash` is not present in the released route artifact
- so `02` cannot use it for flat-reaction dedup

tradeoff:

- the split-preserving single-step release has a weaker mapping-drift collapse
  rule than the experimental raw-adapted reaction builder

### step 4: measure cross-split overlap

after within-split dedup,
`summarize_split_preserving_reaction_overlap()` computes:

- `shared_exact_reaction_signatures`
- `shared_reaction_identities`
- `training_records_with_shared_identity`
- `validation_records_with_shared_identity`

tradeoff:

- this release is easier to reason about as a projection of the route release
- but it is not a strict leakage-free single-step benchmark

if strict reaction-level split hygiene is the goal, the other builder,
`build_training_reaction_records_from_adapted()`, is the more appropriate
starting point. that builder exists for experiments, but it is **not** the
public single-step release path.

### step 5: write the single-step release

`write_training_reaction_release()` writes:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `all.rsmi.txt.gz`
- `training.rsmi.txt.gz`
- `validation.rsmi.txt.gz`
- `manifest.json`

the structured `jsonl.gz` files are the canonical artifact.

the `*.rsmi.txt.gz` files are convenience outputs for users who only want one
mapped reaction smiles per line.

## where to change behavior

if you need to change the release pipeline, these are the right seams.

### change route holdout behavior

touch:

- `collect_heldout_signatures()`
- `excise_heldout_reactions()`
- `prepare_training_routes_from_adapted()`

### change route dedup behavior

touch:

- `get_exact_chemical_signature()`
- `get_transform_dedup_key()`
- `merge_exact_chemical_duplicates()`
- `merge_transform_equivalent_routes()`

### change single-step split-preserving dedup behavior

touch:

- `get_exact_reaction_signature()`
- `get_transform_reaction_dedup_key()`
- `merge_exact_reaction_duplicates()`
- `merge_transform_equivalent_reactions()`
- `build_training_reaction_records_from_route_records()`

### change public artifact schema

touch:

- `TrainingRouteRecord`
- `TrainingReactionRecord`
- `load_training_route_records()`
- `load_training_reaction_records()`
- `write_training_release()`
- `write_training_reaction_release()`

## shortest summary

when you run `01-create-training-release.py`, retrocast:

1. adapts raw paroutes
2. applies route or reaction holdout
3. deduplicates routes in two stages
4. assigns the final route split
5. writes the two route releases

when you run `02-create-single-step-release.py`, retrocast:

1. loads the released `reaction-heldout-n1-n5` route artifact
2. preserves its split
3. flattens each split into reactions
4. deduplicates within each split
5. reports cross-split overlap
6. writes the single-step release

that is the current intended model. it keeps the route releases strict where
they need to be, and keeps the public single-step release understandable as a
projection of the released route dataset.
