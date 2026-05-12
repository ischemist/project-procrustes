---
icon: lucide/package-open
---

# Training Set Releases

This page explains how retrocast creates the public PaRoutes training releases. This doc aims to give a compact mental model of the pipeline:

- what artifacts we produce
- what problem each artifact solves
- which functions own each stage
- which tradeoffs the current design makes

## Historical Context

[PaRoutes](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00015f) ([Github](https://github.com/MolecularAI/PaRoutes)) is a landmark first-in-class effort to curate a large-scale, high-quality dataset of multistep synthesis plans extracted from patent literature. The original paper provided three artifacts: full set of multistep routes (`all-routes`) and two carefully constructed test subsets `n1-routes` and `n5-routes`. Unfortunately, the authors did not provide a canonical training set split, which results in either refusal to adopt PaRoutes as training/test set (e.g. [DESP](https://arxiv.org/abs/2407.06334)) or inconsistent splitting:

- [DirectMultiStep](https://directmultistep.com) performs a _route heldout_: removing n1 and n5 routes from the `all-routes`
- [TempRe](https://arxiv.org/abs/2507.21762) performs a _reaction heldout_: removing all single step reactions from n1 and n5 from `all-routes`

Given the recent inclusion of [synthesis planning in the list of Grand Challenges in Drug Discovery](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000615/v1) and the subsequent [proposal of synthesis planning as a pretraining objective](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001278/v3), one might reasonably expect an influx of new researchers to the field, and so standardization of the training set preparation is a top priority.

RetroCast is an open-source effort, so we do not release this as an authoritative final say, but rather we invite scrutiny and feedback from the community.

!!! info "Why does standardization of implementation of the split matters?"

    Separating test routes from the `all-routes` is not as straightforward as it might seem because the correct procedure depends on the representation of the data. For example, if you represent routes as nested/bigraph dictionaries, you need to come up with serialization strategy if you want to exclude by containment check (you can't put dictionary in a set). Or even if you're willing to pay the price of O(n) comparison, you still need to make sure your equality check is permutation-invariant (a route A + B -> C (+ D) -> E is the same regardless of whether C or D is the left child). [DirectMultiStep](https://directmultistep.com) implemented generator of all permutations of routes and used flattening serialization. In RetroCast, we utilize route signatures (see more below). The point here is not that this is an unusually algorithmically hard problem, but rather that it requires careful consideration and there's no reason for every single model developer to have to implement this themselves.


## Overview

RetroCast currently produces three paroutes-derived training artifacts:

1. `route-heldout-n1-n5`
2. `reaction-heldout-n1-n5`
3. `single-step-reaction-heldout-n1-n5`

The first two are **route releases**, and the third is a **flat reaction release**. All three ultimately come from the same raw paroutes assets `all-routes.json.gz`, `n1-routes.json.gz`, and `n5-routes.json.gz`.

### `route-heldout-n1-n5`

This is a DirectMultiStep-style dataset that removes route structures that appear in the `n1 ∪ n5` reference union. Internally, this is achieved by comparing `Route.get_structural_signature()` of each route to the heldout signature set.

### `reaction-heldout-n1-n5`

This is a TempRe-style dataset that implements a stricter holdout: no single-step reaction contained in any route of `n1 ∪ n5` can be present in any route of the training set.

Internally, this is achieved by:

- removing exact heldout routes first (`Route.get_structural_signature()`)
- then excising heldout reactions from surviving routes

### `single-step-reaction-heldout-n1-n5`

A prevalent approach to multistep planning explicitly separates a single-step reaction predictor from a multistep planner. This dataset is derived from the released `reaction-heldout-n1-n5` route artifact by flattening the `training` and `validation` splits into single-step reaction sets.

We intentionally decide to derive this dataset from the route-based release rather than a separate curation pipeline because single-step prediction problem is primarily a problem we're interested in in the context of multistep planning, so having a separate curation pipeline would present unnecessary challenges in the construction of the multistep routes for subsequent training of the planner.

Release preparation workflow:

- start from the released `reaction-heldout-n1-n5` route dataset
- preserve its `training` / `validation` split
- flatten each split separately
- deduplicate reactions within each split
- remove validation reactions that still overlap with training
- report overlap before and after cleanup


## Core Data Shapes

the release code lives mainly in
`src/retrocast/curation/training_sets.py`. the important in-memory / persisted record types are:

- `RawRouteSource` provenance for one raw paroutes route
- `AdaptedTrainingRoute` adapted route plus route/reaction signatures and raw transform sidecar
- `PreparedTrainingRoute` post-holdout, post-dedup route ready for split assignment
- `TrainingRouteRecord` persisted route-release row
- `TrainingReactionSource` provenance for one flattened reaction row
- `PreparedTrainingReaction` in-memory flat reaction candidate
- `TrainingReactionRecord` persisted single-step release row

## Mental Model: Running `01-create-training-release.py`

Entrypoint:

- `scripts/paroutes/training-set-prep/01-create-training-release.py`

Main functions involved:

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

During adaptation, RetroCast sanity-checks PaRoutes `reaction_hash` values against
RetroCast reaction signatures. PaRoutes hashes are reaction identities composed from
InChIKeys, so they should be equivalent to our `(reactant inchikeys, product inchikey)`
signature. If that equivalence fails, release prep raises instead of carrying a
PaRoutes-specific identity sidecar through the pipeline.

!!! note

    in v2026-05-11 release, `all-routes` contains 457 166 entries, of which 457 157 are succesfully adapted into RetroCast schema (failures by error: 3 `adapter.cycle_detected` and 6 `chem.invalid_smiles`)

### step 3: apply holdout

`prepare_training_routes_from_adapted()` first computes the heldout reference
signatures with `collect_heldout_signatures()`.

then it handles the two route-release modes differently.

#### if `holdout_mode="route"`

the rule is simple: drop any candidate route whose `structural_signature` is in the heldout route signature set

this produces `route-heldout-n1-n5`.

!!! note

    in v2026-05-11 release, this step removes 50 026 entries

#### if `holdout_mode="reaction"`

the rule is stricter:

- drop exact heldout routes first
- then call `excise_heldout_reactions()` which relies on `retrocast.curation.filtering.excise_reactions_from_route()`
- if a heldout reaction is inside a route, cut it out and keep any surviving fragments
- deduplicate those fragments with `deduplicate_routes()`

this produces the candidate pool for `reaction-heldout-n1-n5`.

!!! note

    in v2026-05-11 release, after 50 026 exact matches are removed, 103 604 routes are subject to excision, which results in complete removal of 3 840 routes and produces 101 379 fragmented routes. 

### step 4: deduplicate routes

There are two sources of duplication in the original PaRoutes. 

1. Some routes are chemical duplicates that were extracted from different patents, and as such differ in the patent ID associated with them. We merge these in stage 4a.
2. Some routes are structurally identical but contain reactions that are atom mapped (annotated) differently. 

#### stage 4a: exact chemistry duplicates

`merge_exact_chemical_duplicates()` groups routes by
`get_exact_chemical_signature()`.

today that exact signature is:

- `route.get_annotated_signature(include_mapped_smiles=True)`

when duplicates are merged:

- raw provenance is preserved in `PreparedTrainingRoute.sources`
- `metadata["patent_id"]` is replaced with `metadata["source_patent_ids"]` by `sync_route_source_metadata()`. after dedup, the released route is no longer "from patent x", it is "supported by patents x, y, z".

!!! note

    in v2026-05-11 release, this step removes 237 923 and 239 862 duplicates from route and reaction holdout releases respectively

#### stage 4b: transform-equivalent route collapse

`merge_transform_equivalent_routes()` groups routes by
`get_transform_dedup_key()`.

that key includes:

- route structural signature `route.get_structural_signature()`
- per-step condition identity from `get_step_condition_identity()`

canonical mapped reactions are chosen by:

1. most frequent mapped-smiles profile
2. lexicographic tie-break
3. raw-route-hash tie-break between equally weighted candidates

non-canonical mapped variants are preserved on the kept route step via
`merge_alternative_mapped_smiles()`, which writes to `ReactionStep.metadata["alternative_mapped_smiles"]`

!!! note

    in v2026-05-11 release, this step merges 82 and 71 duplicates from route and reaction holdout releases respectively

### step 5: assign route split

`materialize_training_route_records()` assigns `training` / `validation` with
`assign_train_val_splits()`.

the stratification key is:

- `route.length`
- `route.has_convergent_reaction`

!!! tip

    you can verify that the training and validation splits are distributionally similar by running `scripts/paroutes/training-set-prep/03-audit-release.py`

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
- `summarize_cross_split_reaction_overlap()`
- `drop_cross_split_validation_overlap()`
- `write_training_reaction_release()`

the key point is that `02` does **not** re-adapt raw paroutes. it starts from the already released `reaction-heldout-n1-n5` route artifact.

you can think about `02` as a 5-step pipeline.

### step 1: load the released route artifact

the script loads:

- `reaction-heldout-n1-n5/training.jsonl.gz`
- `reaction-heldout-n1-n5/validation.jsonl.gz`

using `load_training_route_records()`, which reconstructs `TrainingRouteRecord`

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
- `TrainingReactionSource` (every released reaction row can still be traced back to the route record that produced it)

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

for the route-release-derived public single-step release, that key is effectively:

- `reactants`
- `product`
- condition identity

### step 4: remove cross-split validation overlap

after within-split dedup,
`summarize_cross_split_reaction_overlap()` computes the overlap snapshot before
cleanup:

- `shared_exact_reaction_signatures`
- `shared_reaction_identities`
- `training_records_with_shared_identity`
- `validation_records_with_shared_identity`

then `drop_cross_split_validation_overlap()` removes validation reactions whose
identity already appears in training.

the identity used for this cleanup is:

- `reactants`
- `product`
- condition identity

after cleanup, `summarize_cross_split_reaction_overlap()` runs again and the
public release expects zero remaining shared reaction identities.

tradeoff:

- this is a stricter and safer default for single-step model development
- but it means the released validation split is not simply “whatever reactions
  happened to come from route-validation records”

if strict reaction-level split hygiene is the goal, the other builder,
`build_training_reaction_records_from_adapted()`, is the more appropriate
starting point for experiments that want to rebuild single-step data directly
from raw-adapted routes. that builder exists for experiments, but it is **not**
the public single-step release path.

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

### change single-step release dedup behavior

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
5. removes validation reactions that overlap with training
6. writes the single-step release

that is the current intended model. it keeps the route releases strict where
they need to be, and makes the public single-step release safer to use as an
actual training/validation artifact.
