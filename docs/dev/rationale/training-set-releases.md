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

[PaRoutes](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00015f) ([GitHub](https://github.com/MolecularAI/PaRoutes)) is a landmark first-in-class effort to curate a large-scale, high-quality dataset of multistep synthesis plans extracted from patent literature. The original paper provided three artifacts: full set of multistep routes (`all-routes`) and two carefully constructed test subsets `n1-routes` and `n5-routes`. Unfortunately, the authors did not provide a canonical training set split, which results in either refusal to adopt PaRoutes as training/test set (e.g. [DESP](https://arxiv.org/abs/2407.06334)) or inconsistent splitting:

- [DirectMultiStep](https://directmultistep.com) performs a _route holdout_: removing n1 and n5 routes from the `all-routes`
- [TempRe](https://arxiv.org/abs/2507.21762) performs a _reaction holdout_: removing all single step reactions from n1 and n5 from `all-routes`

Given the recent inclusion of [synthesis planning in the list of Grand Challenges in Drug Discovery](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000615/v1) and the subsequent [proposal of synthesis planning as a pretraining objective](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001278/v3), one might reasonably expect an influx of new researchers to the field, and so standardization of the training set preparation is a top priority.

RetroCast is an open-source effort, so we do not release this as an authoritative final say, but rather we invite scrutiny and feedback from the community.

!!! info "Why does standardization of implementation of the split matters?"

    Separating test routes from the `all-routes` is not as straightforward as it might seem because the correct procedure depends on the representation of the data. For example, if you represent routes as nested/bigraph dictionaries, you need to come up with serialization strategy if you want to exclude by containment check (you can't put dictionary in a set). Or even if you're willing to pay the price of O(n) comparison, you still need to make sure your equality check is permutation-invariant (a route A + B -> C (+ D) -> E is the same regardless of whether C or D is the left child). [DirectMultiStep](https://directmultistep.com) implemented generator of all permutations of routes and used flattening serialization. In RetroCast, we utilize route signatures (see more below). The point here is not that this is an unusually algorithmically hard problem, but rather that it requires careful consideration and there's no reason for every single model developer to have to implement this themselves.

## Overview

RetroCast produces PaRoutes-derived training and test set artifacts from `all-routes.json.gz`, `n1-routes.json.gz`, and `n5-routes.json.gz`.

Training artifacts:

1. `route-holdout-n1-n5`
2. `reaction-holdout-n1-n5`
3. `single-step-route-holdout-n1-n5`
4. `single-step-reaction-holdout-n1-n5`

Test set artifacts:

1. `n1-routes`
2. `n5-routes`
3. `n1-single-step-reactions`
4. `n5-single-step-reactions`

`all` is the candidate training universe. `n1` and `n5` are the holdout test sets.

## Route Releases

Entrypoint:

- `scripts/paroutes/training-set-prep/01-create-training-release.py`

Implementation:

- `adapt_training_routes()` converts raw PaRoutes dictionaries into RetroCast `Route` objects and records provenance.
- `TrainingRouteReleaseBuilder` applies holdout, deduplication, and split assignment.
- `write_training_release()` writes the final artifact.

The route release has two modes.

`route-holdout-n1-n5` removes every candidate route whose schema-2 `Route.signature()` appears in `n1 ∪ n5`.

`reaction-holdout-n1-n5` first removes exact holdout routes, then removes holdout reactions from surviving routes. If a route still has valid fragments after excision, those fragments remain candidates.

During adaptation, RetroCast sanity-checks PaRoutes `reaction_hash` values against `ReactionSignature`. PaRoutes `reaction_hash` is reaction SMILES represented with InChIKeys, so it should describe the same identity as RetroCast's reactant/product InChIKey signature. If the check passes, `reaction_hash` is not carried through the release pipeline.

PaRoutes condition slots stay in reaction annotations. We do not populate structured `solvents` or `reagents` fields because the slot is not reliably one or the other; it can also contain material that should have been modeled as a reactant. RetroCast tries to canonicalize the slot into `condition_slot_smiles`, but keeps the raw `condition_slot` when parsing fails or when the raw text may still help an end user.

## Route Deduplication

Routes are deduplicated twice:

1. exact annotated chemistry: `route.get_annotated_signature(include_mapped_smiles=True)`
2. same route structure and conditions, different atom mapping

The second pass groups by route structural signature plus per-step condition identity. Condition identity is `condition_slot_smiles` when available and `condition_slot` otherwise.

When mapped variants collapse, the kept mapped profile is chosen by source support, then lexicographic order, then raw route hash. Non-kept mapped reactions are preserved in reaction annotations as `alternative_mapped_smiles`.

Merged route provenance stays in `sources`. Released route metadata drops the single-source `patent_id` and writes `source_patent_ids` at materialization.

## Splits And Files

Route releases assign `training` / `validation` after holdout and deduplication. The split is stratified by route length and whether the route is convergent, using `val_fraction` and `seed` from config.

Each route release writes:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `manifest.json`

## Single-Step Release

Entrypoint:

- `scripts/paroutes/training-set-prep/02-create-single-step-release.py`

Implementation:

- `load_training_route_records()` reads released route artifacts.
- `TrainingReactionReleaseBuilder` flattens routes into reactions, deduplicates each split, and applies holdout-specific split cleanup.
- `write_training_reaction_release()` writes the final artifact.

The single-step releases do not re-adapt raw PaRoutes. They derive from released route artifacts so the single-step predictor and multistep planner train from compatible data.

`single-step-route-holdout-n1-n5` derives from `route-holdout-n1-n5`. Because route holdout only excludes exact holdout routes, flattened training and validation splits may share reaction identities. RetroCast reports that overlap in the build summary and audit instead of removing it.

`single-step-reaction-holdout-n1-n5` derives from `reaction-holdout-n1-n5`. Because this artifact is intended for one-step reaction training, validation reactions that overlap training are removed after split-level deduplication.

Each flattened reaction keeps reactants, product, mapped smiles, alternative mapped smiles, condition metadata, and route-step provenance. Reaction sources store `route_id`, `step_index`, and optional PaRoutes `source_id`; raw route hashes and patent ids stay in the parent route release.

The structured `jsonl.gz` files are canonical. The `*.rsmi.txt.gz` files are convenience exports.

## Test Set Releases

Entrypoint:

- `scripts/paroutes/training-set-prep/06-create-test-set-release.py`

`n1-routes` and `n5-routes` adapt the original PaRoutes test set routes into RetroCast `Route` records. They publish only `all.jsonl.gz`.

`n1-single-step-reactions` and `n5-single-step-reactions` flatten those adapted test set routes into route-step reaction records. They are occurrence-preserving: if the same reaction appears in multiple test set routes, or multiple times inside one test set route, each occurrence remains in the release with route-step provenance. They publish `all.jsonl.gz` and `all.rsmi.txt.gz`.

## Audit

Run:

- `scripts/paroutes/training-set-prep/03-audit-release.py`

The audit checks release counts, split balance, route/reaction overlap, and metadata expectations.

## Change Points

Route release behavior lives in:

- `packages/retrocast-rs/python/retrocast/curation/training/route_release.py`
- `packages/retrocast-rs/python/retrocast/curation/filtering.py`
- `packages/retrocast-rs/python/retrocast/curation/training/records.py`

Single-step release behavior lives in:

- `packages/retrocast-rs/python/retrocast/curation/training/reaction_release.py`
- `packages/retrocast-rs/python/retrocast/curation/training/records.py`

PaRoutes adaptation behavior lives in:

- `packages/retrocast-rs/python/retrocast/adapters/paroutes.py`
