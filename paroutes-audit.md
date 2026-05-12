# paroutes duplicate audit

## scope

this audit covers duplicate routes encountered during `scripts/paroutes/training-set-prep/01-create-training-release.py`, using the same structural key as release prep: `route.get_structural_signature()`.

inputs audited:

- `data/retrocast/0-assets/paroutes/all-routes.json.gz`
- `data/retrocast/0-assets/paroutes/n1-routes.json.gz`
- `data/retrocast/0-assets/paroutes/n5-routes.json.gz`

release manifests audited:

- `data/retrocast/releases/paroutes-training-sets/v2026-05-12/route-heldout-n1-n5/manifest.json`
- `data/retrocast/releases/paroutes-training-sets/v2026-05-12/reaction-heldout-n1-n5/manifest.json`

## definitions

- `topology duplicate`: same `route.get_structural_signature()`
- `patent-only variant`: same adapted route after stripping `route.metadata.patent_id` and reaction-step source ids
- `reaction template variant`: `paroutes` does not expose a SMARTS template. `ReactionStep.template` therefore stays empty; the `reaction_hash` comparison below was done against raw source data only.
- `condition variant`: there is no structured conditions field in `paroutes`; i used the middle slot of `rsmi` as the only available unstructured reagent/condition string

## topline

- `paroutes` adapted `457,157` usable routes out of `457,166` raw routes. `9` fail adaptation.
- those `457,157` routes collapse to `181,507` unique topology signatures.
- so `275,650` route instances are redundant at the topology level before any n1/n5 holdout.
- every duplicate topology group spans multiple patent ids.
- zero duplicate groups are byte-for-byte identical after adaptation.
- duplicate groups do not carry a true template field from source data.

## release-prep duplication sources

route-heldout release prep decomposes cleanly:

- adapted `all-routes`: `457,157`
- exact heldout matches removed: `50,026`
- remaining topology duplicates removed: `243,495`
- final route-heldout release size: `163,636`

that matches the manifest exactly.

the heldout sets themselves explain part of the exact-match removal:

- `n1` has `10,000` routes and `10,000` unique signatures
- `n5` has `10,000` routes and `10,000` unique signatures
- `n1` and `n5` share `2,129` signatures
- heldout union size is therefore `17,871` unique signatures, which is what the manifest reports
- `all-routes` contains `50,026` routes whose topology is in that heldout union
- of those `50,026`, only `17,871` are the first copy of each heldout topology; the other `32,155` are already redundant duplicates inside `all-routes`

`all-routes` coverage of heldout signatures:

- matches `n1` signatures: `29,498`
- matches `n5` signatures: `26,471`
- matches signatures present in both `n1` and `n5`: `5,943`
- so `n1`-only contributes `23,555`, `n5`-only contributes `20,528`, and the shared slice contributes `5,943`

reaction-heldout release prep adds one more duplication source:

- routes with overlapping heldout reactions: `103,604`
- fragments kept after excision: `101,379`
- routes fully removed after excision: `3,840`
- duplicate routes removed in reaction-heldout mode: `244,625`

vs route-heldout, reaction-heldout removes `1,130` additional duplicate route instances and yields `3,355` fewer final records (`160,281` instead of `163,636`).

## why same-topology routes repeat in paroutes

duplicate topology groups in `all-routes`:

- groups: `113,413`
- raw route instances inside those groups: `389,063`
- redundant route instances inside those groups: `275,650`

breakdown of redundant topology duplicates:

| category | groups | raw instances | redundant instances | share of redundant instances |
| --- | ---: | ---: | ---: | ---: |
| patent-only variants | 107,418 | 355,981 | 248,563 | 90.17% |
| condition-only variants | 5,730 | 31,578 | 25,848 | 9.38% |
| reaction-transform variants (core `rsmi` differs) | 265 | 1,504 | 1,239 | 0.45% |

interpretation:

- the dominant story is not “bad dedup.” it is multiple patents yielding the same route topology.
- most of those multi-patent repeats are identical after adaptation except for patent identity.
- a smaller but real slice keeps the same topology and same transform while changing only the unstructured `rsmi` condition/reagent slot.
- changed core `rsmi` strings under the same topology are rare.
- as an experiment, i temporarily compared those `265` groups using source `reaction_hash` as a transform id. `0 / 265` showed a difference in `reaction_hash`.
- that means the old “reaction transform difference” bucket was not evidence of different source transforms. afaict it was evidence of mapped-`rsmi` drift or provenance-level variation while the source transform id stayed constant.

## important caveats

- `paroutes` still does not supply a SMARTS template string.
- the old transform-variant bucket relied on changed core `rsmi`; when compared against raw source `reaction_hash`, none of those groups show a `reaction_hash` difference.
- bc core `rsmi` can vary while `reaction_hash` stays fixed, that bucket should be read as mapping/provenance drift, not as strong evidence of different chemistry.
- the “condition variant” count is also a PROXY count, using the middle `rsmi` field. that field mixes reagents/solvents/conditions into one opaque string; there is no richer structured condition schema in the source data.
- afaict every topology duplicate group is fully explained by patent id, reaction transform proxy, or `rsmi` condition string. i did not find any leftover “mystery metadata-only” duplicate bucket.
