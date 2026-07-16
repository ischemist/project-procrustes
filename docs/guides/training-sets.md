---
icon: lucide/database-zap
---

# Training Sets

RetroCast publishes PaRoutes training artifacts as hosted, versioned datasets. The standalone `retrocast get-training-data` command resolves releases, downloads artifacts, and verifies checksums.

## Quick Start

Use the managed cache when the exact local path is unimportant:

```bash
retrocast get-training-data reaction-holdout-n1-n5 --split training
```

Use `--dir` for a project-owned dataset root:

```bash
retrocast get-training-data reaction-holdout-n1-n5 \
  --split training \
  --dir data/datasets/paroutes
```

The project-owned layout is:

```text
data/datasets/paroutes/<release>/SHA256SUMS
data/datasets/paroutes/<release>/<artifact>/manifest.json
data/datasets/paroutes/<release>/<artifact>/<file>
```

One-step reaction training uses `single-step-reaction-holdout-n1-n5` or `single-step-route-holdout-n1-n5`. The default is canonical JSONL; pass `--format rsmi` for the plain reaction-SMILES projection.

The original PaRoutes n1/n5 test sets are also published as all-only artifacts: `n1-routes`, `n5-routes`, `n1-single-step-reactions`, and `n5-single-step-reactions`. Use `--split all` for those.

## Artifact Matrix

Use `reaction-holdout-n1-n5` unless you specifically need a route-holdout baseline.

| artifact | intended training target | holdout rule | valid `format` values | files published per split |
| --- | --- | --- | --- | --- |
| `n1-routes` | test set/evaluation routes | original PaRoutes n1 routes adapted to RetroCast `Route` records | `jsonl` | `all.jsonl.gz` |
| `n5-routes` | test set/evaluation routes | original PaRoutes n5 routes adapted to RetroCast `Route` records | `jsonl` | `all.jsonl.gz` |
| `route-holdout-n1-n5` | multistep route models | remove exact `n1 ∪ n5` routes | `jsonl` | `all.jsonl.gz`, `training.jsonl.gz`, `validation.jsonl.gz` |
| `reaction-holdout-n1-n5` | multistep route models | remove exact holdout routes, then excise holdout reactions | `jsonl` | `all.jsonl.gz`, `training.jsonl.gz`, `validation.jsonl.gz` |
| `n1-single-step-reactions` | test set/evaluation reactions | flatten original n1 routes, preserving route-step occurrences | `jsonl`, `rsmi` | `all.jsonl.gz`, `all.rsmi.txt.gz` |
| `n5-single-step-reactions` | test set/evaluation reactions | flatten original n5 routes, preserving route-step occurrences | `jsonl`, `rsmi` | `all.jsonl.gz`, `all.rsmi.txt.gz` |
| `single-step-route-holdout-n1-n5` | one-step reaction models | flatten `route-holdout-n1-n5` routes into deduplicated reactions; cross-split reaction overlap is reported, not removed | `jsonl`, `rsmi` | `all.jsonl.gz`, `training.jsonl.gz`, `validation.jsonl.gz`, `all.rsmi.txt.gz`, `training.rsmi.txt.gz`, `validation.rsmi.txt.gz` |
| `single-step-reaction-holdout-n1-n5` | one-step reaction models | flatten `reaction-holdout-n1-n5` routes into deduplicated reactions | `jsonl`, `rsmi` | `all.jsonl.gz`, `training.jsonl.gz`, `validation.jsonl.gz`, `all.rsmi.txt.gz`, `training.rsmi.txt.gz`, `validation.rsmi.txt.gz` |

Each artifact directory includes:

- `manifest.json`

Each release directory includes:

- `SHA256SUMS`

## Canonical Wire Examples

These are the public on-disk wire formats published by the hosted training-set artifacts.

### `route_records`

One row from `training.jsonl.gz` for a route artifact:

```json
{
  "id": "paroutes-reaction-holdout-n1-n5-000001",
  "split": "training",
  "source": {
    "dataset": "all-routes",
    "raw_indices": [0],
    "raw_route_hashes": ["route-hash-1"],
    "patent_ids": ["patent-1"]
  },
  "route": {
    "target": {
      "smiles": "cc",
      "inchikey": "CCCCCCCCCCCCCC-DDDDDDDDDD-N",
      "synthesis_step": {
        "reactants": [
          {
            "smiles": "c",
            "inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "synthesis_step": null,
            "metadata": {},
            "is_leaf": true
          }
        ],
        "mapped_smiles": null,
        "template": null,
        "reagents": null,
        "solvents": null,
        "metadata": {},
        "is_convergent": false
      },
      "metadata": {},
      "is_leaf": false
    },
    "rank": 1,
    "metadata": {},
    "retrocast_version": "0.5.4.dev11",
    "length": 1,
    "leaves": [
      {
        "smiles": "c",
        "inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        "synthesis_step": null,
        "metadata": {},
        "is_leaf": true
      }
    ],
    "has_convergent_reaction": false,
    "content_hash": "2de9f081bc2f5f85de358ecae045a35744de7ad89effe42a5c177c7d4dda5478",
    "signature": "d4fd40da3b1046814438c1c24cc56331747af3ed578ec8244d57b8793a45c1b6"
  }
}
```

### `routes`

`routes` is the nested `route` object from a `route_records` row, validated as a `Route`. The hosted file is still the same `*.jsonl.gz` artifact shown above.

### `reaction_records`

One row from `training.jsonl.gz` for the single-step artifact:

```json
{
  "id": "paroutes-rxn-000001",
  "split": "training",
  "reactants": ["c"],
  "product": "cc",
  "mapped_smiles": "c>o>cc",
  "alternative_mapped_smiles": [],
  "condition_slot": "o",
  "condition_slot_smiles": ["o"],
  "sources": [
    {
      "route_id": "paroutes-reaction-holdout-n1-n5-000001",
      "step_index": 0,
      "source_id": null
    }
  ]
}
```

### `reaction_smiles`

One line from `training.rsmi.txt.gz`:

```text
c>o>cc
```

## Local Layout

Default cache root:

```text
~/.cache/retrocast/training-sets
```

Shared cache layout:

```text
~/.cache/retrocast/training-sets/paroutes/<release>/SHA256SUMS
~/.cache/retrocast/training-sets/paroutes/<release>/<artifact>/manifest.json
~/.cache/retrocast/training-sets/paroutes/<release>/<artifact>/<file>
```

Concrete example:

```text
~/.cache/retrocast/training-sets/paroutes/v2026-05-12/SHA256SUMS
~/.cache/retrocast/training-sets/paroutes/v2026-05-12/reaction-holdout-n1-n5/manifest.json
~/.cache/retrocast/training-sets/paroutes/v2026-05-12/reaction-holdout-n1-n5/training.jsonl.gz
```

When you pass `output_dir=Path("data/datasets/paroutes")`, the resulting path is:

```text
data/datasets/paroutes/v2026-05-12/SHA256SUMS
data/datasets/paroutes/v2026-05-12/reaction-holdout-n1-n5/manifest.json
data/datasets/paroutes/v2026-05-12/reaction-holdout-n1-n5/training.jsonl.gz
```

Override the shared cache with:

- `RETROCAST_CACHE_DIR` for both the CLI and Python
- `cache_dir=...` in Python for per-call control

Use `cache_dir` when you want to relocate the shared cache root but still keep the managed `paroutes/<release>/...` structure. Use `output_dir` when you want to own the dataset root yourself.

## CLI API

The CLI mirrors the Python surface for release downloads.

Download route JSONL:

```bash
retrocast get-training-data reaction-holdout-n1-n5 --split training
```

Download `Route` JSONL for a pinned release:

```bash
retrocast get-training-data reaction-holdout-n1-n5 --split validation --release v2026-05-12
```

Download single-step mapped reaction SMILES:

```bash
retrocast get-training-data single-step-reaction-holdout-n1-n5 --split training --format rsmi
```

Materialize into a project directory instead of the default cache:

```bash
retrocast get-training-data reaction-holdout-n1-n5 --split training --dir data/datasets/paroutes
```

## Conditions

PaRoutes condition slots remain metadata, not structured solvents or reagents, because the source labels are not trustworthy enough for that. A slot may be a solvent, reagent, mixed bag, or even something that should have been modeled as a reactant.

Single-step records expose both:

- `condition_slot`: raw PaRoutes text
- `condition_slot_smiles`: best-effort canonicalized SMILES tokens

Use `condition_slot_smiles` when present. Keep `condition_slot` when you want the original raw signal.
