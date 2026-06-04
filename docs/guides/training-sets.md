---
icon: lucide/database-zap
---

# Training Sets

RetroCast publishes PaRoutes training artifacts as hosted, versioned datasets. The official Python workflow is centered on `retrocast.datasets`. Use the shell helper when you need a download path outside Python.

!!! warning "Requires the hosted training-set API"

    This guide assumes a RetroCast build that includes
    `retrocast.datasets.download_training_set`. That surface landed after `v0.5.3`
    in commit `2c6387a`.

    If your install does not expose `retrocast.datasets`, upgrade before
    running the examples on this page.

## Quick Start

Pick one of two workflows:

=== "managed cache"

    use this when you do not care where the files live beyond “some verified local cache”.

    python:

    ```python
    from retrocast.datasets import download_training_set
    from retrocast.io import iter_training_routes

    path = download_training_set(
        "paroutes",
        artifact="reaction-holdout-n1-n5",
        split="training",
    )

    for route in iter_training_routes(path):
        ...
    ```

    shell fallback:

    ```bash
    curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split training
    ```

    resulting local layout:

    ```text
    ~/.cache/retrocast/training-sets/paroutes/<release>/SHA256SUMS
    ~/.cache/retrocast/training-sets/paroutes/<release>/<artifact>/manifest.json
    ~/.cache/retrocast/training-sets/paroutes/<release>/<artifact>/<file>
    ```

=== "project-owned directory"

    use this when you want to choose the dataset root yourself. pass a dataset-specific directory such as `data/datasets/paroutes`, and retrocast will place releases directly under it.

    python:

    ```python
    from pathlib import Path

    from retrocast.datasets import download_training_set
    from retrocast.io import iter_training_routes

    path = download_training_set(
        "paroutes",
        artifact="reaction-holdout-n1-n5",
        split="training",
        output_dir=Path("data/datasets/paroutes"),
    )

    for route in iter_training_routes(path):
        ...
    ```

    shell fallback:

    ```bash
    curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split training --dir data/datasets/paroutes
    ```

    resulting local layout:

    ```text
    data/datasets/paroutes/<release>/SHA256SUMS
    data/datasets/paroutes/<release>/<artifact>/manifest.json
    data/datasets/paroutes/<release>/<artifact>/<file>
    ```

One-step reaction training uses the same flow with `artifact="single-step-reaction-holdout-n1-n5"` or `artifact="single-step-route-holdout-n1-n5"`. By default that downloads the canonical `jsonl` artifact. Pass `format="rsmi"` if you specifically want the plain reaction-smiles text file.

The original PaRoutes n1/n5 test sets are also published as all-only artifacts: `n1-routes`, `n5-routes`, `n1-single-step-reactions`, and `n5-single-step-reactions`. Use `split="all"` for those.

When a real download happens in an interactive terminal, RetroCast shows a progress bar automatically. Pass `show_progress=False` to suppress it or `show_progress=True` to force it.

## Public Imports

The stable public import path for training-set download helpers is `retrocast.datasets`:

```python
from retrocast.datasets import download_training_set, resolve_latest_training_set_release
```

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

## Python API

`download_training_set()` gives you the verified local artifact path. it also materializes sibling `manifest.json` and release-level `SHA256SUMS`. `format` describes which wire file you want:

- `jsonl` -> structured JSONL artifact
- `rsmi` -> plain reaction-smiles text artifact

Use `jsonl` unless you explicitly want the single-step reaction-smiles text projection. route artifacts only support `jsonl`.

### Streaming

Use `retrocast.io` when you want to stream a verified local artifact without loading the full file into memory:

```python
from retrocast.datasets import download_training_set
from retrocast.io import iter_training_reaction_records

path = download_training_set("paroutes", artifact="single-step-reaction-holdout-n1-n5", split="training")

for record in iter_training_reaction_records(path):
    ...
```

Available local streaming helpers:

- `retrocast.io.iter_training_routes(path)`
- `retrocast.io.iter_training_route_records(path)`
- `retrocast.io.iter_training_reaction_records(path)`
- `retrocast.io.iter_training_reaction_smiles(path)`

Available local eager helpers:

- `retrocast.io.load_training_routes(path)`
- `retrocast.io.load_training_route_records(path)`
- `retrocast.io.load_training_reaction_records(path)`
- `retrocast.io.load_training_reaction_smiles(path)`

The intended split is:

- `retrocast.datasets` resolves releases, downloads artifacts, and verifies checksums
- `retrocast.io` parses eager or streaming views from a local path

That keeps the local artifact path explicit, which is usually useful in real training pipelines.

### Local Metadata

`download_training_set()` returns the verified local `Path`:

```python
from retrocast.datasets import download_training_set

path = download_training_set(
    "paroutes",
    artifact="reaction-holdout-n1-n5",
    split="training",
)
```

The sibling artifact manifest and release checksum file are always there:

```python
from retrocast.datasets import download_training_set

path = download_training_set(
    "paroutes",
    artifact="reaction-holdout-n1-n5",
    split="training",
)

manifest_path = path.parent / "manifest.json"
checksums_path = path.parent.parent / "SHA256SUMS"

print(path)
print(manifest_path)
print(checksums_path)
```

That is usually enough for downstream training pipelines:

- keep `path` as the canonical downloaded artifact
- inspect `manifest.json` later for release provenance and build metadata
- compare a local `sha256` against `SHA256SUMS` if you need an explicit audit step

### Release Resolution

For the common case, use `resolve_latest_training_set_release()`:

```python
from retrocast.datasets import resolve_latest_training_set_release

release = resolve_latest_training_set_release("paroutes")
```

`resolve_training_set_release()` is still available when you want to resolve a specific label yourself:

```python
from retrocast.datasets import resolve_training_set_release

release = resolve_training_set_release(dataset="paroutes", release="latest")
```

### Explicit Output Directories

Use `output_dir=...` when you want a project-owned dataset root instead of the managed cache. this path is treated as the root for one dataset, so releases land directly under it with no extra `retrocast/training-sets/<dataset>` scaffolding:

```python
from pathlib import Path

from retrocast.datasets import download_training_set

path = download_training_set(
    "paroutes",
    artifact="reaction-holdout-n1-n5",
    split="training",
    output_dir=Path("data/datasets/paroutes"),
)
```

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

- `RETROCAST_CACHE_DIR` for both shell and Python
- `RETROCAST_TRAINING_SET_CACHE_DIR` for the shell helper's training-set cache only
- `cache_dir=...` in Python for per-call control

Use `cache_dir` when you want to relocate the shared cache root but still keep the managed `paroutes/<release>/...` structure. Use `output_dir` when you want to own the dataset root yourself.

## Shell API

The shell helper mirrors the Python surface, but it is the fallback story, not the primary one.

Download route JSONL:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split training
```

Download `Route` JSONL for a pinned release:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split validation --release v2026-05-12
```

Download single-step mapped reaction SMILES:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- single-step-reaction-holdout-n1-n5 --split training --format rsmi
```

Materialize into a project directory instead of the default cache:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split training --dir data/datasets/paroutes
```

## Conditions

PaRoutes condition slots remain metadata, not structured solvents or reagents, because the source labels are not trustworthy enough for that. A slot may be a solvent, reagent, mixed bag, or even something that should have been modeled as a reactant.

Single-step records expose both:

- `condition_slot`: raw PaRoutes text
- `condition_slot_smiles`: best-effort canonicalized SMILES tokens

Use `condition_slot_smiles` when present. Keep `condition_slot` when you want the original raw signal.
