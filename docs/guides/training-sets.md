---
icon: lucide/database-zap
---

# Training Sets

RetroCast publishes PaRoutes training artifacts as hosted, versioned datasets. The standalone command and both library interfaces call the same native resolver, downloader, and checksum verifier.

## Quick Start

The standalone command downloads into the managed cache:

```bash
retrocast get-training-data reaction-holdout-n1-n5 --split training
```

The library interfaces build the same native request:

=== "Python 0.8.x"

    ```python
    import json
    from pathlib import Path

    import retrocast

    request = {
        "dataset": "paroutes",
        "artifact": "reaction-holdout-n1-n5",
        "split": "training",
        "release": "latest",
        "format": "jsonl",
        "cache_dir": None,
        "output_dir": None,
        "base_url": "https://files.ischemist.com/retrocast/training-sets",
    }
    path = Path(
        retrocast.dataset_download_training_set_json(json.dumps(request))
    )
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::dataset::{
        download_training_set,
        TrainingSetRequest,
    };

    let path = download_training_set(&TrainingSetRequest {
        dataset: "paroutes".to_owned(),
        artifact: "reaction-holdout-n1-n5".to_owned(),
        split: "training".to_owned(),
        release: "latest".to_owned(),
        format: "jsonl".to_owned(),
        cache_dir: None,
        output_dir: None,
        base_url: "https://files.ischemist.com/retrocast/training-sets".to_owned(),
    })?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.datasets import download_training_set

    path = download_training_set(
        "paroutes",
        artifact="reaction-holdout-n1-n5",
        split="training",
    )
    ```

The Python request is JSON because the direct PyO3 binding accepts the same request schema that Rust deserializes into `TrainingSetRequest`. The returned path has already passed checksum verification.

The managed layout is:

```text
~/.cache/retrocast/training-sets/paroutes/<release>/SHA256SUMS
~/.cache/retrocast/training-sets/paroutes/<release>/<artifact>/manifest.json
~/.cache/retrocast/training-sets/paroutes/<release>/<artifact>/<file>
```

Use a project-owned directory when the dataset should live beside an experiment:

```bash
retrocast get-training-data reaction-holdout-n1-n5 \
  --split training \
  --dir data/datasets/paroutes
```

=== "Python 0.8.x"

    ```python
    request["output_dir"] = "data/datasets/paroutes"
    path = Path(
        retrocast.dataset_download_training_set_json(json.dumps(request))
    )
    ```

=== "Rust 0.8.x"

    ```rust
    let mut request = request;
    request.output_dir = Some("data/datasets/paroutes".into());
    let path = download_training_set(&request)?;
    ```

=== "Python 0.7.1"

    ```python
    from pathlib import Path

    path = download_training_set(
        "paroutes",
        artifact="reaction-holdout-n1-n5",
        split="training",
        output_dir=Path("data/datasets/paroutes"),
    )
    ```

That produces:

```text
data/datasets/paroutes/<release>/SHA256SUMS
data/datasets/paroutes/<release>/<artifact>/manifest.json
data/datasets/paroutes/<release>/<artifact>/<file>
```

One-step reaction training uses `single-step-reaction-holdout-n1-n5` or `single-step-route-holdout-n1-n5`. The default `jsonl` format is the canonical structured artifact. Select `rsmi` only when a plain reaction-SMILES projection is required.

The original PaRoutes n1/n5 test sets are published as all-only artifacts: `n1-routes`, `n5-routes`, `n1-single-step-reactions`, and `n5-single-step-reactions`. Use `split="all"` for those.

## Artifact Matrix

Use `reaction-holdout-n1-n5` unless you specifically need a route-holdout baseline.

| Artifact | Intended training target | Holdout rule | Formats | Files published per split |
| --- | --- | --- | --- | --- |
| `n1-routes` | Test set and evaluation routes | Original PaRoutes n1 routes adapted to RetroCast routes | `jsonl` | `all.jsonl.gz` |
| `n5-routes` | Test set and evaluation routes | Original PaRoutes n5 routes adapted to RetroCast routes | `jsonl` | `all.jsonl.gz` |
| `route-holdout-n1-n5` | Multistep route models | Remove exact `n1 ∪ n5` routes | `jsonl` | `all.jsonl.gz`, `training.jsonl.gz`, `validation.jsonl.gz` |
| `reaction-holdout-n1-n5` | Multistep route models | Remove exact holdout routes, then excise holdout reactions | `jsonl` | `all.jsonl.gz`, `training.jsonl.gz`, `validation.jsonl.gz` |
| `n1-single-step-reactions` | Test and evaluation reactions | Flatten original n1 routes while preserving step occurrences | `jsonl`, `rsmi` | `all.jsonl.gz`, `all.rsmi.txt.gz` |
| `n5-single-step-reactions` | Test and evaluation reactions | Flatten original n5 routes while preserving step occurrences | `jsonl`, `rsmi` | `all.jsonl.gz`, `all.rsmi.txt.gz` |
| `single-step-route-holdout-n1-n5` | One-step reaction models | Flatten route-holdout routes into deduplicated reactions; report cross-split overlap | `jsonl`, `rsmi` | all, training, and validation in both formats |
| `single-step-reaction-holdout-n1-n5` | One-step reaction models | Flatten reaction-holdout routes into deduplicated reactions | `jsonl`, `rsmi` | all, training, and validation in both formats |

Each artifact directory contains `manifest.json`. Each release directory contains `SHA256SUMS`.

## Read A Downloaded Artifact

The download function returns a verified local path. Parsing is a separate operation so downstream training code retains explicit ownership of memory and streaming.

=== "Python 0.8.x"

    ```python
    records = json.loads(retrocast.read_jsonl_json(str(path)))
    for record in records:
        ...
    ```

=== "Rust 0.8.x"

    ```rust
    let records = retrocast_core::io::read_jsonl_values(&path, true)?;
    for record in records {
        // ...
    }
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.io import load_training_route_records

    for record in load_training_route_records(path):
        ...
    ```

For corpus-scale training, open the gzip file with the host language's streaming JSONL reader instead of materializing every row through these convenience functions.

## Local Metadata

The sibling artifact manifest and release checksum file are always present:

=== "Python 0.8.x"

    ```python
    manifest_path = path.parent / "manifest.json"
    checksums_path = path.parent.parent / "SHA256SUMS"
    ```

=== "Rust 0.8.x"

    ```rust
    let manifest_path = path.parent().unwrap().join("manifest.json");
    let checksums_path = path.parent().unwrap().parent().unwrap().join("SHA256SUMS");
    ```

=== "Python 0.7.1"

    ```python
    manifest_path = path.parent / "manifest.json"
    checksums_path = path.parent.parent / "SHA256SUMS"
    ```

Keep the returned path as the canonical downloaded artifact. Use the manifest for release provenance and `SHA256SUMS` for an explicit integrity audit.

## Release Resolution

=== "Python 0.8.x"

    ```python
    release = retrocast.dataset_resolve_release(
        "paroutes",
        "latest",
        "https://files.ischemist.com/retrocast/training-sets",
    )
    ```

=== "Rust 0.8.x"

    ```rust
    let release = retrocast_core::dataset::resolve_release(
        "paroutes",
        "latest",
        "https://files.ischemist.com/retrocast/training-sets",
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.datasets import resolve_latest_training_set_release

    release = resolve_latest_training_set_release("paroutes")
    ```

## Canonical Wire Examples

These are the public on-disk wire formats published by the hosted training-set artifacts.

### `route_records`

One row from `training.jsonl.gz` for a route artifact:

```json
{
  "id": "paroutes-reaction-holdout-n1-n5-000001",
  "split": "training",
  "route": {
    "target": {
      "smiles": "CC",
      "inchikey": "CCCCCCCCCCCCCC-DDDDDDDDDD-N",
      "product_of": {
        "reactants": [
          {
            "smiles": "C",
            "inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "annotations": {}
          }
        ],
        "annotations": {}
      },
      "annotations": {}
    },
    "annotations": {},
    "schema_version": "2"
  },
  "sources": [
    {
      "dataset": "all-routes",
      "raw_index": 0,
      "raw_route_hash": "route-hash-1",
      "patent_id": "patent-1"
    }
  ]
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
      "reaction_id": "rc:r:/"
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

When the request sets `output_dir` to `data/datasets/paroutes`, the resulting path is:

```text
data/datasets/paroutes/v2026-05-12/SHA256SUMS
data/datasets/paroutes/v2026-05-12/reaction-holdout-n1-n5/manifest.json
data/datasets/paroutes/v2026-05-12/reaction-holdout-n1-n5/training.jsonl.gz
```

Override the shared cache with:

- `RETROCAST_CACHE_DIR` for the CLI
- `cache_dir` in a Python or Rust request for per-call control

Use `cache_dir` when you want to relocate the shared cache root but still keep the managed `paroutes/<release>/...` structure. Use `output_dir` when you want to own the dataset root yourself.

## CLI API

The CLI builds the same native download request as the library interfaces.

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
