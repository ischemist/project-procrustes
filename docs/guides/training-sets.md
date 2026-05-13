---
icon: lucide/database-zap
---

# Training Sets

RetroCast publishes PaRoutes training artifacts as hosted, versioned datasets.
You should be able to do one of two things:

- download a split from the shell without memorizing URLs
- load a split in Python as `list[Route]`, route records, reaction records, or
  mapped reaction SMILES

The interface below is the intended product surface.

## Quick Start

Shell:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split training
```

Python:

```python
from retrocast.datasets import load_training_set

train_routes = load_training_set(
    "paroutes",
    artifact="reaction-holdout-n1-n5",
    split="training",
    as_="routes",
)
```

One-step reaction training:

```python
from retrocast.datasets import load_training_set

train_reactions = load_training_set(
    "paroutes",
    artifact="single-step-reaction-holdout-n1-n5",
    split="training",
    as_="reaction_records",
)
```

## Artifacts

Use `reaction-holdout-n1-n5` unless you specifically need a route-holdout
baseline.

| artifact | use it for | guarantee |
| --- | --- | --- |
| `route-holdout-n1-n5` | multistep route models | exact `n1 ∪ n5` routes removed |
| `reaction-holdout-n1-n5` | multistep route models | exact routes removed, then holdout reactions excised |
| `single-step-reaction-holdout-n1-n5` | one-step reaction models | flattened from `reaction-holdout-n1-n5` |

Route artifacts expose:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `manifest.json`

Single-step artifacts expose:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `all.rsmi.txt.gz`
- `training.rsmi.txt.gz`
- `validation.rsmi.txt.gz`
- `manifest.json`

## Python API

`load_training_set()` is the high-level entrypoint:

```python
from retrocast.datasets import load_training_set

val_records = load_training_set(
    "paroutes",
    artifact="reaction-holdout-n1-n5",
    split="validation",
    as_="route_records",
    release="latest",
)
```

Supported `as_` values:

- `routes` -> `list[Route]`
- `route_records` -> `list[TrainingRouteRecord]`
- `reaction_records` -> `list[TrainingReactionRecord]`
- `reaction_smiles` -> `list[str]`

Low-level control stays available:

```python
from retrocast.datasets import download_training_set
from retrocast.io import load_training_route_records, load_training_routes

path = download_training_set(
    "paroutes",
    artifact="reaction-holdout-n1-n5",
    split="training",
    release="latest",
    as_="routes",
)

records = load_training_route_records(path)
routes = load_training_routes(path)
```

`download_training_set()`:

- resolve `latest`
- download the requested file into a local cache
- verify the artifact before returning it
- return the local `Path`

Use `output_dir=...` when you want the artifact materialized into an explicit
project-owned location instead of the managed cache:

```python
from pathlib import Path

from retrocast.datasets import download_training_set

path = download_training_set(
    "paroutes",
    artifact="reaction-holdout-n1-n5",
    split="training",
    as_="routes",
    output_dir=Path("data/training"),
)
```

## Shell API

The shell path mirrors the Python path. `get-training-set.sh` defaults to
`release=latest` and prints the local path it downloaded.

Download route records:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split training
```

Download `Route` JSONL for a pinned release:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split validation --release v2026-05-12
```

Download single-step mapped reaction SMILES:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- single-step-reaction-holdout-n1-n5 --split training --format reaction-smiles
```

Materialize into a project directory instead of the default cache:

```bash
curl -fsSL https://files.ischemist.com/retrocast/get-training-set.sh | bash -s -- reaction-holdout-n1-n5 --split training --dir data/training
```

## Release Resolution

Users should be able to rely on three things:

- `release="latest"` resolves to the newest published release
- pinned releases like `v2026-05-12` remain stable
- artifact downloads are verified before they are returned to Python or shell callers

## Conditions

PaRoutes condition slots remain metadata, not structured solvents/reagents,
because the source labels are not trustworthy enough for that. A slot may be a
solvent, reagent, mixed bag, or even something that should have been modeled as
a reactant.

Single-step records expose both:

- `condition_slot`: raw PaRoutes text
- `condition_slot_smiles`: best-effort canonicalized SMILES tokens

Use `condition_slot_smiles` when present. Keep `condition_slot` when you want
the original raw signal.

## Local Cache

Downloaded artifacts live in a RetroCast-managed local cache, so callers do not
need to choose download destinations or manually de-duplicate files across
notebooks, scripts, and trainers.

Default location:

```text
~/.cache/retrocast/training-sets
```

Both the shell downloader and the Python dataset API use the same cache layout:

```text
~/.cache/retrocast/training-sets/paroutes/<release>/<artifact>/<file>
```

Override it with:

- `RETROCAST_TRAINING_SET_CACHE_DIR` for both shell and Python
- `cache_dir=...` in Python when you want per-call control

If you want the downloaded artifact in an explicit project-owned location rather
than the shared cache:

- shell: `--dir PATH`
- Python: `output_dir=Path(...)`
