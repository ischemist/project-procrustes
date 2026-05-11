# paroutes curation scripts

## benchmark-prep

Scripts that build benchmark definitions from raw PaRoutes assets.

```bash
uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py --prune-intermediates
uv run scripts/paroutes/benchmark-prep/01-cast-paroutes.py --check-buyables --prune-intermediates
uv run scripts/paroutes/benchmark-prep/02-create-subsets.py
```

## training-set-prep

Scripts that build public training-set release files.

```bash
uv run scripts/paroutes/training-set-prep/01-create-training-release.py
uv run scripts/paroutes/training-set-prep/01-create-training-release.py --mode route
uv run scripts/paroutes/training-set-prep/01-create-training-release.py --mode reaction
uv run scripts/paroutes/training-set-prep/02-create-single-step-release.py
```

`01-create-training-release.py` writes route releases for `route-heldout-n1-n5` and/or
`reaction-heldout-n1-n5`.

Each route release folder contains:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `manifest.json`

`02-create-single-step-release.py` writes the flat reaction release by loading the
released `reaction-heldout-n1-n5` route artifact, preserving the route
`training`/`validation` split, deduplicating within each split, and reporting
cross-split reaction overlap.

The single-step release folder contains:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `all.rsmi.txt.gz`
- `training.rsmi.txt.gz`
- `validation.rsmi.txt.gz`
- `manifest.json`
