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
uv run scripts/paroutes/training-set-prep/03-audit-release.py
```

`01-create-training-release.py` writes route releases for `route-holdout-n1-n5` and/or
`reaction-holdout-n1-n5`.

Each route release folder contains:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `manifest.json`

`02-create-single-step-release.py` writes the flat reaction release by loading the
released `reaction-holdout-n1-n5` route artifact, flattening and deduplicating
each split separately, then removing validation reactions whose reaction
identity also appears in training.

The single-step release folder contains:

- `all.jsonl.gz`
- `training.jsonl.gz`
- `validation.jsonl.gz`
- `all.rsmi.txt.gz`
- `training.rsmi.txt.gz`
- `validation.rsmi.txt.gz`
- `manifest.json`

`03-audit-release.py` reads the released route artifacts and writes one master
markdown audit for the whole release version:

- `release-audit.md`
