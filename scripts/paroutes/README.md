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
```

Each release folder contains:

- `all.jsonl.gz`
- `train.jsonl.gz`
- `val.jsonl.gz`
- `buyables/all.jsonl.gz`
- `buyables/train.jsonl.gz`
- `buyables/val.jsonl.gz`
- `manifest.json`
