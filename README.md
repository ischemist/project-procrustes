# RetroCast: A Unified Format for Multistep Retrosynthesis

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![coverage](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/ischemist/project-procrustes/master/coverage.json&query=$.totals.percent_covered_display&label=coverage&color=brightgreen&suffix=%25)

## The Problem

Every multistep retrosynthesis model returns routes in a different format. AiZynthFinder uses bipartite molecule-reaction graphs. Retro* outputs precursor maps. DirectMultiStep produces recursive dictionaries. SynPlanner has its own schema. This fragmentation makes working with routes unnecessarily difficult.

## The Solution

RetroCast provides:

1. **A canonical data model** for retrosynthesis routes ([`schemas.py`](src/retrocast/schemas.py)) - a simple, recursive `Molecule`/`ReactionStep`/`Route` structure that any model output can be cast into.

2. **Tested adapters for every major model** - AiZynthFinder, Retro*, DirectMultiStep, SynPlanner, Syntheseus, ASKCOS, RetroChimera, DreamRetro, MultiStepTTL, SynLlama, PARoutes (14 adapters and counting).

3. **Reproducible infrastructure** - UV-managed dependencies with conflict resolution, locked versions, and deterministic processing with cryptographic hashing.

4. **Curated evaluation sets** - Subsets of the PaRoutes n=1 and n=5 test sets (100, 200, 500, 1k, 2k targets) designed to preserve statistical properties while enabling faster benchmarking.

## Quick Start

### Install

```bash
git clone https://github.com/ischemist/project-procrustes
cd project-procrustes
```

No need to manage virtual environments - UV handles everything.

### Run Any Model in Three Commands

**Example: AiZynthFinder with MCTS**

```bash
# 1. Download model assets (once)
uv run scripts/aizynthfinder/1-download-assets.py data/models/aizynthfinder

# 2. Prepare stock file (once)
uv run --extra aizyn scripts/aizynthfinder/2-prepare-stock.py \
    --files data/models/assets/retrocast-bb-stock-v3-canon.csv \
    --source plain \
    --output data/models/assets/retrocast-bb-stock-v3.hdf5 \
    --target hdf5

# 3. Run predictions
uv run --extra aizyn scripts/aizynthfinder/3-run-aizyn-mcts.py --target-name "uspto-190"
```

**Example: DirectMultiStep**

```bash
# 1. Download model checkpoint
bash scripts/directmultistep/1-download-assets.sh

# 2. Run predictions
uv run --extra dms --extra torch-gpu scripts/directmultistep/2-run-dms.py \
    --model-name "explorer-xl" \
    --use-fp16 \
    --target-name "uspto-190"
```

Each model follows the same pattern: numbered scripts in `scripts/<model-name>/`. UV automatically handles conflicting dependencies (PyTorch versions, NumPy pinning, etc.) via optional dependency groups.

### Convert to Unified Format

Once you have raw model outputs, convert them to the canonical RetroCast format:

```bash
# Process a single model run
uv run scripts/process-predictions.py process --model aizynthfinder-mcts --dataset uspto-190

# List available models
uv run scripts/process-predictions.py list

# Show configuration for a specific model
uv run scripts/process-predictions.py info --model directmultistep
```

This will:
- Validate the raw output using model-specific schemas
- Transform it via the appropriate adapter to `Route` objects
- Deduplicate routes
- Save canonical output with a deterministic hash

### Use as a Python Library

You can also use RetroCast programmatically to adapt individual routes from any supported model:

```python
from retrocast import adapt_single_route, TargetInput

# Define your target
target = TargetInput(id="aspirin", smiles="CC(=O)Oc1ccccc1C(=O)O")

# Your model's raw prediction (e.g., DMS format)
raw_route = {
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "children": [
        {"smiles": "Oc1ccccc1C(=O)O", "children": []},
        {"smiles": "CC(=O)Cl", "children": []}
    ]
}

# Adapt to unified format - works with both route-centric (DMS, AiZynth)
# and target-centric (RetroChimera, ASKCOS) adapter formats
route = adapt_single_route(raw_route, target, adapter_name="dms")

if route:
    print(f"Route depth: {route.depth}")
    print(f"Starting materials: {len(route.leaves)}")
```

See [`docs/api_usage.md`](docs/api_usage.md) for complete API documentation and examples.

## Available Models

Adapters are implemented and tested for:

- **AiZynthFinder** (MCTS, Retro*)
- **Retro*** (original implementation)
- **DirectMultiStep** (Flash, Explorer variants)
- **SynPlanner**
- **Syntheseus** (BFS, Retro-0)
- **ASKCOS**
- **RetroChimera**
- **DreamRetro**
- **MultiStepTTL**
- **SynLlama**
- **PARoutes**

See [`retrocast-config.yaml`](retrocast-config.yaml) for full configuration details.

## Evaluation Sets

We provide curated subsets of the PaRoutes benchmark:

- **uspto-190**: Full USPTO test set (190 targets)
- **paroutes-n1-{100,200,500,1k,2k}**: Stratified subsets of the n=1 test set
- **paroutes-n5-{100,200,500,1k,2k}**: Stratified subsets of the n=5 test set

Each subset is:
- Hashed for reproducibility
- Balanced across route lengths and complexities
- Small enough for rapid iteration (100 targets ~10min vs 10k targets ~10hrs)

Subsets are selected such that top-k accuracy on the subset is within 0.05-1% of the full set, depending on size.

## The Canonical Format

At the core of RetroCast is a clean recursive schema ([`src/retrocast/schemas.py`](src/retrocast/schemas.py)):

```python
class Molecule(BaseModel):
    smiles: SmilesStr
    inchikey: InchiKeyStr
    synthesis_step: ReactionStep | None  # None = leaf (starting material)
    metadata: dict[str, Any]

class ReactionStep(BaseModel):
    reactants: list[Molecule]
    mapped_smiles: ReactionSmilesStr | None
    template: str | None
    reagents: list[SmilesStr] | None
    solvents: list[SmilesStr] | None
    metadata: dict[str, Any]

class Route(BaseModel):
    target: Molecule
    rank: int
    solvability: dict[str, bool]  # per building block set
    metadata: dict[str, Any]
```

Every route from every model gets cast into this structure. No ambiguity, no special cases.

## Architecture

RetroCast is built on three principles:

1. **Adapters are the air gap** - All model-specific logic is isolated in pluggable adapters. The core pipeline never touches raw formats directly.

2. **Contracts, not handshakes** - Pydantic schemas enforce validation at every boundary. Invalid data is rejected immediately.

3. **Deterministic & auditable** - Every output is identified by a cryptographic hash of its inputs. Results are reproducible and traceable.

The pipeline:

```
load raw data → adapter → Route → deduplicate → save + manifest
```

See [`docs/adapters.md`](docs/adapters.md) for details on adding new adapters.

## Citation

If you use RetroCast in your research, please cite:

```bibtex
# ArXiv citation - TODO: add link
```

## License

MIT License - see [LICENSE](LICENSE) for details.
