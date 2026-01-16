# RetroCast: A Unified Format for Multistep Retrosynthesis

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![coverage](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/ischemist/project-procrustes/master/coverage.json&query=$.totals.percent_covered_display&label=coverage&color=brightgreen&suffix=%25)
[![arXiv](https://img.shields.io/badge/arXiv-2512.07079-b31b1b.svg)](https://arxiv.org/abs/2512.07079)

**RetroCast** is a comprehensive toolkit for standardizing, scoring, and analyzing multistep retrosynthesis models. It decouples **prediction** from **evaluation**, allowing rigorous, apples-to-apples comparison of disparate algorithms on a unified playing field.

## The Crisis of Evaluation

The field of retrosynthesis is fragmented. 
1.  **Incompatible Outputs:** AiZynthFinder outputs bipartite graphs; Retro* outputs precursor maps; DirectMultiStep outputs recursive dictionaries. Comparing them requires writing bespoke parsers for every paper.
2.  **Ad-Hoc Metrics:** "Solvability" is often calculated differently across publications, with varying definitions of commercial stock (e.g., using made-to-order libraries vs. actual off-the-shelf compounds).
3.  **Flawed Benchmarks:** The standard PaRoutes n5 dataset is heavily skewed (74% of routes are length 3-4), masking performance failures on complex targets. Furthermore, the standard stock definition for PaRoutes creates synthetic "ground truths" that are often physically unobtainable.

**RetroCast solves this.** It provides a canonical schema, adapters for 10+ models, and a rigorous statistical pipeline to turn retrosynthesis from a qualitative art into a quantitative science.

---

## Key Features

*   **Universal Adapters:** "Air-gapped" translation layers for *AiZynthFinder*, *Retro\**, *DirectMultiStep*, *SynPlanner*, *Syntheseus*, *ASKCOS*, *RetroChimera*, *DreamRetro*, *MultiStepTTL*, *SynLlama*, and *PaRoutes*.
*   **Canonical Schema:** All routes are cast into a strict, recursive `Molecule` / `ReactionStep` Pydantic model.
*   **Curated Benchmarks:** Includes the **Reference Series** (for algorithm comparison) and **Market Series** (for practical utility), stratified by route length and topology to eliminate statistical noise.
*   **Rigorous Statistics:** Built-in bootstrapping (95% CI), pairwise tournaments, and probabilistic ranking. No more "Model A is 0.1% better than Model B" without significance testing.
*   **Reproducibility:** Every artifact is tracked via cryptographic manifests (`SHA256`).

---

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Install as a standalone tool
uv tool install retrocast

# Or add to your project
uv add retrocast
```

## Get Data

**Latest Data (Updated Regularly)**

For the most up-to-date benchmarks and stocks, use `get-data.sh`:

```bash
# Show available targets and their sizes
curl -fsSL https://files.ischemist.com/retrocast/get-data.sh | bash -s

# Check version and last update
curl -fsSL https://files.ischemist.com/retrocast/get-data.sh | bash -s -- -V

# Download a specific benchmark (includes definition + required stock)
curl -fsSL https://files.ischemist.com/retrocast/get-data.sh | bash -s -- mkt-cnv-160
```

**Publication Data (Frozen)**

The complete `data/` folder as used in the preprint is available at [files.ischemist.com/retrocast/publication-data](https://files.ischemist.com/retrocast/publication-data):

```bash
# Show available folders and their sizes
curl -fsSL https://files.ischemist.com/retrocast/get-pub-data.sh | bash -s

# Download all benchmark definitions
curl -fsSL https://files.ischemist.com/retrocast/get-pub-data.sh | bash -s -- definitions
```

you can verify the integrity of downloaded files against the manifests by running

```bash
retrocast verify --all
```

that command might warn you about missing files---that is expected. Manifests for, say 4-scored, contain hashes of input files from 3-results, and if you downloaded only 4-scored, you will get warnings about missing 3-results files.

a dump of the sqlite db with the stocks, routes, and results loaded into SynthArena can be found in [ischemist/syntharena repo](https://github.com/ischemist/syntharena).

---

## Quick Start

### 1. The Ad-Hoc Workflow
Have a raw output file from a model? Score it immediately.

```bash
# Convert raw AiZynthFinder JSON to RetroCast format
retrocast adapt \
    --input raw_predictions.json.gz \
    --adapter aizynth \
    --output routes.json.gz

# Score against a stock file
retrocast score-file \
    --benchmark data/1-benchmarks/definitions/ref-lin-600.json.gz \
    --routes routes.json.gz \
    --stock data/1-benchmarks/stocks/n5-stock.txt \
    --output scores.json.gz \
    --model-name "My-Experimental-Model"
```

### 2. The Project Workflow
For full-scale benchmarking, RetroCast enforces a structured data lifecycle: `Ingest` $\to$ `Score` $\to$ `Analyze`.

**Initialize a project:**
```bash
retrocast init
```

**Configure your model in `retrocast-config.yaml`:**
```yaml
models:
  dms-explorer:
    adapter: dms
    raw_results_filename: predictions.json
    sampling: { strategy: top-k, k: 50 }
```

**Run the pipeline:**
```bash
# 1. Ingest: Standardize raw outputs from data/2-raw/
retrocast ingest --model dms-explorer --dataset ref-lin-600

# 2. Score: Evaluate against the benchmark's defined stock
retrocast score --model dms-explorer --dataset ref-lin-600

# 3. Analyze: Generate bootstrap statistics and HTML plots
retrocast analyze --model dms-explorer --dataset ref-lin-600 --make-plots
```

**Output:** Interactive diagnostic plots (Solvability vs Depth, Top-K) and a Markdown report in `data/5-results/`.

---

## The Benchmarks

RetroCast introduces two new benchmark series derived from PaRoutes, fixing the skew and stock issues of the original dataset. These subsets were selected via **seed stability analysis** to ensure they are statistically representative of the underlying difficulty distribution.

### The Reference Series (`ref-`)
*Target Audience: Algorithm Developers*
Designed to compare search algorithms (e.g., MCTS vs. Retro* vs. Transformers). Uses the internal PaRoutes stock to isolate search failures from stock availability issues.

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **ref-lin-600** | 600 | **Linear** routes stratified by length (100 each for lengths 2–7). |
| **ref-cnv-400** | 400 | **Convergent** routes stratified by length (100 each for lengths 2–5). |
| **ref-lng-84** | 84 | All available routes of extreme length (8–10 steps). |

### The Market Series (`mkt-`)
*Target Audience: Computational Chemists*
Designed to assess practical utility. Targets are filtered to be solvable using **Buyables**, a curated catalog of 300k compounds available for <$100/g.

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **mkt-lin-500** | 500 | Linear routes solvable with commercial buyables (Stratified). |
| **mkt-cnv-160** | 160 | Convergent routes solvable with commercial buyables (Stratified). |

---

## Python API

RetroCast is also a library. You can use it to integrate standardization directly into your training or inference loops.

```python
from retrocast import adapt_single_route, TargetInput

# Define the target
target = TargetInput(id="t1", smiles="CC(=O)Oc1ccccc1C(=O)O")

# Your model's raw output (any supported format)
raw_output = {
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "children": [...]
}

# Cast to the canonical Route object
route = adapt_single_route(raw_output, target, adapter_name="dms")

print(f"Depth: {route.length}")
print(f"Leaves: {[m.smiles for m in route.leaves]}")
```

---

## Visualization: SynthArena

RetroCast powers **[SynthArena](https://syntharena.ischemist.com)**, an open-source web platform for visualizing and comparing retrosynthetic routes.

*   Compare predictions from any two models side-by-side.
*   Visualize ground truth vs. predicted routes with diff overlays.
*   Inspect stratified performance metrics interactively.

---

## Vision: Structural AI for Chemistry

We distinguish between two fundamental classes of problems in scientific machine learning: **quantitative** (predicting scalar targets like toxicity or binding affinity) and **structural** (generating complex objects governed by an underlying grammar). Quantitative problems, analogous to early NLP challenges like sentiment analysis, are often constrained by data scarcity. In contrast, the most transformative AI breakthroughs—from large language models to AlphaFold—have occurred in structural domains.

**Mastery of structure is a prerequisite for solving downstream quantitative tasks.** Foundation models trained on the structure of language, for instance, now excel at sentiment analysis with little to no task-specific fine-tuning. In organic chemistry, the paramount structural challenge is retrosynthesis: designing a valid synthetic pathway to a molecule of interest. This capability is the key to unlocking critical quantitative problems like predicting synthetic accessibility, a significant bottleneck in drug discovery. Current accessibility heuristics, however, bypass the core structural challenge, relying on learned patterns that correlate with accessibility without ever generating the pathway itself.

**A model cannot judge the difficulty of a journey it cannot first articulate.**

Achieving structural mastery in retrosynthesis is a long journey—one that requires moving beyond fragmented data formats, inconsistent evaluation methods, and unreliable metrics. Progress demands unified, rigorous infrastructure to standardize outputs, track provenance, and measure improvements with statistical rigor.

**RetroCast is that infrastructure.**

## Citation

If you use RetroCast in your research, please cite:

```bibtex
@misc{retrocast,
  title         = {Procrustean Bed for AI-Driven Retrosynthesis: A Unified Framework for Reproducible Evaluation},
  author        = {Anton Morgunov and Victor S. Batista},
  year          = {2025},
  eprint        = {2512.07079},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2512.07079}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
