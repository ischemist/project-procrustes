---
icon: lucide/home
---

# RetroCast: A Unified Framework for Multistep Retrosynthesis

[![isChemist Protocol v1.0.0](https://img.shields.io/badge/protocol-isChemist%20v1.0.0-blueviolet)](https://github.com/ischemist/protocol)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
![coverage](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/ischemist/project-procrustes/master/coverage.json&query=$.totals.percent_covered_display&label=coverage&color=brightgreen&suffix=%25)
[![arXiv](https://img.shields.io/badge/arXiv-2512.07079-b31b1b.svg)](https://arxiv.org/abs/2512.07079)

**RetroCast** is a comprehensive toolkit for standardizing, scoring, and analyzing multistep retrosynthesis models. It decouples **prediction** from **evaluation**, enabling rigorous, apples-to-apples comparison of disparate algorithms on a unified playing field.

## The Problem

The field of retrosynthesis evaluation is fragmented:

1. **Incompatible Outputs:** AiZynthFinder outputs bipartite graphs; Retro\* outputs precursor maps; DirectMultiStep outputs recursive dictionaries. Comparing them requires bespoke parsers for every paper.
2. **Ad-Hoc Metrics:** "Solvability" is calculated differently across publications, with varying stock definitions (e.g., made-to-order libraries vs. actual off-the-shelf compounds).
3. **Flawed Benchmarks:** Standard datasets are heavily skewed (74% of PaRoutes routes are length 3-4), masking performance failures on complex targets.

**RetroCast solves this.** It provides a canonical schema, adapters for 10+ models, and a rigorous statistical pipeline to turn retrosynthesis from a qualitative art into a quantitative science.

## Key Features

- **Universal Adapters:** Translation layers for *AiZynthFinder*, *Retro\**, *DirectMultiStep*, *SynPlanner*, *Syntheseus*, *ASKCOS*, *RetroChimera*, *DreamRetro*, *MultiStepTTL*, *SynLlama*, and *PaRoutes*.
- **Canonical Schema:** All routes cast into a strict, recursive `Molecule` / `ReactionStep` Pydantic model.
- **Curated Benchmarks:** **Reference Series** (algorithm comparison) and **Market Series** (practical utility), stratified by route length and topology.
- **Rigorous Statistics:** Built-in bootstrapping (95% CI), pairwise tournaments, and probabilistic ranking.
- **Reproducibility:** Every artifact tracked via cryptographic manifests (`SHA256`).

## Installation

=== "uv (recommended)"

    We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management:

    ```bash
    # Install as a standalone tool
    uv tool install retrocast

    # Or add to your project
    uv add retrocast
    ```

    If you don't have `uv`, install it in one minute:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "pip"

    ```bash
    pip install retrocast
    ```

## Getting Started

New to RetroCast? Start here:

- **[Quick Start](quick-start.md)** - Get from raw model output to a statistical report in 5 minutes
- **[Concepts](concepts.md)** - Understand the architecture and philosophy
- **[CLI Reference](guides/cli.md)** - Full command documentation
- **[Python Library](guides/library.md)** - Integrate RetroCast into your research pipelines

## Benchmarks

RetroCast introduces two benchmark series derived from PaRoutes:

### Reference Series (`ref-`)
*For algorithm developers*

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **ref-lin-600** | 600 | Linear routes stratified by length (100 each for lengths 2–7) |
| **ref-cnv-400** | 400 | Convergent routes stratified by length (100 each for lengths 2–5) |
| **ref-lng-84** | 84 | All available routes of extreme length (8–10 steps) |

### Market Series (`mkt-`)
*For practical utility*

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **mkt-lin-500** | 500 | Linear routes solvable with commercial buyables (stratified) |
| **mkt-cnv-160** | 160 | Convergent routes solvable with commercial buyables (stratified) |

See **[Benchmarks Guide](guides/benchmarks.md)** for details.

## Get Data

=== "Latest (recommended)"

    For the most up-to-date benchmarks and stocks, use `get-data.sh`:

    ```bash
    # Show available targets and their sizes
    curl -fsSL https://files.ischemist.com/retrocast/get-data.sh | bash -s

    # Check version and last update
    curl -fsSL https://files.ischemist.com/retrocast/get-data.sh | bash -s -- -V

    # Download a specific benchmark (includes definition + required stock)
    curl -fsSL https://files.ischemist.com/retrocast/get-data.sh | bash -s -- mkt-cnv-160
    ```

=== "Publication (frozen)"

    The complete `data/` folder as used in the preprint is available at [files.ischemist.com/retrocast/publication-data](https://files.ischemist.com/retrocast/publication-data):

    ```bash
    # Show available folders and their sizes
    curl -fsSL https://files.ischemist.com/retrocast/get-pub-data.sh | bash -s

    # Download all benchmark definitions
    curl -fsSL https://files.ischemist.com/retrocast/get-pub-data.sh | bash -s -- definitions
    ```

Verify integrity against manifests:

```bash
retrocast verify --all
```

## Visualization: SynthArena

RetroCast powers **[SynthArena](https://syntharena.ischemist.com)**, an open-source platform for visualizing and comparing retrosynthetic routes.

- Compare predictions from any two models side-by-side
- Visualize ground truth vs. predicted routes with diff overlays
- Inspect stratified performance metrics interactively

## Citation

If you use RetroCast in your research, please cite: [arXiv:2512.07079](https://arxiv.org/abs/2512.07079)


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

## Vision: Structural AI for Chemistry

We distinguish between two fundamental classes of problems in scientific machine learning: **quantitative** (predicting scalar targets like toxicity or binding affinity) and **structural** (generating complex objects governed by an underlying grammar). Quantitative problems, analogous to early NLP challenges like sentiment analysis, are often constrained by data scarcity. In contrast, the most transformative AI breakthroughs—from large language models to AlphaFold—have occurred in structural domains.

**Mastery of structure is a prerequisite for solving downstream quantitative tasks.** Foundation models trained on the structure of language, for instance, now excel at sentiment analysis with little to no task-specific fine-tuning. In organic chemistry, the paramount structural challenge is retrosynthesis: designing a valid synthetic pathway to a molecule of interest. This capability is the key to unlocking critical quantitative problems like predicting synthetic accessibility, a significant bottleneck in drug discovery. Current accessibility heuristics, however, bypass the core structural challenge, relying on learned patterns that correlate with accessibility without ever generating the pathway itself.

**A model cannot judge the difficulty of a journey it cannot first articulate.**

Achieving structural mastery in retrosynthesis is a long journey—one that requires moving beyond fragmented data formats, inconsistent evaluation methods, and unreliable metrics. Progress demands unified, rigorous infrastructure to standardize outputs, track provenance, and measure improvements with statistical rigor.

**RetroCast is that infrastructure.**

## License

MIT License. See [LICENSE](https://github.com/ischemist/project-procrustes/blob/master/LICENSE) for details.
