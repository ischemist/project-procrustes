# RetroCast: A Unified Framework for Multistep Retrosynthesis

**RetroCast** is a comprehensive toolkit for standardizing, scoring, and analyzing multistep retrosynthesis models. It decouples **prediction** from **evaluation**, enabling rigorous, apples-to-apples comparison of disparate algorithms on a unified playing field.

## The Problem

The field of retrosynthesis evaluation is fragmented:

1. **Incompatible Outputs:** AiZynthFinder outputs bipartite graphs; Retro\* outputs precursor maps; DirectMultiStep outputs recursive dictionaries. Comparing them requires bespoke parsers for every paper.
2. **Ad-Hoc Metrics:** "Solvability" is calculated differently across publications, with varying stock definitions (e.g., eMolecules vs. actual buyable catalogs).
3. **Flawed Benchmarks:** Standard datasets are heavily skewed (74% of PaRoutes routes are length 3-4), masking performance failures on complex targets.

**RetroCast solves this.** It provides a canonical schema, adapters for 10+ models, and a rigorous statistical pipeline to turn retrosynthesis from a qualitative art into a quantitative science.

## Key Features

- **Universal Adapters:** Translation layers for **AiZynthFinder**, **Retro\***, **DirectMultiStep**, **SynPlanner**, **Syntheseus**, **ASKCOS**, **RetroChimera**, **DreamRetro**, **MultiStepTTL**, **SynLlama**, and **PaRoutes**.
- **Canonical Schema:** All routes cast into a strict, recursive `Molecule` / `ReactionStep` Pydantic model.
- **Curated Benchmarks:** **Reference Series** (algorithm comparison) and **Market Series** (practical utility), stratified by route length and topology.
- **Rigorous Statistics:** Built-in bootstrapping (95% CI), pairwise tournaments, and probabilistic ranking.
- **Reproducibility:** Every artifact tracked via cryptographic manifests (`SHA256`).

## Installation

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

## Get Publication Data

The complete `data/` folder with benchmarks, stocks, raw predictions, processed routes, scores, and results is available at [files.ischemist.com/retrocast/publication-data](https://files.ischemist.com/retrocast/publication-data).

Download all benchmark definitions:

```bash
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

**Paper:** [arXiv:2512.07079](https://arxiv.org/abs/2512.07079)

## Vision: Structural AI for Chemistry

Applications of ML to chemistry have mostly centered on **quantitative problems** (predicting toxicity, pKd, yield)—tasks constrained by the scarcity of labeled data.

However, the most transformative AI breakthroughs (LLMs, AlphaFold) have occurred in **structural problems**: tasks requiring complex, structured outputs rather than regression scalars.

Retrosynthesis is the premier structural problem of organic chemistry. Solving it requires moving beyond fragmented data formats and inconsistent evaluation methods. We need unified, rigorous infrastructure to standardize, track, and measure progress.

**RetroCast is that infrastructure.**

## License

MIT License. See [LICENSE](https://github.com/ischemist/project-procrustes/blob/master/LICENSE) for details.
