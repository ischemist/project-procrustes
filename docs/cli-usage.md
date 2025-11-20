# RetroCast CLI User Guide

RetroCast is a toolkit for standardizing, scoring, and analyzing retrosynthesis predictions. It allows you to compare models fairly by casting disparate output formats into a canonical schema and running rigorous statistical evaluations.

## Installation

RetroCast can be installed as a standalone tool using `uv` (recommended) or via pip.

```bash
# Install as a tool
uv tool install retrocast

# Or install into your current environment
uv add retrocast

# or if you don't use uv (you really should)
pip install retrocast
```

## 1. The Benchmark Pipeline (CLI)

This is the standard workflow for running a full evaluation campaign. It relies on a structured data directory (`data/`) and a configuration file (`retrocast-config.yaml`).

### Prerequisites

1.  **Define your model** in `retrocast-config.yaml`. You must tell RetroCast which adapter to use to parse your raw output.

    ```yaml
    models:
      my-model:
        adapter: aizynth  # or 'retrostar', 'dms', etc.
        raw_results_filename: predictions.json
        sampling:
          strategy: top-k
          k: 10
    ```

2.  Run the desired model on a benchmark, which will create a result in:
    `data/2-raw/<model-name>/<benchmark-name>/<filename>`

### Step A: Ingest
Convert raw model outputs into the canonical RetroCast format. This step handles deduplication, sampling, and standardization.

```bash
# Process a specific model on a specific benchmark
retrocast ingest --model my-model --dataset paroutes-n1

# Optional: Anonymize the output folder names (useful for blind peer review)
retrocast ingest --model my-model --dataset paroutes-n1 --anonymize
```

### Step B: Score
Evaluate the ingested routes against a specific stock file and benchmark definition.

```bash
# Score using the stock defined in the benchmark
retrocast score --model my-model --dataset paroutes-n1

# Override with a specific stock file
retrocast score --model my-model --dataset paroutes-n1 --stock emolecules-2023
```

### Step C: Analyze
Generate statistical reports, plots, and diagnostic HTML files.

```bash
retrocast analyze --model my-model --dataset paroutes-n1
```

**Output:** Results are saved to `data/5-results/<benchmark>/<model>/<stock>/`. You will find:
- `report.md`: Markdown summary of metrics.
- `diagnostics.html`: Interactive plots (Solvability vs Length, Top-K, etc.).
- `statistics.json.gz`: Raw statistical data with confidence intervals.

---

## 2. Ad-Hoc Scoring (CLI)

If you don't want to set up a full directory structure or config file, you can score a single predictions file directly.

**Prerequisites:**
- A benchmark definition file (`.json.gz`).
- A predictions file in RetroCast format (dict of `target_id` -> `list[Route]`).
- A stock file (text file with one SMILES per line).

```bash
retrocast score-file \
  --benchmark data/1-benchmarks/definitions/paroutes-n1.json.gz \
  --routes my_predictions.json.gz \
  --stock data/1-benchmarks/stocks/zinc-stock.txt \
  --output my_scores.json.gz \
  --model-name "My-Experimental-Run"
```

---

## 3. Python API

You can use RetroCast as a library to integrate scoring or standardization into your own scripts.

### Adapting Raw Data
If you have raw outputs from a model (e.g., AiZynthFinder JSON) and want to cast them to RetroCast `Route` objects programmatically:

```python
from retrocast.adapters import adapt_routes
from retrocast.models.chem import TargetInput

# 1. Load your raw data (e.g., from a file)
raw_data = [
    {"target": "CCO", "trees": [...]},
    {"target": "c1ccccc1", "trees": [...]}
]

# 2. Define the targets (ID and SMILES are required for validation)
targets = [
    TargetInput(id="t1", smiles="CCO"),
    TargetInput(id="t2", smiles="c1ccccc1")
]

# 3. Cast to Route objects
# Returns a dict: { "t1": [Route, ...], "t2": [Route, ...] }
routes_dict = adapt_routes(
    raw_data,
    targets=targets,
    adapter_type="aizynth"  # See ADAPTER_MAP in factory.py for options
)
```

### Programmatic Scoring
Run a full evaluation without using the CLI.

```python
from retrocast.api import score_predictions, load_benchmark

# 1. Load Benchmark
benchmark = load_benchmark("paroutes-n1.json.gz")

# 2. Define Stock (or load from file)
stock = {"CCO", "Cl", "Br"} 

# 3. Score
results = score_predictions(
    benchmark=benchmark,
    predictions=routes_dict,  # From previous step
    stock=stock,
    model_name="MyModel"
)

# Access results
print(f"Is route solved: {results.results['t1'].is_solvable}")
```

### Statistical Analysis
Compute metrics with bootstrapped confidence intervals.

```python
from retrocast.api import compute_metric_with_ci
from retrocast.metrics.bootstrap import get_is_solvable

# Extract TargetEvaluation objects
evals = list(results.results.values())

# Compute Solvability with 95% CI
stats = compute_metric_with_ci(
    targets=evals,
    extractor=get_is_solvable,
    metric_name="Solvability",
    n_boot=10000
)

print(f"Solvability: {stats.overall.value:.1%}")
print(f"95% CI: [{stats.overall.ci_lower:.1%}, {stats.overall.ci_upper:.1%}]")
```
