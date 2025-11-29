# Command Line Interface

The RetroCast CLI provides a unified interface for standardizing, scoring, and analyzing retrosynthesis predictions. It supports two modes of operation:
1.  **Project Mode**: A structured workflow for reproducible benchmarking of multiple models.
2.  **Ad-Hoc Mode**: Direct commands for processing individual files without a configuration setup.

## Installation

```bash
uv tool install retrocast
```

## 1. Ad-Hoc Workflow (Quick Start)

Use these commands to process single files immediately without setting up a project directory.

### Adapt Raw Predictions
Convert raw output from a supported model (e.g., AiZynthFinder, Retro*) into the standardized RetroCast format.

```bash
retrocast adapt \
  --input raw_predictions.json.gz \
  --adapter aizynth \
  --output standardized_routes.json.gz
```

*   `--adapter`: See `retrocast list` for available adapters.
*   `--benchmark`: (Optional) Provide a benchmark definition file to ensure target IDs match exactly.

### Score Predictions
Evaluate standardized routes against a stock file.

```bash
retrocast score-file \
  --benchmark benchmark.json.gz \
  --routes standardized_routes.json.gz \
  --stock stock_smiles.txt \
  --output scores.json.gz \
  --model-name "My-Experiment"
```

### Create Benchmark
Generate a benchmark definition file from a simple list of SMILES strings (TXT or CSV).

```bash
retrocast create-benchmark \
  --input targets.txt \
  --name "custom-benchmark" \
  --stock-name "zinc-stock" \
  --output custom-benchmark.json.gz
```

---

## 2. Project Workflow (Reproducible Benchmarking)

For large-scale evaluations, RetroCast enforces a directory structure (`data/1-benchmarks`, `data/2-raw`, etc.) and uses a configuration file to manage model settings.

### Initialization

To start a new benchmarking project, generate a default configuration file in your current directory:

```bash
retrocast init
```

This creates `retrocast-config.yaml`. Edit this file to register your models and their specific settings (adapter type, sampling strategy, etc.).

**Configuration Example:**
```yaml
models:
  dms-explorer:
    adapter: dms
    raw_results_filename: predictions.json
    sampling:
      strategy: top-k
      k: 50
```

### Step A: Ingest
Transforms raw model outputs from `data/2-raw/` into standardized routes in `data/3-processed/`.

```bash
retrocast ingest --model dms-explorer --dataset paroutes-n1
```

*   `--anonymize`: Hashes the model name in the output directory for blind review.

### Step B: Score
Evaluates processed routes against the stock defined in the benchmark (or an override). Results are saved to `data/4-scored/`.

```bash
retrocast score --model dms-explorer --dataset paroutes-n1
```

### Step C: Analyze
Aggregates scores into statistical reports with confidence intervals. Results are saved to `data/5-results/`.

```bash
retrocast analyze --model dms-explorer --dataset paroutes-n1 --make-plots
```

*   `--make-plots`: Generates interactive HTML visualizations (requires `viz` dependencies).
*   `--top-k`: Customizes the K values in the summary report (default: 1, 3, 5, 10, 20, 50, 100).

---

## 3. Verification & Auditing

RetroCast generates a `manifest.json` for every file it creates. These manifests track the lineage (inputs, parameters, hashes) of the data.

To verify the integrity of your data pipeline:

```bash
retrocast verify --target data/5-results/paroutes-n1/dms-explorer
```

*   **Standard Check**: Verifies that the file on disk matches the SHA256 hash in its manifest.
*   **Deep Check (`--deep`)**: Recursively verifies the entire dependency graph (Analyze -> Score -> Ingest -> Raw), ensuring logical consistency across the pipeline.

## Helper Commands

*   `retrocast list`: Lists all models configured in `retrocast-config.yaml`.
*   `retrocast info --model <name>`: Displays details for a specific model configuration.
