# Command Line Interface

The RetroCast CLI manages the evaluation lifecycle, from ingesting raw model outputs to generating statistical reports. It enforces the directory structure and provenance tracking described in the [Concepts](concepts.md) documentation.

## Configuration

The CLI relies on a configuration file, typically named `retrocast-config.yaml`, to map model names to their specific adapters and settings.

**Example Configuration:**
```yaml
data_dir: data
models:
  dms-explorer:
    adapter: dms
    raw_results_filename: predictions.json
    sampling:
      strategy: top-k
      k: 50
  aizynth-mcts:
    adapter: aizynth
    raw_results_filename: results.json.gz
```

*   **`adapter`**: Matches a key in the internal `ADAPTER_MAP` (e.g., `dms`, `aizynth`, `retrostar`).
*   **`raw_results_filename`**: The expected filename within the `data/2-raw/<model>/<benchmark>/` directory.
*   **`sampling`**: Defines how to select routes if the model produces more than required. Strategies include `top-k`, `random-k`, and `by-length`.

## The Standard Pipeline

The evaluation workflow consists of three sequential stages: `ingest`, `score`, and `analyze`.

### 1. Ingest

The `ingest` command transforms raw model outputs into the canonical `Route` format. It validates the data against the adapter's schema, deduplicates routes based on their topological signature, and applies sampling strategies.

**Usage:**
```bash
retrocast ingest --model <model_name> --dataset <benchmark_name> [options]
```

**Options:**
*   `--model`: The model key defined in `retrocast-config.yaml`.
*   `--dataset`: The name of the benchmark (corresponding to a file in `data/1-benchmarks/definitions`).
*   `--anonymize`: If set, the output directory uses a hashed model name. This supports blind peer review.
*   `--sampling-strategy` / `--k`: Overrides the sampling configuration defined in the YAML file.

**Output:**
Saves a dictionary of routes (`target_id` -> `list[Route]`) to `data/3-processed/<benchmark>/<model>/routes.json.gz`.

### 2. Score

The `score` command evaluates processed routes against a specific building block stock. It determines solvability and checks for ground truth matches. This step is separated from ingestion to allow the same predictions to be scored against multiple stocks.

**Usage:**
```bash
retrocast score --model <model_name> --dataset <benchmark_name> [options]
```

**Options:**
*   `--model`: The model name.
*   `--dataset`: The benchmark name.
*   `--stock`: (Optional) The name of the stock file in `data/1-benchmarks/stocks/`. If omitted, the default stock defined in the benchmark definition is used.

**Output:**
Saves `EvaluationResults` to `data/4-scored/<benchmark>/<model>/<stock>/evaluation.json.gz`.

### 3. Analyze

The `analyze` command aggregates scored results into statistical distributions. It uses bootstrap resampling to calculate confidence intervals for metrics such as Top-K accuracy and solvability.

**Usage:**
```bash
retrocast analyze --model <model_name> --dataset <benchmark_name> [options]
```

**Options:**
*   `--model`: The model name.
*   `--dataset`: The benchmark name.
*   `--stock`: (Optional) The specific stock context to analyze.
*   `--top-k`: A list of K values to include in the summary table (e.g., `1 5 10`).
*   `--make-plots`: Generates interactive HTML visualizations (requires the `viz` dependency group).

**Output:**
Generates the following in `data/5-results/<benchmark>/<model>/<stock>/`:
*   `statistics.json.gz`: Raw statistical data.
*   `report.md`: A Markdown summary suitable for GitHub or documentation.
*   `diagnostics.html`: (If `--make-plots` is used) Interactive Plotly charts.

## Ad-Hoc Workflows

RetroCast supports workflows outside the standard directory structure for quick checks or integration into external pipelines.

### Score File
The `score-file` command processes a single predictions file against a benchmark and stock file, bypassing the `data/` directory requirements.

**Usage:**
```bash
retrocast score-file \
  --benchmark path/to/benchmark.json.gz \
  --routes path/to/predictions.json.gz \
  --stock path/to/stock.txt \
  --output path/to/output_scores.json.gz \
  --model-name "My-Experiment"
```

### Create Benchmark
The `create-benchmark` command generates a valid `BenchmarkSet` file from a simple list of SMILES strings (TXT or CSV).

**Usage:**
```bash
retrocast create-benchmark \
  --input targets.txt \
  --name "custom-benchmark" \
  --stock-name "zinc-stock" \
  --output custom-benchmark.json.gz
```

## Verification

The `verify` command audits the data lineage using the manifest files generated at each step.

**Usage:**
```bash
retrocast verify --target <path_to_manifest_or_dir> [--deep]
```

*   **Standard Verification:** Checks that files on disk match the SHA256 hashes recorded in their manifests.
*   **Deep Verification (`--deep`)**: Recursively loads the entire dependency graph (e.g., Score -> Ingest -> Raw). It validates logical consistency, ensuring that the input hash recorded in a child step matches the output hash of the parent step.

## Helper Commands

*   `retrocast list`: Lists all models currently defined in the configuration file.
*   `retrocast info --model <name>`: Displays the configuration details for a specific model.
