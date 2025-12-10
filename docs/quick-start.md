# Quick Start

This guide gets you from raw model output to a rigorous statistical report in under 5 minutes.

## 1. Install

We recommend installing RetroCast as a standalone tool using `uv`.

```bash
uv tool install retrocast
```

If you don't have `uv`, you can install it in a minute:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Initialize Project

Go to your working directory and create the default configuration and directory structure.

```bash
retrocast init
```

This creates `retrocast-config.yaml`. Open it and register your model. You need to tell RetroCast which **adapter** to use to parse your files.

**Example `retrocast-config.yaml`:**
```yaml
models:
  # The name you will use in CLI commands
  my-new-model:
    # The parser logic (see docs/adapters.md)
    adapter: aizynth
    # The filename RetroCast looks for in data/2-raw/
    raw_results_filename: predictions.json
    sampling: # optional, can be omitted completely to keep all routes
      strategy: top-k
      k: 10
```

## 3. The Workflow (Ingest -> Score -> Analyze)

RetroCast enforces a structured workflow to ensure reproducibility.

### Step A: Place Raw Data
Put your model's raw output file in the `data/2-raw/` directory following this structure:
`data/2-raw/<model-name>/<benchmark-name>/<filename>`

*Example:* `data/2-raw/my-new-model/paroutes-n1/predictions.json`

> **Note:** See [Benchmarks](benchmarks.md) for details on available evaluation sets.

### Step B: Ingest
Convert the raw output into the canonical RetroCast `Route` format. This standardizes the data and removes duplicates.

```bash
retrocast ingest --model my-new-model --dataset mkt-cnv-160
```

### Step C: Score
Evaluate the routes against the benchmark's defined stock.

```bash
retrocast score --model my-new-model --dataset mkt-cnv-160
```

### Step D: Analyze
Generate the final report with bootstrapped confidence intervals and visualization plots.

```bash
retrocast analyze --model my-new-model --dataset mkt-cnv-160 --make-plots
```

**Done.**
Check `data/5-results/paroutes-n1/my-new-model/` for your Markdown report (`report.md`) and interactive HTML plots.

---

## Alternative: The "Just Score This File" Method

If you don't want to set up a project structure and just want to check a single file against a benchmark:

```bash
retrocast score-file \
  --benchmark data/1-benchmarks/definitions/mkt-cnv-160.json.gz \
  --routes my_predictions.json.gz \
  --stock data/1-benchmarks/stocks/buyables-stock.txt \
  --output scores.json.gz \
  --model-name "Quick-Check"
```

## Next Steps

*   **Understanding the Architecture**: Read [Concepts](concepts.md) to understand why we use adapters and manifests.
*   **Python API**: Want to use RetroCast inside your own scripts? See the [Library Guide](library.md).
*   **Custom Models**: Need to support a new output format? Learn how to write an [Adapter](adapters.md).
*   **Reference**: Full command documentation is available in the [CLI Reference](cli.md).
