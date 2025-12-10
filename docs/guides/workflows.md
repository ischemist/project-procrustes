---
icon: lucide/chart-column-big
---

# Analysis Workflows

RetroCast includes a suite of analysis scripts designed for rigorous statistical evaluation.

!!! info "Prerequisites"

    - Scripts are located in the `scripts/` directory
    - Execute using `uv run scripts/<script-name>.py`
    - Requires scored predictions in `data/4-scored/`

## Quick Reference

| Script | Purpose | Key Output |
|:-------|:--------|:-----------|
| `02-compare.py` | Multi-model comparison on one benchmark | Interactive HTML plots |
| `03-compare-paired.py` | Statistical significance testing | Confidence intervals for differences |
| `04-rank.py` | Probabilistic ranking with uncertainty | Rank probability heatmap |
| `05-tournament.py` | Round-robin model comparison | Win/loss matrix |
| `06-check-seed-stability.py` | Benchmark subset variance analysis | Forest plot across seeds |
| `07-create-model-profile.py` | Single model across benchmarks | Cross-benchmark performance |

## Directory Structure

These scripts rely on the standard RetroCast data directory structure:

```
data/
├── 1-benchmarks/
│   ├── definitions/     # *.json.gz benchmark definitions
│   └── stocks/          # *.txt stock files (one SMILES per line)
├── 4-scored/            # Scored predictions (output of `retrocast score`)
│   └── <benchmark>/<model>/<stock>/evaluation.json.gz
└── 6-comparisons/       # Output directory for visualizations
```

!!! warning "Required before running workflows"

    You must run `retrocast score` before using these analysis scripts. They require `data/4-scored/` to exist.

## General Model Comparison

**Script:** `02-compare.py`  
**Use case:** Compare multiple models on a single benchmark

Generates comprehensive visual comparison with interactive Plotly HTML files for:

- Overall Solvability
- Stratified Solvability (by route length/depth)
- Top-K accuracy

### Usage

```bash title="scripts/02-compare.py"
uv run scripts/02-compare.py \
    --benchmark stratified-linear-600 \
    --models dms-explorer-xl aizynthfinder-mcts retro-star \
    --stock n5-stock
```

### Output

`data/6-comparisons/stratified-linear-600/`

- `solvability_overall.html` - Overall performance comparison
- `solvability_stratified.html` - Performance by route length
- `topk_accuracy.html` - Top-1, Top-5, Top-10 comparison


## Paired Hypothesis Testing

**Script:** `03-compare-paired.py`  
**Use case:** Statistical significance testing between models

!!! warning "Don't trust overlapping CIs"

    Comparing overlapping confidence intervals is insufficient for determining if one model outperforms another. Use this script instead.

Performs paired difference test using bootstrap resampling. Calculates 95% confidence interval of the difference:

$$\Delta = \text{Challenger} - \text{Baseline}$$

If the 95% CI for $\Delta$ does not contain zero, the difference is **statistically significant**.

### Usage

```bash title="scripts/03-compare-paired.py"
uv run scripts/03-compare-paired.py \
    --benchmark stratified-linear-600 \
    --baseline dms-deep \
    --challengers dms-flash dms-wide \
    --n-boot 10000 # (1)!
```

1. Number of bootstrap resamples (10,000 recommended for publication)

### Output

**Terminal:** Rich-formatted table showing:

- Mean difference ($\bar{\Delta}$)
- 95% confidence interval
- Significance flag (✅) for Solvability and Top-K metrics

**Example:**

```
┌──────────────┬────────────┬──────────────────┬──────────┐
│ Challenger   │ Metric     │ Δ [95% CI]       │ Sig.     │
├──────────────┼────────────┼──────────────────┼──────────┤
│ dms-flash    │ Top-1      │ +2.3% [0.8, 3.9] │ ✅       │
│ dms-wide     │ Top-1      │ -0.5% [-2.1, 1.2]│          │
└──────────────┴────────────┴──────────────────┴──────────┘
```

## Probabilistic Ranking

**Script:** `04-rank.py`  
**Use case:** Quantify uncertainty in model rankings

!!! question "Is Model A really better than Model B?"

    Ranking by single scalar values (e.g., "45.2% vs 45.1%") is misleading due to statistical noise. This script calculates the probability each model is the true winner.

Performs Monte Carlo simulation:

1. Resample the dataset 10,000 times
2. Rank models in each sample
3. Aggregate results to compute rank probabilities

### Usage

```bash title="scripts/04-rank.py"
uv run scripts/04-rank.py \
    --benchmark stratified-linear-600 \
    --models dms-explorer-xl dms-flash dms-wide \
    --metric top-1 # (1)!
```

1. Metric to rank by: `top-1`, `top-5`, `top-10`, or `solvability`

### Output

**Terminal:** Table showing:

- Expected rank (E[rank])
- Probability of achieving 1st place (P(rank=1))

**File:** `data/6-comparisons/stratified-linear-600/ranking_heatmap_top-1.html`

Interactive heatmap visualizing the full rank probability distribution

## Pairwise Tournament

**Script:** `05-tournament.py`  
**Use case:** Round-robin comparison to establish model hierarchy

Runs a round-robin tournament where every model is compared against every other model using the paired difference test. Useful for:

- Identifying non-transitive relationships (A > B, B > C, but C > A)
- Establishing hierarchy in a large field of models

### Usage

```bash title="scripts/05-tournament.py"
uv run scripts/05-tournament.py \
    --benchmark stratified-linear-600 \
    --models dms-explorer-xl dms-flash dms-wide aizynthfinder-mcts \
    --metric top-1
```

### Output

**Terminal:** Matrix table where cell $(i, j)$ shows:

$$\text{Model}_i - \text{Model}_j$$

- :green_circle: Significant wins (green)
- :red_circle: Significant losses (red)

**File:** `data/6-comparisons/stratified-linear-600/pairwise_matrix_top-1.html`

Interactive heatmap of tournament results

## Seed Stability Analysis

**Script:** `06-check-seed-stability.py`  
**Use case:** Validate benchmark subset representativeness

!!! info "Why does seed matter?"

    When creating subsets of benchmarks (e.g., 100-route test set from a larger pool), the random seed can influence difficulty. This script quantifies that variance.

Analyzes model performance across many different random seeds of the same benchmark class. Generates forest plot to visualize variance in measured performance caused by dataset selection.

### Usage

```bash title="scripts/06-check-seed-stability.py"
uv run scripts/06-check-seed-stability.py \
    --model dms-explorer-xl \
    --base-benchmark stratified-linear-600 \
    --stock n5-stock \
    --seeds 42 299792458 19910806 17760704 20251030 # (1)!
```

1. Test multiple random seeds to measure variance

### Output

**Terminal:** Z-scores for each seed

- Identifies which seed produces the most **representative** (closest to mean) difficulty

**File:** `data/7-meta-analysis/seed_stability.html`

Forest plot showing metric distribution across seeds with confidence intervals

## Model Profiling

**Script:** `07-create-model-profile.py`  
**Use case:** Profile a single model across multiple benchmarks

!!! tip "Inverse of script 02"

    - **Script 02:** Many models on one benchmark
    - **Script 07:** One model on many benchmarks

Useful for profiling:

- Sensitivity to route type (linear vs. convergent)
- Performance degradation with route length
- Comparison across different stock definitions

### Usage

```bash title="scripts/07-create-model-profile.py"
uv run scripts/07-create-model-profile.py \
    --model dms-explorer-xl \
    --benchmarks ref-lin-600 ref-cnv-400 mkt-lin-500 # (1)!
```

1. Compare the same model across different benchmark types

### Output

`data/6-comparisons/<model-name>/`

Standard comparison plots where x-axis or legend groups represent different benchmarks rather than different models

---

## Best Practices

!!! success "Recommended workflow"

    1. **Start broad:** Use `02-compare.py` to visualize all models
    2. **Test significance:** Use `03-compare-paired.py` for top performers
    3. **Quantify uncertainty:** Use `04-rank.py` to get probabilistic rankings
    4. **Deep dive:** Use `05-tournament.py` for comprehensive comparisons
    5. **Validate benchmarks:** Use `06-check-seed-stability.py` when creating new subsets
    6. **Profile models:** Use `07-create-model-profile.py` to understand strengths/weaknesses
