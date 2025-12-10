---
icon: lucide/ruler-dimension-line
---

# Benchmarks

Evaluating retrosynthesis models on large, uncurated datasets is computationally expensive and statistically noisy. RetroCast provides a suite of **stratified evaluation subsets** derived from the PaRoutes dataset.

!!! question "Which benchmark should I use?"

    - **Chemists & application developers** → Use [Market Series](#1-market-series-mkt) (`mkt-*`) with commercial stock
    - **Algorithm researchers** → Use [Reference Series](#2-reference-series-ref) (`ref-*`) with ground-truth stock
    - **Have to deal with a reviewer 2?** → Use [Legacy Random Sets](#3-legacy-random-sets) (not recommended) 

## Stratification Methodology

We employ stratified sampling to address two specific structural issues with raw patent datasets:

!!! warning "Problems with raw patent datasets"

    1. **Metric Insensitivity due to Imbalance**: 74% of the routes in n5 are length 3-4. General Solvability/Top-K metrics can mask significant performance differences on longer routes (5+ steps) or specific topologies (linear vs. convergent).
    
    2. **The Stock Definition Problem**: Only ~46% of PaRoutes leaf molecules are present in Buyables stock, suggesting many "routes" are arbitrary fragments cut off where the patent description ended.

To address this, we provide two series of benchmarks: one for **practical utility** (Market) and one for **algorithmic comparison** (Reference).

### Terminology

!!! info "Route topology definitions"

    - **Convergent route**: Contains at least one reaction combining ≥2 non-leaf molecules
    - **Linear route**: All other routes (no convergent steps)
    
    Note: Only ~10% of routes in n5 are convergent.

## 1. Market Series (`mkt-`)

**Target Audience:** Chemists and Application Developers  
**Research Question:** What is the best off-the-shelf solution for multistep retrosynthetic planning?

These benchmarks evaluate practical utility for synthetic chemists. There are no restrictions on training data—we simply want to know which model provides the best routes right now.

**Construction methodology:**

1. Filter PaRoutes n5 to retain only routes where all starting materials are commercially available (Buyables catalog)
2. Stratify by route length to measure performance across difficulty spectrum

**Stock to use:** `buyables-stock`

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **mkt-lin-500** | 500 | Linear routes of lengths 2, 3, 4, 5, 6 (100 each) |
| **mkt-cnv-160** | 160 | Convergent routes of depths 2, 3, 4, 5 (40 each) |

!!! tip "Training data decontamination"

    For fairness, remove these benchmark targets from your training set. You can do this manually by converting your training routes to RetroCast schema and filtering by route signatures.

## 2. Reference Series (`ref-`)

**Target Audience:** Algorithm Researchers  
**Research Question:** Is Algorithm A better than Algorithm B?

These subsets use the **original stock definition** from PaRoutes (the leaves of the ground truth route). This isolates search algorithm performance from material availability. If the model fails, it's a failure of the search or one-step model, not the stock.

**Stock to use:** `n5-stock` (except `ref-lng-84` which uses `n1-n5-stock`)

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **ref-lin-600** | 600 | Linear routes of lengths 2–7 (100 each) |
| **ref-cnv-400** | 400 | Convergent routes of lengths 2–5 (100 each) |
| **ref-lng-84** | 84 | All available routes with length 8–10 from n1 and n5 |

## 3. Legacy Random Sets

**Target Audience:** Reviewer #3 :eyes:

We provide random samples of the n5 dataset (100, 200, 500, 1k, 2k targets) for cheaper estimation of performance on the full n5 dataset. 

!!! warning "Not recommended"

    We **strongly recommend** using the stratified sets above instead. Random sampling suffers from the imbalance issues described in [Stratification Methodology](#stratification-methodology).

## Validation and Stability

We validated these subsets using a **seed stability analysis**.

!!! note "Why not just compare to the full dataset?"

    Since the subsets are stratified (forced uniform distribution of difficulty), their aggregate metrics will fundamentally differ from the full, skewed dataset. We cannot validate by comparing means.

Instead, we ensured the subsets are **internally representative**:

1. Reused evaluation results of DirectMultiStep (DMS) Explorer XL model on full source datasets (n1 and n5)
2. Generated 15 candidate subsets for each benchmark configuration using different seeds
3. Calculated Z-score for key metrics (Solvability, Top-1, Top-10) for each seed against group mean
4. Selected the seed that minimizes Z-score (most typical representation)

This ensures that, e.g., `ref-lin-600` is the most **representative** sample of linear routes of length 2–7, minimizing noise from sampling luck.

??? example "Seed stability methodology details"

    For full details on the seed stability analysis methodology, see `scripts/06-check-seed-stability.py` in the repository.
