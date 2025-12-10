# Benchmarks and Evaluation Sets

Evaluating retrosynthesis models on large, uncurated datasets is computationally expensive and statistically noisy. RetroCast provides a suite of **stratified evaluation subsets** derived from the PaRoutes dataset. 

## Stratification Methodology

We employ stratified sampling to address two specific structural issues with raw patent datasets:

1.  **Metric Insensitivity due to Imbalance**: 74% of the routes in the n5 dataset are length 3 and 4. As such, general Solvability/Top-K on n5 (or any random subset thereof) can mask significant performance differences on longer routes (lengths 5+) or specific topologies (linear vs. convergent routes).
2.  **The Stock Definition Problem**: The "ground truth" stock in PaRoutes consists of the leaf nodes of the extracted routes. Only ~46% of these molecules are present in Buyables stock, which may imply that many "routes" in the dataset are simply arbitrary fragments of a synthesis, cut off where the patent description ended.

To address this, we provide two series of benchmarks: one for practical utility (**Market**) and one for algorithmic comparison (**Reference**).

### Terminology

A **convergent route** is defined as a route that contains at least one reaction that combines at least two non-leaf molecules. All other routes are termed **linear routes**. Only 10% of routes in n5 are convergent.

## 1. Market Series (`mkt-`)
*Target Audience: Chemists and Application Developers*

These benchmarks are designed to answer the question: **What is the best off the shelf solution for multistep retrosynthetic planning?**. As such, there are no restrictions on which datasets the model can be trained on, just as is, here and now, which model provides the best routes to a synthetic chemist?

To construct these subsets, we filtered the PaRoutes n5 dataset to retain only routes where all starting materials are present in a standard commercial catalog (Buyables). We then stratified the targets by route length to ensure the benchmark measures performance across a spectrum of difficulty, rather than just on trivial targets.

**Stock to use**: `buyables-stock`

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **mkt-lin-500** | 500 | Linear routes of lengths 2, 3, 4, 5, 6 (100 each) |
| **mkt-cnv-160** | 160 | Convergent routes of depths 2, 3, 4, 5 (40 each) |

> In the interest of fairness, it is still desired to remove targets/routes from these subsets from your training set. We're going to provide an API method to do this, but you can already do this manually by converting your training routes to the RetroCast schema and filtering by route signatures.

## 2. Reference Series (`ref-`)
*Target Audience: Algorithm Researchers*

These benchmarks answer the question: **Is Algorithm A better than Algorithm B?**

These subsets use the original stock definition from PaRoutes (the leaves of the ground truth route). This isolates the search algorithm's performance from the availability of materials. If the model fails, it is a failure of the search or the one-step model, not the stock.

**Stock to use**: `n5-stock` (except for `ref-lng-84`)

| Benchmark | Targets | Description |
| :--- | :--- | :--- |
| **ref-lin-600** | 600 | Linear routes of lengths 2, 3, 4, 5, 6, 7 (100 each) |
| **ref-cnv-400** | 400 | Convergent routes of lengths 2, 3, 4, 5 (100 each) |
| **ref-lng-84** | 84 | All available routes with length 8, 9, or 10 from both n1 and n5 sets. |

## 3. Legacy Random Sets
*Target Audience: Reviewer #3*

We provide random samples of the n5 dataset (100, 200, 500, 1k, 2k targets) for cheaper estimation of the performance on the full n5 dataset. We strongly recommend using the stratified sets above.

## Validation and Stability

We validated these subsets using a **seed stability analysis**.

Since the subsets are stratified (forced uniform distribution of difficulty), their aggregate metrics (e.g., Top-1 accuracy or Solvability) will fundamentally differ from the full, skewed dataset. Therefore, we cannot validate them by comparing their means to the full dataset mean.

Instead, we ensured the subsets are **internally representative**:
1.  We reused the evaluation results of the DirectMultiStep (DMS) Explorer XL model on the full source datasets (n1 and n5).
2.  We generated 15 candidate subsets for each benchmark configuration using different seeds.
3.  We calculated the Z-score for key metrics (Solvability, Top-1, Top-10) for each seed against the group mean.
4.  The single seed for each set was chosen to minimize the Z-score.

This ensures that, e.g., the `ref-lin-600` benchmark is the most "typical" representation of linear routes of length 2-7, minimizing noise from sampling luck.
