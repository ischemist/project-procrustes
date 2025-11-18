# """
# Aggregate statistics computation for benchmark evaluation.

# This module computes dataset-level statistics from per-target evaluation results.
# """

# from collections import defaultdict
# from collections.abc import Callable
# from typing import Any

# import numpy as np
# from pydantic import BaseModel, Field

# from retrocast.models.chem import EvaluationResults, TargetEvaluation
# from retrocast.utils.logging import logger


# class BenchmarkStatistics(BaseModel):
#     """Aggregate statistics for a single model on a dataset with a specific stock."""

#     model_name: str
#     dataset_name: str
#     stock_name: str

#     # Overall metrics
#     solve_rate: float
#     solve_rate_ci: tuple[float, float] | None = Field(
#         default=None, description="95% confidence interval for solve rate (lower, upper)"
#     )
#     topk_accuracy: dict[int, float] = Field(
#         default_factory=dict, description="Top-K accuracy for K in [1,2,3,4,5,10,20,50]"
#     )
#     topk_accuracy_ci: dict[int, tuple[float, float]] = Field(
#         default_factory=dict, description="95% CI for top-K accuracy"
#     )

#     # Stratified by experimental route length
#     solve_rate_by_length: dict[int, float] = Field(
#         default_factory=dict, description="Solve rate stratified by experimental route length"
#     )
#     solve_rate_by_length_ci: dict[int, tuple[float, float]] = Field(
#         default_factory=dict, description="95% CI for solve rate by length"
#     )
#     topk_by_length: dict[int, dict[int, float]] = Field(
#         default_factory=dict,
#         description="Top-K accuracy stratified by experimental route length. {length: {k: accuracy}}",
#     )
#     topk_by_length_ci: dict[int, dict[int, tuple[float, float]]] = Field(
#         default_factory=dict, description="95% CI for top-K by length"
#     )

#     # Sample sizes
#     n_targets: int
#     n_targets_by_length: dict[int, int] = Field(
#         default_factory=dict, description="Number of targets for each experimental route length"
#     )

#     def to_dict(self) -> dict[str, Any]:
#         """Convert to a plain dictionary for JSON serialization."""
#         return self.model_dump(mode="json")


# def compute_solve_rate(results: EvaluationResults, stock_name: str) -> float:
#     """
#     Compute the fraction of targets that have at least one solved route.

#     Args:
#         results: Evaluation results for all targets.
#         stock_name: Name of the building block stock to use.

#     Returns:
#         Solve rate as a float between 0 and 1.
#     """
#     if not results.targets:
#         return 0.0

#     num_solved = sum(1 for target in results.targets.values() if target.solvability.get(stock_name, False))
#     return num_solved / len(results.targets)


# def compute_topk_accuracy(results: EvaluationResults, stock_name: str, k: int) -> float:
#     """
#     Compute top-K accuracy: fraction of targets where experimental route is in top-K solved predictions.

#     Args:
#         results: Evaluation results for all targets.
#         stock_name: Name of the building block stock to use.
#         k: The K in top-K (e.g., 1 for top-1, 5 for top-5).

#     Returns:
#         Top-K accuracy as a float between 0 and 1.
#     """
#     if not results.targets:
#         return 0.0

#     num_in_topk = 0

#     for target in results.targets.values():
#         rank = target.experimental_route_rank.get(stock_name)
#         if rank is not None and rank <= k:
#             num_in_topk += 1

#     return num_in_topk / len(results.targets)


# def compute_solve_rate_by_length(results: EvaluationResults, stock_name: str) -> dict[int, float]:
#     """
#     Compute solve rate stratified by experimental route length.

#     Args:
#         results: Evaluation results for all targets.
#         stock_name: Name of the building block stock to use.

#     Returns:
#         Dictionary mapping route length to solve rate.
#     """
#     targets_by_length: dict[int, list] = defaultdict(list)

#     for target in results.targets.values():
#         if target.experimental_route_length is not None:
#             targets_by_length[target.experimental_route_length].append(target)

#     solve_rates = {}
#     for length, targets in targets_by_length.items():
#         num_solved = sum(1 for t in targets if t.solvability.get(stock_name, False))
#         solve_rates[length] = num_solved / len(targets) if targets else 0.0

#     return solve_rates


# def compute_topk_by_length(results: EvaluationResults, stock_name: str, k: int) -> dict[int, float]:
#     """
#     Compute top-K accuracy stratified by experimental route length.

#     Args:
#         results: Evaluation results for all targets.
#         stock_name: Name of the building block stock to use.
#         k: The K in top-K.

#     Returns:
#         Dictionary mapping route length to top-K accuracy.
#     """
#     targets_by_length: dict[int, list] = defaultdict(list)

#     for target in results.targets.values():
#         if target.experimental_route_length is not None:
#             targets_by_length[target.experimental_route_length].append(target)

#     topk_by_length = {}
#     for length, targets in targets_by_length.items():
#         num_in_topk = 0

#         for target in targets:
#             rank = target.experimental_route_rank.get(stock_name)
#             if rank is not None and rank <= k:
#                 num_in_topk += 1

#         topk_by_length[length] = num_in_topk / len(targets)

#     return topk_by_length


# def compute_targets_by_length(results: EvaluationResults) -> dict[int, int]:
#     """
#     Count number of targets for each experimental route length.

#     Args:
#         results: Evaluation results for all targets.

#     Returns:
#         Dictionary mapping route length to count of targets.
#     """
#     counts: dict[int, int] = defaultdict(int)

#     for target in results.targets.values():
#         if target.experimental_route_length is not None:
#             counts[target.experimental_route_length] += 1

#     return dict(counts)


# def bootstrap_metric(
#     results: EvaluationResults,
#     metric_fn: Callable[[EvaluationResults], float],
#     n_bootstrap: int = 10000,
#     confidence: float = 0.95,
#     random_seed: int = 42,
# ) -> tuple[float, float, float]:
#     """
#     Compute bootstrap confidence interval for a metric.

#     Uses the percentile bootstrap method: resample targets with replacement,
#     compute the metric on each bootstrap sample, and use percentiles of the
#     bootstrap distribution as confidence intervals.

#     Args:
#         results: Evaluation results for all targets.
#         metric_fn: Function that takes EvaluationResults and returns a scalar metric.
#         n_bootstrap: Number of bootstrap samples (default 10000).
#         confidence: Confidence level, e.g., 0.95 for 95% CI.
#         random_seed: Random seed for reproducibility.

#     Returns:
#         Tuple of (point_estimate, ci_lower, ci_upper).
#     """
#     if not results.targets:
#         return 0.0, 0.0, 0.0

#     rng = np.random.default_rng(random_seed)
#     target_ids = list(results.targets.keys())
#     n = len(target_ids)

#     # Point estimate on full data
#     point_estimate = metric_fn(results)

#     # Bootstrap
#     bootstrap_values = []
#     from tqdm import tqdm

#     for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
#         # Resample target IDs with replacement
#         resampled_ids = rng.choice(target_ids, size=n, replace=True)

#         # Create resampled results
#         resampled_targets = {tid: results.targets[tid] for tid in resampled_ids}
#         resampled_results = EvaluationResults(
#             model_name=results.model_name,
#             dataset_name=results.dataset_name,
#             stock_names=results.stock_names,
#             targets=resampled_targets,
#             metadata=results.metadata,
#         )

#         # Compute metric on bootstrap sample
#         bootstrap_values.append(metric_fn(resampled_results))

#     # Compute percentile-based confidence interval
#     alpha = 1 - confidence
#     ci_lower = float(np.percentile(bootstrap_values, 100 * alpha / 2))
#     ci_upper = float(np.percentile(bootstrap_values, 100 * (1 - alpha / 2)))

#     return point_estimate, ci_lower, ci_upper


# def bootstrap_metric_by_length(
#     results: EvaluationResults,
#     metric_fn: Callable[[list[TargetEvaluation]], float],
#     n_bootstrap: int = 10000,
#     confidence: float = 0.95,
#     random_seed: int = 42,
# ) -> dict[int, tuple[float, float, float]]:
#     """
#     Compute bootstrap CI for a metric stratified by route length.

#     Args:
#         results: Evaluation results for all targets.
#         metric_fn: Function that takes a list of TargetEvaluation and returns a scalar.
#         n_bootstrap: Number of bootstrap samples.
#         confidence: Confidence level.
#         random_seed: Random seed for reproducibility.

#     Returns:
#         Dictionary mapping route length to (point_estimate, ci_lower, ci_upper).
#     """
#     # Group targets by length
#     targets_by_length: dict[int, list[str]] = defaultdict(list)
#     for target_id, target in results.targets.items():
#         if target.experimental_route_length is not None:
#             targets_by_length[target.experimental_route_length].append(target_id)

#     # Bootstrap each length separately
#     results_by_length = {}
#     for length, target_ids in targets_by_length.items():
#         rng = np.random.default_rng(random_seed + length)  # Different seed per length
#         n = len(target_ids)

#         # Point estimate
#         targets_list = [results.targets[tid] for tid in target_ids]
#         point_estimate = metric_fn(targets_list)

#         # Bootstrap
#         bootstrap_values = []
#         for _ in range(n_bootstrap):
#             resampled_ids = rng.choice(target_ids, size=n, replace=True)
#             resampled_targets = [results.targets[tid] for tid in resampled_ids]
#             bootstrap_values.append(metric_fn(resampled_targets))

#         # CI
#         alpha = 1 - confidence
#         ci_lower = float(np.percentile(bootstrap_values, 100 * alpha / 2))
#         ci_upper = float(np.percentile(bootstrap_values, 100 * (1 - alpha / 2)))

#         results_by_length[length] = (point_estimate, ci_lower, ci_upper)

#     return results_by_length


# def compute_benchmark_statistics(
#     results: EvaluationResults, stock_name: str, compute_ci: bool = True, n_bootstrap: int = 10000
# ) -> BenchmarkStatistics:
#     """
#     Compute all benchmark statistics for a given stock.

#     Args:
#         results: Evaluation results for all targets.
#         stock_name: Name of the building block stock to use.
#         compute_ci: Whether to compute bootstrap confidence intervals (default True).
#         n_bootstrap: Number of bootstrap samples for CI estimation (default 10000).

#     Returns:
#         BenchmarkStatistics object with all metrics computed.
#     """
#     logger.info(f"Computing statistics for {results.model_name} on {results.dataset_name} with {stock_name}")

#     stats = BenchmarkStatistics(
#         model_name=results.model_name,
#         dataset_name=results.dataset_name,
#         stock_name=stock_name,
#         solve_rate=compute_solve_rate(results, stock_name),
#         n_targets=len(results.targets),
#     )

#     # Compute top-K accuracy for standard K values
#     for k in [1, 2, 3, 4, 5, 10, 20, 50]:
#         stats.topk_accuracy[k] = compute_topk_accuracy(results, stock_name, k)

#     # Compute stratified metrics
#     stats.solve_rate_by_length = compute_solve_rate_by_length(results, stock_name)
#     stats.n_targets_by_length = compute_targets_by_length(results)

#     # Compute top-K by length for each K
#     for k in [1, 2, 3, 4, 5, 10, 20, 50]:
#         topk_by_len = compute_topk_by_length(results, stock_name, k)
#         for length, accuracy in topk_by_len.items():
#             if length not in stats.topk_by_length:
#                 stats.topk_by_length[length] = {}
#             stats.topk_by_length[length][k] = accuracy

#     # Compute bootstrap CIs if requested
#     if compute_ci:
#         logger.info("Computing bootstrap confidence intervals...")

#         # Overall solve rate CI
#         _, ci_lower, ci_upper = bootstrap_metric(
#             results, lambda r: compute_solve_rate(r, stock_name), n_bootstrap=n_bootstrap
#         )
#         stats.solve_rate_ci = (ci_lower, ci_upper)

#         # Top-K accuracy CIs
#         for k in [1, 2, 3, 4, 5, 10, 20, 50]:
#             _, ci_lower, ci_upper = bootstrap_metric(
#                 results, lambda r: compute_topk_accuracy(r, stock_name, k), n_bootstrap=n_bootstrap
#             )
#             stats.topk_accuracy_ci[k] = (ci_lower, ci_upper)

#         # Solve rate by length CIs
#         solve_rate_ci_by_length = bootstrap_metric_by_length(
#             results,
#             lambda targets: sum(1 for t in targets if t.solvability.get(stock_name, False)) / len(targets)
#             if targets
#             else 0.0,
#             n_bootstrap=n_bootstrap,
#         )
#         for length, (_, ci_lower, ci_upper) in solve_rate_ci_by_length.items():
#             stats.solve_rate_by_length_ci[length] = (ci_lower, ci_upper)

#         # Top-K by length CIs
#         for k in [1, 2, 3, 4, 5, 10, 20, 50]:

#             def topk_metric(targets):
#                 num_in_topk = sum(
#                     1
#                     for t in targets
#                     if t.experimental_route_rank.get(stock_name) is not None
#                     and t.experimental_route_rank.get(stock_name) <= k
#                 )
#                 num_with_exp = sum(1 for t in targets if t.experimental_route_rank.get(stock_name) is not None)
#                 return num_in_topk / num_with_exp if num_with_exp > 0 else 0.0

#             topk_ci_by_length = bootstrap_metric_by_length(results, topk_metric, n_bootstrap=n_bootstrap)

#             for length, (_, ci_lower, ci_upper) in topk_ci_by_length.items():
#                 if length not in stats.topk_by_length_ci:
#                     stats.topk_by_length_ci[length] = {}
#                 stats.topk_by_length_ci[length][k] = (ci_lower, ci_upper)

#     logger.info(f"Statistics computed: solve_rate={stats.solve_rate:.3f}, top-1={stats.topk_accuracy.get(1, 0.0):.3f}")

#     return stats


# def format_statistics_as_markdown(stats: BenchmarkStatistics) -> str:
#     """
#     Format benchmark statistics as a markdown table.

#     Args:
#         stats: BenchmarkStatistics object to format.

#     Returns:
#         Markdown-formatted string with tables.
#     """
#     lines = []

#     # Header
#     lines.append(f"# Benchmark Statistics: {stats.model_name} on {stats.dataset_name} ({stats.stock_name})")
#     lines.append("")

#     # Overall performance
#     lines.append("## Overall Performance")
#     lines.append("")

#     # Check if we have CIs
#     has_ci = stats.solve_rate_ci is not None

#     if has_ci:
#         lines.append("| Metric | Value | 95% CI |")
#         lines.append("|--------|-------|--------|")
#         ci_lower, ci_upper = stats.solve_rate_ci
#         lines.append(f"| Solve Rate | {stats.solve_rate:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] |")
#         for k in [1, 2, 3, 4, 5, 10, 20, 50]:
#             if k in stats.topk_accuracy:
#                 ci_lower, ci_upper = stats.topk_accuracy_ci[k]
#                 lines.append(f"| Top-{k} Accuracy | {stats.topk_accuracy[k]:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] |")
#     else:
#         lines.append("| Metric | Value |")
#         lines.append("|--------|-------|")
#         lines.append(f"| Solve Rate | {stats.solve_rate:.3f} |")
#         for k in [1, 2, 3, 4, 5, 10, 20, 50]:
#             if k in stats.topk_accuracy:
#                 lines.append(f"| Top-{k} Accuracy | {stats.topk_accuracy[k]:.3f} |")

#     lines.append(f"| Total Targets | {stats.n_targets} |")
#     lines.append("")

#     # Performance by route length
#     if stats.solve_rate_by_length:
#         lines.append("## Performance by Route Length")
#         lines.append("")

#         # Determine which K values we have data for
#         k_values = [1, 2, 3, 4, 5, 10, 20, 50]
#         available_k = [k for k in k_values if k in stats.topk_accuracy]

#         # Check if we have CIs for stratified metrics
#         has_stratified_ci = len(stats.solve_rate_by_length_ci) > 0

#         if has_stratified_ci:
#             # Build header with CI columns
#             header = "| Length | N | Solve Rate | 95% CI |"
#             for k in available_k[:4]:  # Limit to 4 K values when showing CIs
#                 header += f" Top-{k} | 95% CI |"
#             lines.append(header)

#             # Build separator
#             separator = "|--------|---|------------|--------|"
#             for _ in available_k[:4]:
#                 separator += "--------|--------|"
#             lines.append(separator)

#             # Build rows
#             lengths = sorted(stats.n_targets_by_length.keys())
#             for length in lengths:
#                 n = stats.n_targets_by_length.get(length, 0)
#                 solve_rate = stats.solve_rate_by_length.get(length, 0.0)

#                 # Solve rate with CI
#                 if length in stats.solve_rate_by_length_ci:
#                     ci_lower, ci_upper = stats.solve_rate_by_length_ci[length]
#                     row = f"| {length} | {n} | {solve_rate:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] |"
#                 else:
#                     row = f"| {length} | {n} | {solve_rate:.3f} | - |"

#                 # Top-K with CIs
#                 for k in available_k[:4]:
#                     topk_val = stats.topk_by_length.get(length, {}).get(k, 0.0)
#                     if length in stats.topk_by_length_ci and k in stats.topk_by_length_ci[length]:
#                         ci_lower, ci_upper = stats.topk_by_length_ci[length][k]
#                         row += f" {topk_val:.3f} | [{ci_lower:.3f}, {ci_upper:.3f}] |"
#                     else:
#                         row += f" {topk_val:.3f} | - |"

#                 lines.append(row)
#         else:
#             # Build header without CIs
#             header = "| Length | N | Solve Rate |"
#             for k in available_k[:5]:  # Limit to first 5 K values for readability
#                 header += f" Top-{k} |"
#             lines.append(header)

#             # Build separator
#             separator = "|--------|---|------------|"
#             for _ in available_k[:5]:
#                 separator += "--------|"
#             lines.append(separator)

#             # Build rows
#             lengths = sorted(stats.n_targets_by_length.keys())
#             for length in lengths:
#                 n = stats.n_targets_by_length.get(length, 0)
#                 solve_rate = stats.solve_rate_by_length.get(length, 0.0)
#                 row = f"| {length} | {n} | {solve_rate:.3f} |"

#                 for k in available_k[:5]:
#                     topk_val = stats.topk_by_length.get(length, {}).get(k, 0.0)
#                     row += f" {topk_val:.3f} |"

#                 lines.append(row)

#         lines.append("")

#     return "\n".join(lines)
