use std::collections::{BTreeMap, BTreeSet};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::model::{
    Candidate, Evaluation, MetricSummary, Predictions, ReliabilityFlag, TargetResult,
};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CandidateRunStatistics {
    pub total_candidates_seen: usize,
    pub successful_candidates: usize,
    pub failed_candidates: usize,
    pub final_candidates_saved: usize,
    #[serde(default)]
    pub targets_with_at_least_one_candidate: BTreeSet<String>,
    #[serde(default)]
    pub candidates_per_target: BTreeMap<String, usize>,
    #[serde(default)]
    pub failures_by_code: BTreeMap<String, usize>,
    #[serde(default)]
    pub failures_by_target: BTreeMap<String, BTreeMap<String, usize>>,
}

impl CandidateRunStatistics {
    pub fn manifest(&self) -> Value {
        let counts = self
            .candidates_per_target
            .values()
            .copied()
            .collect::<Vec<_>>();
        let average = if counts.is_empty() {
            0.0
        } else {
            counts.iter().sum::<usize>() as f64 / counts.len() as f64
        };
        let median = median_usize(&counts);
        json!({
            "total_candidates_seen": self.total_candidates_seen,
            "successful_candidates": self.successful_candidates,
            "failed_candidates": self.failed_candidates,
            "final_candidates_saved": self.final_candidates_saved,
            "num_targets_with_at_least_one_candidate": self.targets_with_at_least_one_candidate.len(),
            "min_candidates_per_target": counts.iter().min().copied().unwrap_or(0),
            "max_candidates_per_target": counts.iter().max().copied().unwrap_or(0),
            "avg_candidates_per_target": round_two(average),
            "median_candidates_per_target": round_two(median),
            "failures_by_code": self.failures_by_code,
        })
    }
}

pub fn candidate_statistics(candidates: &[Candidate]) -> CandidateRunStatistics {
    let mut statistics = CandidateRunStatistics {
        total_candidates_seen: candidates.len(),
        final_candidates_saved: candidates.len(),
        ..CandidateRunStatistics::default()
    };
    for candidate in candidates {
        if let Some(failure) = &candidate.failure {
            statistics.failed_candidates += 1;
            *statistics
                .failures_by_code
                .entry(failure.code.clone())
                .or_default() += 1;
        } else {
            statistics.successful_candidates += 1;
        }
    }
    statistics
}

pub fn collected_candidate_statistics(predictions: &Predictions) -> CandidateRunStatistics {
    let flattened = predictions.values().flatten().cloned().collect::<Vec<_>>();
    let mut statistics = candidate_statistics(&flattened);
    for (target_id, candidates) in predictions {
        if !candidates.is_empty() {
            statistics
                .targets_with_at_least_one_candidate
                .insert(target_id.clone());
            statistics
                .candidates_per_target
                .insert(target_id.clone(), candidates.len());
        }
        for failure in candidates
            .iter()
            .filter_map(|candidate| candidate.failure.as_ref())
        {
            *statistics
                .failures_by_target
                .entry(target_id.clone())
                .or_default()
                .entry(failure.code.clone())
                .or_default() += 1;
        }
    }
    statistics
}

pub fn evaluation_statistics(evaluation: &Evaluation) -> Value {
    let candidates = evaluation
        .targets
        .values()
        .flat_map(|target| &target.candidates)
        .collect::<Vec<_>>();
    let wall_times = evaluation
        .targets
        .values()
        .filter_map(|target| target.wall_time)
        .collect::<Vec<_>>();
    let cpu_times = evaluation
        .targets
        .values()
        .filter_map(|target| target.cpu_time)
        .collect::<Vec<_>>();
    let mut result = serde_json::Map::from_iter([
        (
            "n_targets".to_owned(),
            Value::from(evaluation.targets.len()),
        ),
        ("n_candidates".to_owned(), Value::from(candidates.len())),
        (
            "n_failed_candidates".to_owned(),
            Value::from(
                candidates
                    .iter()
                    .filter(|candidate| candidate.failure.is_some())
                    .count(),
            ),
        ),
    ]);
    insert_runtime_statistics(&mut result, "wall", &wall_times);
    insert_runtime_statistics(&mut result, "cpu", &cpu_times);
    for tier in &evaluation.tiers {
        let solvable = evaluation
            .targets
            .values()
            .filter(|target| {
                target
                    .candidates
                    .iter()
                    .any(|candidate| candidate.satisfies_solv(*tier))
            })
            .count();
        result.insert(format!("n_solv_{tier}"), Value::from(solvable));
    }
    Value::Object(result)
}

fn insert_runtime_statistics(
    result: &mut serde_json::Map<String, Value>,
    name: &str,
    values: &[f64],
) {
    if values.is_empty() {
        return;
    }
    let total = values.iter().sum::<f64>();
    result.insert(
        format!("total_{name}_time_seconds"),
        json!(round_six(total)),
    );
    result.insert(
        format!("mean_{name}_time_seconds"),
        json!(round_six(total / values.len() as f64)),
    );
}

fn median_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) as f64 / 2.0
    } else {
        values[middle] as f64
    }
}

fn round_two(value: f64) -> f64 {
    (value * 100.0).round_ties_even() / 100.0
}

fn round_six(value: f64) -> f64 {
    (value * 1_000_000.0).round_ties_even() / 1_000_000.0
}

#[derive(Clone, Debug, Serialize)]
pub struct RankResult {
    pub model_name: String,
    pub rank_probs: BTreeMap<usize, f64>,
    pub expected_rank: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairwiseComparison {
    pub metric: String,
    pub model_a: String,
    pub model_b: String,
    pub diff_mean: f64,
    pub diff_ci_low: f64,
    pub diff_ci_high: f64,
    pub is_significant: bool,
    pub count: usize,
}

pub fn bootstrap_distribution(values: &[f64], n_boot: usize, seed: u64) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0; n_boot];
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n_boot)
        .map(|_| {
            (0..values.len())
                .map(|_| values[rng.random_range(0..values.len())])
                .sum::<f64>()
                / values.len() as f64
        })
        .collect()
}

pub fn summarize_values(
    values: &[f64],
    n_boot: usize,
    seed: u64,
    alpha: f64,
    include_reliability: bool,
) -> MetricSummary {
    if values.is_empty() {
        return MetricSummary {
            value: 0.0,
            count: 0,
            ci_low: None,
            ci_high: None,
            reliability: include_reliability.then(|| ReliabilityFlag {
                code: "LOW_N".to_owned(),
                message: "No data.".to_owned(),
            }),
        };
    }
    let value = mean(values);
    let mut distribution = bootstrap_distribution(values, n_boot, seed);
    distribution.sort_by(f64::total_cmp);
    MetricSummary {
        value,
        count: values.len(),
        ci_low: Some(percentile(&distribution, alpha / 2.0)),
        ci_high: Some(percentile(&distribution, 1.0 - alpha / 2.0)),
        reliability: include_reliability.then(|| reliability_flag(values.len(), value)),
    }
}

pub fn probabilistic_ranking(
    model_values: &BTreeMap<String, Vec<f64>>,
    n_boot: usize,
    seed: u64,
) -> Vec<RankResult> {
    if model_values.is_empty() {
        return Vec::new();
    }
    let models: Vec<_> = model_values.keys().cloned().collect();
    let distributions: Vec<_> = models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            bootstrap_distribution(
                &model_values[model],
                n_boot,
                seed.wrapping_add(index as u64),
            )
        })
        .collect();
    let mut rank_counts = vec![vec![0_usize; models.len()]; models.len()];
    for scores in (0..n_boot).map(|sample| {
        distributions
            .iter()
            .map(|distribution| distribution[sample])
            .collect::<Vec<_>>()
    }) {
        let mut order: Vec<_> = (0..models.len()).collect();
        order.sort_by(|left, right| {
            scores[*right]
                .total_cmp(&scores[*left])
                .then_with(|| left.cmp(right))
        });
        for (rank, model_index) in order.into_iter().enumerate() {
            rank_counts[model_index][rank] += 1;
        }
    }
    let mut results: Vec<_> = models
        .into_iter()
        .enumerate()
        .map(|(model_index, model_name)| {
            let rank_probs: BTreeMap<_, _> = rank_counts[model_index]
                .iter()
                .enumerate()
                .map(|(rank, count)| {
                    (
                        rank + 1,
                        if n_boot == 0 {
                            0.0
                        } else {
                            *count as f64 / n_boot as f64
                        },
                    )
                })
                .collect();
            let expected_rank = rank_probs
                .iter()
                .map(|(rank, probability)| *rank as f64 * probability)
                .sum();
            RankResult {
                model_name,
                rank_probs,
                expected_rank,
            }
        })
        .collect();
    results.sort_by(|left, right| {
        left.expected_rank
            .total_cmp(&right.expected_rank)
            .then_with(|| left.model_name.cmp(&right.model_name))
    });
    results
}

pub fn paired_difference(
    values_a: &[f64],
    values_b: &[f64],
    model_a: &str,
    model_b: &str,
    metric: &str,
    n_boot: usize,
    seed: u64,
) -> PairwiseComparison {
    assert_eq!(values_a.len(), values_b.len());
    let differences: Vec<_> = values_a
        .iter()
        .zip(values_b)
        .map(|(left, right)| right - left)
        .collect();
    let diff_mean = mean(&differences);
    let mut distribution = bootstrap_distribution(&differences, n_boot, seed);
    distribution.sort_by(f64::total_cmp);
    let diff_ci_low = percentile(&distribution, 0.025);
    let diff_ci_high = percentile(&distribution, 0.975);
    PairwiseComparison {
        metric: metric.to_owned(),
        model_a: model_a.to_owned(),
        model_b: model_b.to_owned(),
        diff_mean,
        diff_ci_low,
        diff_ci_high,
        is_significant: !(diff_ci_low <= 0.0 && 0.0 <= diff_ci_high),
        count: differences.len(),
    }
}

pub fn target_is_solvable(target: &TargetResult) -> f64 {
    f64::from(
        target
            .candidates
            .iter()
            .any(|candidate| candidate.satisfies_solv(0)),
    )
}

pub fn target_top_k(target: &TargetResult, k: usize) -> f64 {
    let mut candidates: Vec<_> = target.candidates.iter().collect();
    candidates.sort_by_key(|candidate| candidate.rank);
    f64::from(
        candidates
            .into_iter()
            .filter(|candidate| candidate.satisfies_task())
            .take(k)
            .any(|candidate| candidate.matches_acceptable),
    )
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

pub fn reliability_flag(n: usize, probability: f64) -> ReliabilityFlag {
    if n < 30 {
        return ReliabilityFlag {
            code: "LOW_N".to_owned(),
            message: format!("Small sample size (N={n} < 30). CIs may be unstable."),
        };
    }
    if n as f64 * probability < 5.0 || n as f64 * (1.0 - probability) < 5.0 {
        return ReliabilityFlag {
            code: "EXTREME_P".to_owned(),
            message: format!(
                "Extreme value (p={:.1}%) for N={n}. Boundary effects likely.",
                probability * 100.0
            ),
        };
    }
    ReliabilityFlag {
        code: "OK".to_owned(),
        message: "Reliable.".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{paired_difference, probabilistic_ranking, summarize_values};

    #[test]
    fn ranks_stronger_models_first() {
        let values = BTreeMap::from([
            ("strong".to_owned(), vec![1.0; 4]),
            ("weak".to_owned(), vec![0.0, 0.0, 1.0, 0.0]),
        ]);
        let ranking = probabilistic_ranking(&values, 200, 3);
        assert_eq!(ranking[0].model_name, "strong");
    }

    #[test]
    fn summarizes_and_compares_values() {
        let summary = summarize_values(&[0.0, 1.0], 100, 4, 0.05, true);
        assert_eq!(summary.value, 0.5);
        let comparison =
            paired_difference(&[0.0, 0.0], &[1.0, 1.0], "left", "right", "toy", 100, 42);
        assert_eq!(comparison.diff_mean, 1.0);
        assert!(comparison.is_significant);
    }
}
