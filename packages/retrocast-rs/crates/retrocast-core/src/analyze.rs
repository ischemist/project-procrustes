use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use crate::{
    error::Result,
    model::{
        AnalysisReport, Evaluation, MetricSummary, RuntimeSummary, ScoredCandidate, TargetResult,
    },
    route::{root_reaction_signature, route_depth, route_signature},
    stats, with_pool,
};

#[derive(Clone, Copy)]
enum SummaryKind {
    Bootstrap,
    Mean,
}

struct MetricValues {
    name: String,
    values: Vec<f64>,
    kind: SummaryKind,
}

struct PreparedMetric {
    value: Option<f64>,
    kind: SummaryKind,
}

/// Retain the scalar contributions analysis needs after a target's routes are released.
pub(crate) struct PreparedTargetAnalysis {
    metrics: BTreeMap<String, PreparedMetric>,
    stratum: Option<String>,
    wall_time: Option<f64>,
    cpu_time: Option<f64>,
}

pub fn analyze(
    evaluation: &Evaluation,
    ks: &[usize],
    prefix_depths: &[usize],
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> Result<AnalysisReport> {
    let prepared = evaluation
        .targets
        .values()
        .map(|target| {
            prepare_target_analysis(
                target,
                &evaluation.tiers,
                ks,
                prefix_depths,
                &evaluation.metric_label,
                &evaluation.acceptable_match_level,
            )
        })
        .collect::<Vec<_>>();
    analyze_prepared_targets(&prepared, n_boot, seed, workers)
}

pub(crate) fn analyze_prepared_targets(
    targets: &[PreparedTargetAnalysis],
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> Result<AnalysisReport> {
    validate_analysis_options(n_boot)?;
    let target_refs = targets.iter().collect::<Vec<_>>();
    let metrics = summarize_prepared_targets(&target_refs, n_boot, seed, workers)?;
    let mut strata: BTreeMap<&str, Vec<&PreparedTargetAnalysis>> = BTreeMap::new();
    for target in targets {
        if let Some(stratum) = target.stratum.as_deref() {
            strata.entry(stratum).or_default().push(target);
        }
    }
    let mut by_stratum = BTreeMap::new();
    for (label, stratum_targets) in strata {
        by_stratum.insert(
            label.to_owned(),
            summarize_prepared_targets(&stratum_targets, n_boot, seed, workers)?,
        );
    }
    Ok(AnalysisReport {
        schema_version: Default::default(),
        metrics,
        by_stratum,
        bootstrap_resamples: n_boot,
        runtime: prepared_runtime_summary(targets),
    })
}

#[allow(clippy::too_many_arguments)]
/// Derive one target's metric contributions and stratum while its scored routes are available.
pub(crate) fn prepare_target_analysis(
    target: &TargetResult,
    tiers: &[u8],
    ks: &[usize],
    prefix_depths: &[usize],
    metric_label: &str,
    match_level: &str,
) -> PreparedTargetAnalysis {
    let mut metrics = BTreeMap::new();
    for &tier in tiers {
        insert_bootstrap_metric(
            &mut metrics,
            format!("tier_{tier}_validity_rate"),
            rate(target, |candidate| candidate.satisfies_validity(tier)),
        );
        insert_bootstrap_metric(
            &mut metrics,
            format!("tier_{tier}_validity_mrr"),
            mrr(target, |candidate| candidate.satisfies_validity(tier)),
        );
        insert_bootstrap_metric(
            &mut metrics,
            format!("solv_{tier}[{metric_label}]_rate"),
            rate(target, |candidate| candidate.satisfies_solv(tier)),
        );
        insert_bootstrap_metric(
            &mut metrics,
            format!("solv_{tier}[{metric_label}]_mrr"),
            mrr(target, |candidate| candidate.satisfies_solv(tier)),
        );
    }
    if !target.target.acceptable_routes.is_empty() {
        for k in sorted_unique(ks) {
            let reconstruction = top_k_reconstruction(target, k);
            let root_hit = has_root_hit(target, k, match_level);
            insert_bootstrap_metric(
                &mut metrics,
                format!("acceptable_reconstruction_top_{k}[{metric_label}]"),
                reconstruction,
            );
            insert_bootstrap_metric(
                &mut metrics,
                format!("acceptable_root_reconstruction_top_{k}[{metric_label}]"),
                f64::from(root_hit),
            );
            // Preserve the zero-count metric when no target in the slice has
            // an acceptable root.
            metrics.insert(
                format!("acceptable_reconstruction_given_root_top_{k}[{metric_label}]"),
                PreparedMetric {
                    value: root_hit.then_some(reconstruction),
                    kind: SummaryKind::Bootstrap,
                },
            );
            metrics.insert(
                format!("distinct_root_reactions_top_{k}[{metric_label}]"),
                PreparedMetric {
                    value: Some(distinct_root_reactions(target, k, match_level) as f64),
                    kind: SummaryKind::Mean,
                },
            );
            for depth in sorted_unique(prefix_depths) {
                insert_bootstrap_metric(
                    &mut metrics,
                    format!(
                        "acceptable_prefix_reconstruction_depth_{depth}_top_{k}[{metric_label}]"
                    ),
                    f64::from(has_prefix_hit(target, k, depth, match_level)),
                );
            }
        }
    }
    PreparedTargetAnalysis {
        metrics,
        stratum: target_stratum(target),
        wall_time: target.wall_time,
        cpu_time: target.cpu_time,
    }
}

fn insert_bootstrap_metric(
    metrics: &mut BTreeMap<String, PreparedMetric>,
    name: String,
    value: f64,
) {
    metrics.insert(
        name,
        PreparedMetric {
            value: Some(value),
            kind: SummaryKind::Bootstrap,
        },
    );
}

fn target_stratum(target: &TargetResult) -> Option<String> {
    if let Some(route) = target.target.acceptable_routes.first() {
        return Some(format!("depth {}", route_depth(route)));
    }
    target
        .effective_constraints
        .iter()
        .find(|constraint| constraint.kind == "retrocast.route_depth")
        .and_then(|constraint| constraint.fields.get("max_depth"))
        .map(|depth| {
            let label = depth
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| depth.to_string());
            format!("depth {label}")
        })
}

fn summarize_prepared_targets(
    targets: &[&PreparedTargetAnalysis],
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> Result<BTreeMap<String, MetricSummary>> {
    let mut metrics: BTreeMap<String, MetricValues> = BTreeMap::new();
    for target in targets {
        for (name, metric) in &target.metrics {
            let values = &mut metrics
                .entry(name.clone())
                .or_insert_with(|| MetricValues {
                    name: name.clone(),
                    values: Vec::new(),
                    kind: metric.kind,
                })
                .values;
            if let Some(value) = metric.value {
                values.push(value);
            }
        }
    }
    let summaries = with_pool(workers, || {
        metrics
            .into_values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|metric| {
                let summary = match metric.kind {
                    SummaryKind::Bootstrap => {
                        stats::summarize_values(&metric.values, n_boot, seed, 0.05, true)
                    }
                    SummaryKind::Mean => mean_summary(&metric.values),
                };
                (metric.name, summary)
            })
            .collect::<Vec<_>>()
    })?;
    Ok(summaries.into_iter().collect())
}

fn prepared_runtime_summary(targets: &[PreparedTargetAnalysis]) -> RuntimeSummary {
    let wall = targets
        .iter()
        .filter_map(|target| target.wall_time)
        .collect::<Vec<_>>();
    let cpu = targets
        .iter()
        .filter_map(|target| target.cpu_time)
        .collect::<Vec<_>>();
    let total_wall_time = (!wall.is_empty()).then(|| wall.iter().sum());
    let total_cpu_time = (!cpu.is_empty()).then(|| cpu.iter().sum());
    RuntimeSummary {
        total_wall_time,
        mean_wall_time: total_wall_time.map(|value: f64| value / wall.len() as f64),
        total_cpu_time,
        mean_cpu_time: total_cpu_time.map(|value: f64| value / cpu.len() as f64),
        timed_target_count: wall.len().max(cpu.len()),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn summarize_target_results(
    targets: &[TargetResult],
    tiers: &[u8],
    ks: &[usize],
    prefix_depths: &[usize],
    metric_label: &str,
    match_level: &str,
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> Result<BTreeMap<String, MetricSummary>> {
    validate_analysis_options(n_boot)?;
    let prepared = targets
        .iter()
        .map(|target| {
            prepare_target_analysis(target, tiers, ks, prefix_depths, metric_label, match_level)
        })
        .collect::<Vec<_>>();
    summarize_prepared_targets(&prepared.iter().collect::<Vec<_>>(), n_boot, seed, workers)
}

fn validate_analysis_options(n_boot: usize) -> Result<()> {
    if n_boot == 0 {
        return Err(crate::error::EngineError::InvalidBootstrapResamples(n_boot));
    }
    Ok(())
}

fn sorted_unique(values: &[usize]) -> Vec<usize> {
    values
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn rate(target: &TargetResult, predicate: impl Fn(&ScoredCandidate) -> bool) -> f64 {
    f64::from(target.candidates.iter().any(predicate))
}

fn mrr(target: &TargetResult, predicate: impl Fn(&ScoredCandidate) -> bool) -> f64 {
    let mut candidates: Vec<_> = target.candidates.iter().collect();
    candidates.sort_by_key(|candidate| candidate.rank);
    candidates
        .into_iter()
        .find(|candidate| predicate(candidate))
        .map_or(0.0, |candidate| 1.0 / candidate.rank as f64)
}

fn task_satisfying_top_k(target: &TargetResult, k: usize) -> Vec<&ScoredCandidate> {
    let mut candidates: Vec<_> = target.candidates.iter().collect();
    candidates.sort_by_key(|candidate| candidate.rank);
    candidates
        .into_iter()
        .filter(|candidate| candidate.satisfies_task())
        .take(k)
        .collect()
}

fn top_k_reconstruction(target: &TargetResult, k: usize) -> f64 {
    f64::from(
        task_satisfying_top_k(target, k)
            .iter()
            .any(|candidate| candidate.matches_acceptable),
    )
}

fn has_root_hit(target: &TargetResult, k: usize, level: &str) -> bool {
    let acceptable: BTreeSet<_> = target
        .target
        .acceptable_routes
        .iter()
        .filter_map(|route| root_reaction_signature(route, level))
        .collect();
    task_satisfying_top_k(target, k)
        .into_iter()
        .filter_map(|candidate| candidate.route.as_ref())
        .filter_map(|route| root_reaction_signature(route, level))
        .any(|signature| acceptable.contains(&signature))
}

fn has_prefix_hit(target: &TargetResult, k: usize, depth: usize, level: &str) -> bool {
    let acceptable: BTreeSet<_> = target
        .target
        .acceptable_routes
        .iter()
        .map(|route| route_signature(route, level, Some(depth)))
        .collect();
    task_satisfying_top_k(target, k)
        .into_iter()
        .filter_map(|candidate| candidate.route.as_ref())
        .map(|route| route_signature(route, level, Some(depth)))
        .any(|signature| acceptable.contains(&signature))
}

fn distinct_root_reactions(target: &TargetResult, k: usize, level: &str) -> usize {
    task_satisfying_top_k(target, k)
        .into_iter()
        .filter_map(|candidate| candidate.route.as_ref())
        .filter_map(|route| root_reaction_signature(route, level))
        .collect::<BTreeSet<_>>()
        .len()
}

fn mean_summary(values: &[f64]) -> MetricSummary {
    MetricSummary {
        value: if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        },
        count: values.len(),
        ci_low: None,
        ci_high: None,
        reliability: None,
    }
}
