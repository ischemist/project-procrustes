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

enum SummaryKind {
    Bootstrap { reliability: bool },
    Mean,
}

struct MetricValues {
    name: String,
    values: Vec<f64>,
    kind: SummaryKind,
}

pub fn analyze(
    evaluation: &Evaluation,
    ks: &[usize],
    prefix_depths: &[usize],
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> Result<AnalysisReport> {
    let targets: Vec<_> = evaluation.targets.values().collect();
    let metrics = summarize_targets(
        &targets,
        &evaluation.tiers,
        ks,
        prefix_depths,
        &evaluation.metric_label,
        &evaluation.acceptable_match_level,
        n_boot,
        seed,
        workers,
    )?;
    let mut strata: BTreeMap<String, Vec<&TargetResult>> = BTreeMap::new();
    for target in &targets {
        if let Some(route) = target.target.acceptable_routes.first() {
            strata
                .entry(format!("depth {}", route_depth(route)))
                .or_default()
                .push(target);
        } else if let Some(depth) = target
            .effective_constraints
            .iter()
            .find(|constraint| constraint.kind == "retrocast.route_depth")
            .and_then(|constraint| constraint.fields.get("max_depth"))
        {
            let label = depth
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| depth.to_string());
            strata
                .entry(format!("depth {label}"))
                .or_default()
                .push(target);
        }
    }
    let mut by_stratum = BTreeMap::new();
    for (label, stratum_targets) in strata {
        by_stratum.insert(
            label,
            summarize_targets(
                &stratum_targets,
                &evaluation.tiers,
                ks,
                prefix_depths,
                &evaluation.metric_label,
                &evaluation.acceptable_match_level,
                n_boot,
                seed,
                workers,
            )?,
        );
    }
    Ok(AnalysisReport {
        schema_version: Default::default(),
        metrics,
        by_stratum,
        bootstrap_resamples: n_boot,
        runtime: runtime_summary(&targets),
    })
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
    let target_refs = targets.iter().collect::<Vec<_>>();
    summarize_targets(
        &target_refs,
        tiers,
        ks,
        prefix_depths,
        metric_label,
        match_level,
        n_boot,
        seed,
        workers,
    )
}

#[allow(clippy::too_many_arguments)]
fn summarize_targets(
    targets: &[&TargetResult],
    tiers: &[u8],
    ks: &[usize],
    prefix_depths: &[usize],
    metric_label: &str,
    match_level: &str,
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> Result<BTreeMap<String, MetricSummary>> {
    let mut metrics = Vec::new();
    for &tier in tiers {
        metrics.push(metric(
            format!("tier_{tier}_validity_rate"),
            targets
                .iter()
                .map(|target| rate(target, |candidate| candidate.satisfies_validity(tier)))
                .collect(),
        ));
        metrics.push(metric(
            format!("tier_{tier}_validity_mrr"),
            targets
                .iter()
                .map(|target| mrr(target, |candidate| candidate.satisfies_validity(tier)))
                .collect(),
        ));
        metrics.push(metric(
            format!("solv_{tier}[{metric_label}]_rate"),
            targets
                .iter()
                .map(|target| rate(target, |candidate| candidate.satisfies_solv(tier)))
                .collect(),
        ));
        metrics.push(metric(
            format!("solv_{tier}[{metric_label}]_mrr"),
            targets
                .iter()
                .map(|target| mrr(target, |candidate| candidate.satisfies_solv(tier)))
                .collect(),
        ));
    }
    let reconstruction_targets: Vec<_> = targets
        .iter()
        .copied()
        .filter(|target| !target.target.acceptable_routes.is_empty())
        .collect();
    if !reconstruction_targets.is_empty() {
        for k in sorted_unique(ks) {
            metrics.push(metric(
                format!("acceptable_reconstruction_top_{k}[{metric_label}]"),
                reconstruction_targets
                    .iter()
                    .map(|target| top_k_reconstruction(target, k))
                    .collect(),
            ));
            metrics.push(metric(
                format!("acceptable_root_reconstruction_top_{k}[{metric_label}]"),
                reconstruction_targets
                    .iter()
                    .map(|target| f64::from(has_root_hit(target, k, match_level)))
                    .collect(),
            ));
            metrics.push(metric(
                format!("acceptable_reconstruction_given_root_top_{k}[{metric_label}]"),
                reconstruction_targets
                    .iter()
                    .filter(|target| has_root_hit(target, k, match_level))
                    .map(|target| top_k_reconstruction(target, k))
                    .collect(),
            ));
            metrics.push(MetricValues {
                name: format!("distinct_root_reactions_top_{k}[{metric_label}]"),
                values: reconstruction_targets
                    .iter()
                    .map(|target| distinct_root_reactions(target, k, match_level) as f64)
                    .collect(),
                kind: SummaryKind::Mean,
            });
            for depth in sorted_unique(prefix_depths) {
                metrics.push(metric(
                    format!(
                        "acceptable_prefix_reconstruction_depth_{depth}_top_{k}[{metric_label}]"
                    ),
                    reconstruction_targets
                        .iter()
                        .map(|target| f64::from(has_prefix_hit(target, k, depth, match_level)))
                        .collect(),
                ));
            }
        }
    }
    let summaries = with_pool(workers, || {
        metrics
            .into_par_iter()
            .map(|metric| {
                let summary = match metric.kind {
                    SummaryKind::Bootstrap { reliability } => {
                        stats::summarize_values(&metric.values, n_boot, seed, 0.05, reliability)
                    }
                    SummaryKind::Mean => mean_summary(&metric.values),
                };
                (metric.name, summary)
            })
            .collect::<Vec<_>>()
    })?;
    Ok(summaries.into_iter().collect())
}

fn metric(name: String, values: Vec<f64>) -> MetricValues {
    MetricValues {
        name,
        values,
        kind: SummaryKind::Bootstrap { reliability: true },
    }
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

fn runtime_summary(targets: &[&TargetResult]) -> RuntimeSummary {
    let wall: Vec<_> = targets
        .iter()
        .filter_map(|target| target.wall_time)
        .collect();
    let cpu: Vec<_> = targets
        .iter()
        .filter_map(|target| target.cpu_time)
        .collect();
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
