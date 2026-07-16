use std::collections::{BTreeMap, BTreeSet, HashSet};

use rayon::prelude::*;
use serde_json::{Value, json};

use crate::{
    error::{EngineError, Result},
    model::{
        Candidate, CheckResult, Constraint, ConstraintResult, Evaluation, ExecutionStats,
        Predictions, RouteValidity, ScoredCandidate, Target, TargetResult, Task, TierResult,
    },
    route::{leaves, reduce_inchikey, route_depth, route_signature},
    with_pool,
};

pub type Stocks = BTreeMap<String, BTreeSet<String>>;
type RuntimeStocks = BTreeMap<String, HashSet<String>>;

/// Score independent targets against one precomputed runtime stock index.
pub(crate) struct TargetScorer {
    stocks: RuntimeStocks,
    match_level: String,
    acceptable_route_match: String,
}

impl TargetScorer {
    pub(crate) fn new(stocks: &Stocks, match_level: &str, acceptable_route_match: &str) -> Self {
        Self {
            stocks: runtime_stocks(stocks, match_level),
            match_level: match_level.to_owned(),
            acceptable_route_match: acceptable_route_match.to_owned(),
        }
    }

    pub(crate) fn score_owned(
        &self,
        target: Target,
        constraints: Vec<Constraint>,
        candidates: Vec<Candidate>,
        wall_time: Option<f64>,
        cpu_time: Option<f64>,
    ) -> Result<TargetResult> {
        let candidates = score_target_owned(
            candidates,
            &target,
            &constraints,
            &self.stocks,
            &self.match_level,
            &self.acceptable_route_match,
        )?;
        Ok(TargetResult {
            target,
            effective_constraints: constraints,
            candidates,
            wall_time,
            cpu_time,
        })
    }
}

pub fn score(
    predictions: &Predictions,
    task: &Task,
    stocks: &Stocks,
    match_level: &str,
    acceptable_route_match: &str,
    execution_stats: Option<&ExecutionStats>,
    workers: usize,
) -> Result<Evaluation> {
    let runtime_stocks = runtime_stocks(stocks, match_level);
    let results = with_pool(workers, || {
        task.targets
            .par_iter()
            .map(|(target_id, target)| {
                let constraints = task.effective_constraints(target_id);
                let scored = score_target(
                    predictions
                        .get(target_id)
                        .map(Vec::as_slice)
                        .unwrap_or_default(),
                    target,
                    &constraints,
                    &runtime_stocks,
                    match_level,
                    acceptable_route_match,
                )?;
                Ok((
                    target_id.clone(),
                    TargetResult {
                        target: target.clone(),
                        effective_constraints: constraints,
                        candidates: scored,
                        wall_time: execution_stats
                            .and_then(|stats| stats.wall_time.get(target_id).copied()),
                        cpu_time: execution_stats
                            .and_then(|stats| stats.cpu_time.get(target_id).copied()),
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()
    })??;
    Ok(Evaluation {
        task: task.clone(),
        tiers: vec![0],
        metric_label: task.derived_metric_label(),
        acceptable_match_level: match_level.to_owned(),
        acceptable_route_match: acceptable_route_match.to_owned(),
        targets: results.into_iter().collect(),
        schema_version: Default::default(),
    })
}

pub fn score_owned(
    mut predictions: Predictions,
    task: Task,
    stocks: &Stocks,
    match_level: &str,
    acceptable_route_match: &str,
    execution_stats: Option<&ExecutionStats>,
    workers: usize,
) -> Result<Evaluation> {
    let scorer = TargetScorer::new(stocks, match_level, acceptable_route_match);
    let jobs = task
        .targets
        .iter()
        .map(|(target_id, target)| {
            (
                target_id.clone(),
                target.clone(),
                task.effective_constraints(target_id),
                predictions.remove(target_id).unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    let results = with_pool(workers, || {
        jobs.into_par_iter()
            .map(|(target_id, target, constraints, candidates)| {
                let wall_time =
                    execution_stats.and_then(|stats| stats.wall_time.get(&target_id).copied());
                let cpu_time =
                    execution_stats.and_then(|stats| stats.cpu_time.get(&target_id).copied());
                let result =
                    scorer.score_owned(target, constraints, candidates, wall_time, cpu_time)?;
                Ok((target_id, result))
            })
            .collect::<Result<Vec<_>>>()
    })??;
    Ok(Evaluation {
        metric_label: task.derived_metric_label(),
        task,
        tiers: vec![0],
        acceptable_match_level: match_level.to_owned(),
        acceptable_route_match: acceptable_route_match.to_owned(),
        targets: results.into_iter().collect(),
        schema_version: Default::default(),
    })
}

fn score_target(
    candidates: &[Candidate],
    target: &Target,
    constraints: &[Constraint],
    stocks: &RuntimeStocks,
    match_level: &str,
    acceptable_route_match: &str,
) -> Result<Vec<ScoredCandidate>> {
    let identities: Vec<_> = target
        .acceptable_routes
        .iter()
        .enumerate()
        .map(|(index, route)| {
            (
                index,
                route_depth(route),
                route_signature(route, match_level, None),
            )
        })
        .collect();
    candidates
        .iter()
        .map(|candidate| {
            if let Some(failure) = &candidate.failure {
                let check = CheckResult {
                    code: failure.code.clone(),
                    status: "fail".to_owned(),
                    message: failure.message.clone(),
                    details: failure.context.clone(),
                };
                return Ok(ScoredCandidate {
                    rank: candidate.rank,
                    route: None,
                    failure: Some(failure.clone()),
                    validity: RouteValidity {
                        tiers: BTreeMap::from([(
                            0,
                            TierResult {
                                status: "fail".to_owned(),
                                checks: vec![check],
                            },
                        )]),
                        reactions: Vec::new(),
                    },
                    constraints: ConstraintResult {
                        status: "not_evaluated".to_owned(),
                        checks: Vec::new(),
                    },
                    matches_acceptable: false,
                    matched_acceptable_index: None,
                });
            }
            let route = candidate
                .route
                .as_ref()
                .expect("candidate has route or failure");
            let constraint_result =
                check_task_constraints_runtime(route, constraints, stocks, match_level)?;
            let matched = acceptable_match(route, &identities, match_level, acceptable_route_match);
            Ok(ScoredCandidate {
                rank: candidate.rank,
                route: Some(route.clone()),
                failure: None,
                validity: RouteValidity {
                    tiers: BTreeMap::from([(
                        0,
                        TierResult {
                            status: "pass".to_owned(),
                            checks: Vec::new(),
                        },
                    )]),
                    reactions: Vec::new(),
                },
                constraints: constraint_result,
                matches_acceptable: matched.is_some(),
                matched_acceptable_index: matched,
            })
        })
        .collect()
}

fn score_target_owned(
    candidates: Vec<Candidate>,
    target: &Target,
    constraints: &[Constraint],
    stocks: &RuntimeStocks,
    match_level: &str,
    acceptable_route_match: &str,
) -> Result<Vec<ScoredCandidate>> {
    let identities: Vec<_> = target
        .acceptable_routes
        .iter()
        .enumerate()
        .map(|(index, route)| {
            (
                index,
                route_depth(route),
                route_signature(route, match_level, None),
            )
        })
        .collect();
    candidates
        .into_iter()
        .map(|candidate| {
            if let Some(failure) = candidate.failure {
                let check = CheckResult {
                    code: failure.code.clone(),
                    status: "fail".to_owned(),
                    message: failure.message.clone(),
                    details: failure.context.clone(),
                };
                return Ok(ScoredCandidate {
                    rank: candidate.rank,
                    route: None,
                    failure: Some(failure),
                    validity: RouteValidity {
                        tiers: BTreeMap::from([(
                            0,
                            TierResult {
                                status: "fail".to_owned(),
                                checks: vec![check],
                            },
                        )]),
                        reactions: Vec::new(),
                    },
                    constraints: ConstraintResult {
                        status: "not_evaluated".to_owned(),
                        checks: Vec::new(),
                    },
                    matches_acceptable: false,
                    matched_acceptable_index: None,
                });
            }
            let route = candidate.route.expect("candidate has route or failure");
            let constraint_result =
                check_task_constraints_runtime(&route, constraints, stocks, match_level)?;
            let matched =
                acceptable_match(&route, &identities, match_level, acceptable_route_match);
            Ok(ScoredCandidate {
                rank: candidate.rank,
                route: Some(route),
                failure: None,
                validity: RouteValidity {
                    tiers: BTreeMap::from([(
                        0,
                        TierResult {
                            status: "pass".to_owned(),
                            checks: Vec::new(),
                        },
                    )]),
                    reactions: Vec::new(),
                },
                constraints: constraint_result,
                matches_acceptable: matched.is_some(),
                matched_acceptable_index: matched,
            })
        })
        .collect()
}

pub fn check_task_constraints(
    route: &crate::model::Route,
    constraints: &[Constraint],
    stocks: &Stocks,
    match_level: &str,
) -> Result<ConstraintResult> {
    let stocks = runtime_stocks(stocks, match_level);
    check_task_constraints_runtime(route, constraints, &stocks, match_level)
}

fn runtime_stocks(stocks: &Stocks, match_level: &str) -> RuntimeStocks {
    stocks
        .iter()
        .map(|(name, keys)| {
            (
                name.clone(),
                keys.iter()
                    .map(|key| reduce_inchikey(key, match_level))
                    .collect(),
            )
        })
        .collect()
}

fn check_task_constraints_runtime(
    route: &crate::model::Route,
    constraints: &[Constraint],
    stocks: &RuntimeStocks,
    match_level: &str,
) -> Result<ConstraintResult> {
    let mut checks = Vec::new();
    for constraint in constraints {
        match constraint.kind.as_str() {
            "retrocast.stock_termination" => {
                let stock_name = constraint
                    .fields
                    .get("stock")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        EngineError::AdapterSchema("stock constraint is missing stock".to_owned())
                    })?;
                let Some(stock) = stocks.get(stock_name) else {
                    checks.push(check(
                        "constraint.stock_termination.unregistered_stock",
                        json!({"stock": stock_name}),
                    ));
                    continue;
                };
                if stock.is_empty() {
                    checks.push(check(
                        "constraint.stock_termination.empty_stock",
                        json!({"stock": stock_name}),
                    ));
                    continue;
                }
                let missing: BTreeSet<_> = leaves(route)
                    .into_iter()
                    .filter(|leaf| !stock.contains(&reduce_inchikey(&leaf.inchikey, match_level)))
                    .map(|leaf| leaf.inchikey.clone())
                    .collect();
                if !missing.is_empty() {
                    checks.push(check(
                        "constraint.stock_termination.missing_leaf",
                        json!({"missing_leaf_inchikeys": missing}),
                    ));
                }
            }
            "retrocast.required_leaves" => {
                let required = constraint
                    .fields
                    .get("smiles")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        EngineError::AdapterSchema(
                            "required-leaves constraint is malformed".to_owned(),
                        )
                    })?;
                let leaf_keys: BTreeSet<_> = leaves(route)
                    .into_iter()
                    .map(|leaf| reduce_inchikey(&leaf.inchikey, match_level))
                    .collect();
                let mut missing = Vec::new();
                for smiles in required.iter().filter_map(Value::as_str) {
                    let (_, inchikey) = crate::chem::normalize(smiles)?;
                    if !leaf_keys.contains(&reduce_inchikey(&inchikey, match_level)) {
                        missing.push(inchikey);
                    }
                }
                if !missing.is_empty() {
                    checks.push(check(
                        "constraint.required_leaf.missing",
                        json!({"missing_leaf_inchikeys": missing}),
                    ));
                }
            }
            "retrocast.route_depth" => {
                let depth = route_depth(route);
                let maximum = constraint.fields.get("max_depth").ok_or_else(|| {
                    EngineError::AdapterSchema(
                        "route-depth constraint is missing max_depth".to_owned(),
                    )
                })?;
                if let Some(max_depth) = maximum.as_u64() {
                    if depth > max_depth as usize {
                        checks.push(check(
                            "constraint.route_depth.exceeded",
                            json!({"route_depth": depth, "max_depth": max_depth}),
                        ));
                    }
                } else if let Some(label) = maximum.as_str() {
                    let (minimum, maximum): (usize, Option<usize>) = match label {
                        "short" => (1, Some(3)),
                        "medium" => (4, Some(6)),
                        "long" => (7, None),
                        _ => {
                            return Err(EngineError::AdapterSchema(format!(
                                "invalid route-depth label {label:?}"
                            )));
                        }
                    };
                    if depth < minimum || maximum.is_some_and(|value| depth > value) {
                        checks.push(check(
                            "constraint.route_depth.out_of_range",
                            json!({"route_depth": depth, "min_depth": minimum, "max_depth": maximum}),
                        ));
                    }
                }
            }
            other => return Err(EngineError::UnsupportedConstraint(other.to_owned())),
        }
    }
    Ok(ConstraintResult {
        status: if checks.is_empty() { "pass" } else { "fail" }.to_owned(),
        checks,
    })
}

fn check(code: &str, details: Value) -> CheckResult {
    CheckResult {
        code: code.to_owned(),
        status: "fail".to_owned(),
        message: None,
        details: details.as_object().cloned().unwrap_or_default(),
    }
}

fn acceptable_match(
    route: &crate::model::Route,
    identities: &[(usize, usize, String)],
    level: &str,
    mode: &str,
) -> Option<usize> {
    if mode == "exact" {
        let signature = route_signature(route, level, None);
        return identities
            .iter()
            .find(|(_, _, reference)| *reference == signature)
            .map(|value| value.0);
    }
    let depth = route_depth(route);
    let mut signatures_by_depth = BTreeMap::new();
    identities
        .iter()
        .filter(|(_, reference_depth, _)| depth >= *reference_depth)
        .filter(|(_, reference_depth, reference)| {
            signatures_by_depth
                .entry(*reference_depth)
                .or_insert_with(|| route_signature(route, level, Some(*reference_depth)))
                == reference
        })
        .max_by_key(|(_, reference_depth, _)| *reference_depth)
        .map(|value| value.0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::{Stocks, score, score_owned};
    use crate::{adapt::ingest, adapters::AiZynthFinderAdapter, model::Task, route::AdaptMode};

    #[test]
    fn owned_scoring_matches_the_borrowed_api() {
        let task: Task = serde_json::from_value(json!({
            "name": "owned-score-test",
            "targets": {
                "ethanol": {
                    "id": "ethanol",
                    "smiles": "CCO",
                    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
                }
            }
        }))
        .unwrap();
        let raw = json!({
            "ethanol": [{
                "type": "mol",
                "smiles": "OCC",
                "children": [{
                    "type": "reaction",
                    "smiles": "CCO",
                    "children": [
                        {"type": "mol", "smiles": "C"},
                        {"type": "mol", "smiles": "CC"}
                    ]
                }]
            }]
        });
        let predictions = ingest(
            raw,
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            2,
        )
        .unwrap();
        let stocks: Stocks = BTreeMap::new();

        let borrowed = score(&predictions, &task, &stocks, "full", "prefix", None, 2).unwrap();
        let owned = score_owned(predictions, task, &stocks, "full", "prefix", None, 2).unwrap();

        assert_eq!(
            serde_json::to_value(borrowed).unwrap(),
            serde_json::to_value(owned).unwrap()
        );
    }
}
