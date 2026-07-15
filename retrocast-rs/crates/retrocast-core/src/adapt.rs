use std::collections::BTreeMap;

use rayon::prelude::*;
use serde_json::Value;

use crate::{
    adapters::{Adapter, adapt_candidates_with_workers},
    error::{EngineError, Result},
    model::{Candidate, Predictions, Route, Target, Task},
    route::AdaptMode,
    with_pool,
};

pub fn ingest(
    raw_payload: Value,
    adapter: &dyn Adapter,
    task: &Task,
    mode: AdaptMode,
    max_candidates: Option<usize>,
    workers: usize,
) -> Result<Predictions> {
    let jobs = target_jobs(raw_payload, task)?;
    if jobs.len() == 1 {
        let (target_id, target, raw) = jobs.into_iter().next().expect("one target job");
        let candidates = match raw {
            Some((payload, source_key)) => adapt_candidates_with_workers(
                payload,
                adapter,
                mode,
                Some(&target),
                source_key.as_deref(),
                max_candidates,
                workers,
            )?,
            None => Vec::new(),
        };
        return Ok(BTreeMap::from([(target_id, candidates)]));
    }

    let results = with_pool(workers, || {
        jobs.into_par_iter()
            .map(|(target_id, target, raw)| {
                let candidates = match raw {
                    Some((payload, source_key)) => adapt_candidates_with_workers(
                        payload,
                        adapter,
                        mode,
                        Some(&target),
                        source_key.as_deref(),
                        max_candidates,
                        1,
                    ),
                    None => Ok(Vec::new()),
                }?;
                Ok((target_id, candidates))
            })
            .collect::<Result<Vec<_>>>()
    })??;
    Ok(results.into_iter().collect())
}

type TargetJob = (String, Target, Option<(Value, Option<String>)>);

fn target_jobs(raw_payload: Value, task: &Task) -> Result<Vec<TargetJob>> {
    let Some(raw_targets) = raw_payload.as_object() else {
        if task.targets.len() != 1 {
            return Err(EngineError::AdapterSchema(
                "multi-target ingest requires raw payload keyed by target id or target SMILES"
                    .to_owned(),
            ));
        }
        let (target_id, target) = task.targets.first_key_value().expect("one task target");
        return Ok(vec![(
            target_id.clone(),
            target.clone(),
            Some((raw_payload, None)),
        )]);
    };

    Ok(task
        .targets
        .iter()
        .map(|(target_id, target)| {
            let raw = raw_targets
                .get(target_id)
                .cloned()
                .map(|payload| (payload, Some(target_id.clone())))
                .or_else(|| {
                    raw_targets
                        .get(target.smiles.as_str())
                        .cloned()
                        .map(|payload| (payload, Some(target.smiles.to_string())))
                });
            (target_id.clone(), target.clone(), raw)
        })
        .collect())
}

pub fn predictions_from_json(raw: &str) -> Result<Predictions> {
    Ok(serde_json::from_str(raw)?)
}

pub fn collect_candidates(candidates: Vec<Candidate>, task: &Task) -> Predictions {
    let target_by_inchikey: BTreeMap<_, _> = task
        .targets
        .iter()
        .map(|(target_id, target)| (target.inchikey.clone(), target_id.clone()))
        .collect();
    let mut collected: Predictions = task
        .targets
        .keys()
        .map(|target_id| (target_id.clone(), Vec::new()))
        .collect();
    for candidate in candidates {
        let target_id = candidate
            .route
            .as_ref()
            .and_then(|route| target_by_inchikey.get(&route.target.inchikey))
            .or_else(|| {
                candidate.failure.as_ref().and_then(|failure| {
                    failure
                        .target_id
                        .as_ref()
                        .filter(|target_id| task.targets.contains_key(*target_id))
                        .or_else(|| {
                            failure
                                .target_inchikey
                                .as_ref()
                                .and_then(|inchikey| target_by_inchikey.get(inchikey))
                        })
                })
            });
        if let Some(target_id) = target_id {
            collected
                .get_mut(target_id)
                .expect("resolved target belongs to task")
                .push(candidate);
        }
    }
    collected
}

pub fn collect_routes(routes: Vec<Route>, task: &Task) -> BTreeMap<String, Vec<Route>> {
    let target_by_inchikey: BTreeMap<_, _> = task
        .targets
        .iter()
        .map(|(target_id, target)| (target.inchikey.clone(), target_id.clone()))
        .collect();
    let mut collected: BTreeMap<_, Vec<Route>> = task
        .targets
        .keys()
        .map(|target_id| (target_id.clone(), Vec::new()))
        .collect();
    for route in routes {
        if let Some(target_id) = target_by_inchikey.get(&route.target.inchikey) {
            collected
                .get_mut(target_id)
                .expect("resolved target belongs to task")
                .push(route);
        }
    }
    collected
}

pub fn stocks_from_json(raw: &str) -> Result<BTreeMap<String, std::collections::BTreeSet<String>>> {
    Ok(serde_json::from_str(raw)?)
}
