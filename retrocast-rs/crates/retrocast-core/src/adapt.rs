use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex, mpsc::sync_channel},
};

use rayon::prelude::*;
use serde::de::{DeserializeSeed, Error as _, IgnoredAny, MapAccess, Visitor};
use serde_json::Value;

use crate::{
    adapters::{Adapter, adapt_candidates_with_workers},
    error::{EngineError, Result},
    io,
    model::{Candidate, Predictions, Route, Target, Task},
    route::AdaptMode,
    with_pool,
};

pub fn ingest_file(
    path: &Path,
    adapter: &dyn Adapter,
    task: &Task,
    mode: AdaptMode,
    max_candidates: Option<usize>,
    workers: usize,
) -> Result<Predictions> {
    if task.targets.len() <= 1 {
        return ingest(
            io::read_json(path)?,
            adapter,
            task,
            mode,
            max_candidates,
            workers,
        );
    }
    if workers == 0 {
        return Err(EngineError::InvalidWorkers(workers));
    }

    let mut targets_by_source_key = BTreeMap::new();
    for (target_id, target) in &task.targets {
        targets_by_source_key.insert(target_id.clone(), (target_id.clone(), target.clone()));
    }
    for (target_id, target) in &task.targets {
        targets_by_source_key
            .entry(target.smiles.to_string())
            .or_insert_with(|| (target_id.clone(), target.clone()));
    }

    let reader = io::open_reader(path)?;
    let (sender, receiver) = sync_channel::<StreamingTargetJob>(0);
    let receiver = Arc::new(Mutex::new(receiver));
    let worker_count = workers.min(task.targets.len());
    let (parse_result, worker_results) = std::thread::scope(|scope| {
        let handles = (0..worker_count)
            .map(|_| {
                let receiver = Arc::clone(&receiver);
                scope.spawn(move || {
                    let mut results = Vec::new();
                    loop {
                        let job = receiver
                            .lock()
                            .expect("ingest queue is not poisoned")
                            .recv();
                        let Ok((target_id, target, payload, source_key)) = job else {
                            break;
                        };
                        let candidates = adapt_candidates_with_workers(
                            payload,
                            adapter,
                            mode,
                            Some(&target),
                            Some(&source_key),
                            max_candidates,
                            1,
                        );
                        results.push(candidates.map(|candidates| (target_id, candidates)));
                    }
                    results
                })
            })
            .collect::<Vec<_>>();

        let mut deserializer = serde_json::Deserializer::from_reader(reader);
        let parse_result = StreamingTargetMap {
            targets: &targets_by_source_key,
            sender: &sender,
        }
        .deserialize(&mut deserializer)
        .map_err(EngineError::from);
        drop(sender);

        let worker_results = handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("ingest worker did not panic"))
            .collect::<Vec<_>>();
        (parse_result, worker_results)
    });
    parse_result?;

    let mut predictions = task
        .targets
        .keys()
        .map(|target_id| (target_id.clone(), Vec::new()))
        .collect::<Predictions>();
    for result in worker_results {
        let (target_id, candidates) = result?;
        predictions.insert(target_id, candidates);
    }
    Ok(predictions)
}

type StreamingTargetJob = (String, Target, Value, String);

struct StreamingTargetMap<'a> {
    targets: &'a BTreeMap<String, (String, Target)>,
    sender: &'a std::sync::mpsc::SyncSender<StreamingTargetJob>,
}

impl<'de> DeserializeSeed<'de> for StreamingTargetMap<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for StreamingTargetMap<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a planner payload keyed by target id or target SMILES")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(source_key) = map.next_key::<String>()? {
            let Some((target_id, target)) = self.targets.get(&source_key) else {
                map.next_value::<IgnoredAny>()?;
                continue;
            };
            let payload = map.next_value::<Value>()?;
            self.sender
                .send((target_id.clone(), target.clone(), payload, source_key))
                .map_err(A::Error::custom)?;
        }
        Ok(())
    }
}

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
    let Value::Object(mut raw_targets) = raw_payload else {
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
                .remove(target_id)
                .map(|payload| (payload, Some(target_id.clone())))
                .or_else(|| {
                    raw_targets
                        .remove(target.smiles.as_str())
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::{ingest, ingest_file};
    use crate::{adapters::AiZynthFinderAdapter, io, model::Task, route::AdaptMode};

    #[test]
    fn streamed_file_ingest_matches_owned_payload_ingest() {
        let task: Task = serde_json::from_value(json!({
            "name": "streamed-ingest-test",
            "targets": {
                "ethanol": {
                    "id": "ethanol",
                    "smiles": "CCO",
                    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
                },
                "methanol": {
                    "id": "methanol",
                    "smiles": "CO",
                    "inchikey": "OKKJLVBELUTLKV-UHFFFAOYSA-N"
                }
            }
        }))
        .unwrap();
        let raw = json!({
            "ethanol": [{"type": "mol", "smiles": "OCC"}],
            "CO": [{"type": "mol", "smiles": "OC"}],
            "not-in-task": [{"type": "mol", "smiles": "C"}]
        });
        let expected = ingest(
            raw.clone(),
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            2,
        )
        .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "retrocast-streamed-ingest-{}-{nonce}.json.gz",
            std::process::id()
        ));
        io::write_json(&path, &raw).unwrap();

        let actual = ingest_file(
            &path,
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            2,
        )
        .unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(
            serde_json::to_value(expected).unwrap(),
            serde_json::to_value(actual).unwrap()
        );
    }
}
