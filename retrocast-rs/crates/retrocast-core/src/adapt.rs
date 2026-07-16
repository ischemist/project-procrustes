use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
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
    if workers == 0 {
        return Err(EngineError::InvalidWorkers(workers));
    }
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

    let relevant_source_keys = task
        .targets
        .iter()
        .flat_map(|(target_id, target)| [target_id.clone(), target.smiles.to_string()])
        .collect();
    let present_source_keys = read_present_source_keys(path, &relevant_source_keys)?;
    let targets_by_source_key = select_source_keys(task, &present_source_keys);

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
        .and_then(|()| deserializer.end())
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

/// Discover relevant top-level keys without decoding each provider payload.
///
/// `ingest_file` uses this set to resolve ID-over-SMILES precedence before it
/// dispatches work. `StreamingTargetMap` then performs full JSON validation
/// while adapting only the selected payload for each target.
fn read_present_source_keys(path: &Path, relevant: &BTreeSet<String>) -> Result<BTreeSet<String>> {
    let mut reader = io::open_reader(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut present = BTreeSet::new();
    let mut key = Vec::new();
    let mut started = false;
    let mut finished = false;
    let mut depth = 0_usize;
    let mut expecting_key = true;
    let mut in_string = false;
    let mut escaped = false;
    let mut capturing_key = false;

    while !finished {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        for &byte in &buffer[..count] {
            if !started {
                if byte.is_ascii_whitespace() {
                    continue;
                }
                if byte != b'{' {
                    return Err(EngineError::AdapterSchema(
                        "multi-target ingest requires raw payload keyed by target id or target SMILES"
                            .to_owned(),
                    ));
                }
                started = true;
                depth = 1;
                continue;
            }

            if in_string {
                if capturing_key {
                    key.push(byte);
                }
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                    if capturing_key {
                        let source_key: String = serde_json::from_slice(&key)?;
                        if relevant.contains(&source_key) {
                            present.insert(source_key);
                        }
                        capturing_key = false;
                        expecting_key = false;
                    }
                }
                continue;
            }

            match byte {
                b'"' => {
                    in_string = true;
                    capturing_key = depth == 1 && expecting_key;
                    if capturing_key {
                        key.clear();
                        key.push(byte);
                    }
                }
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth = depth.checked_sub(1).ok_or_else(|| {
                        EngineError::AdapterSchema("unbalanced planner JSON".to_owned())
                    })?;
                    if depth == 0 {
                        finished = true;
                        break;
                    }
                }
                b',' if depth == 1 => expecting_key = true,
                byte if depth == 1 && expecting_key && byte.is_ascii_whitespace() => {}
                _ => {}
            }
        }
    }

    if !started || !finished || in_string {
        return Err(EngineError::AdapterSchema(
            "incomplete multi-target planner JSON".to_owned(),
        ));
    }
    Ok(present)
}

fn select_source_keys(
    task: &Task,
    present: &BTreeSet<String>,
) -> BTreeMap<String, (String, Target)> {
    let mut selected = BTreeMap::new();
    let mut matched_targets = BTreeSet::new();

    // Reserve exact target IDs globally before considering SMILES aliases. This
    // prevents one target's alias from consuming another target's explicit key.
    for (target_id, target) in &task.targets {
        if present.contains(target_id) {
            selected.insert(target_id.clone(), (target_id.clone(), target.clone()));
            matched_targets.insert(target_id.clone());
        }
    }
    for (target_id, target) in &task.targets {
        if matched_targets.contains(target_id) {
            continue;
        }
        let source_key = target.smiles.to_string();
        if present.contains(&source_key) && !selected.contains_key(&source_key) {
            selected.insert(source_key, (target_id.clone(), target.clone()));
            matched_targets.insert(target_id.clone());
        }
    }
    selected
}

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
    if workers == 0 {
        return Err(EngineError::InvalidWorkers(workers));
    }
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

    let present_source_keys = raw_targets.keys().cloned().collect();
    let source_key_by_target = select_source_keys(task, &present_source_keys)
        .into_iter()
        .map(|(source_key, (target_id, _))| (target_id, source_key))
        .collect::<BTreeMap<_, _>>();

    Ok(task
        .targets
        .iter()
        .map(|(target_id, target)| {
            let raw = source_key_by_target.get(target_id).and_then(|source_key| {
                raw_targets
                    .remove(source_key)
                    .map(|payload| (payload, Some(source_key.clone())))
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
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::{Value, json};

    use super::{ingest, ingest_file};
    use crate::{
        adapters::AiZynthFinderAdapter, error::EngineError, io, model::Task, route::AdaptMode,
    };

    static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

    fn streamed_task() -> Task {
        serde_json::from_value(json!({
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
        .unwrap()
    }

    fn temporary_path(extension: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "retrocast-streamed-ingest-{}-{nonce}-{}.{extension}",
            std::process::id(),
            NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_temporary_json(raw: &Value) -> PathBuf {
        let path = temporary_path("json.gz");
        io::write_json(&path, raw).unwrap();
        path
    }

    #[test]
    fn streamed_file_ingest_matches_owned_payload_ingest() {
        let task = streamed_task();
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
        let path = write_temporary_json(&raw);

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

    #[test]
    fn streamed_file_ingest_prefers_target_id_over_smiles_alias() {
        let task = streamed_task();
        let raw = json!({
            "CCO": [{"type": "mol", "smiles": "not-smiles"}],
            "ethanol": [{"type": "mol", "smiles": "OCC"}]
        });
        let path = write_temporary_json(&raw);

        let predictions = ingest_file(
            &path,
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            2,
        )
        .unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(predictions["ethanol"].len(), 1);
        assert!(predictions["methanol"].is_empty());
    }

    #[test]
    fn target_id_cannot_be_consumed_as_another_targets_smiles_alias() {
        let task: Task = serde_json::from_value(json!({
            "name": "source-key-collision-test",
            "targets": {
                "0-ethanol": {
                    "id": "0-ethanol",
                    "smiles": "CCO",
                    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
                },
                "CCO": {
                    "id": "CCO",
                    "smiles": "CCC",
                    "inchikey": "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"
                }
            }
        }))
        .unwrap();
        let raw = json!({"CCO": [{"type": "mol", "smiles": "CCC"}]});
        let expected = ingest(
            raw.clone(),
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            2,
        )
        .unwrap();
        let path = write_temporary_json(&raw);

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

        assert!(actual["0-ethanol"].is_empty());
        assert_eq!(
            actual["CCO"].len(),
            1,
            "{}",
            serde_json::to_string_pretty(&actual).unwrap()
        );
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }

    #[test]
    fn streamed_file_ingest_rejects_trailing_content() {
        let task = streamed_task();
        let path = temporary_path("json");
        std::fs::write(
            &path,
            br#"{"ethanol": [{"type": "mol", "smiles": "OCC"}]} trailing"#,
        )
        .unwrap();

        let error = ingest_file(
            &path,
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            2,
        )
        .unwrap_err();
        std::fs::remove_file(path).unwrap();

        assert!(matches!(error, EngineError::Json(_)));
    }

    #[test]
    fn streamed_file_ingest_rejects_zero_workers_before_single_target_fallback() {
        let mut task = streamed_task();
        task.targets.remove("methanol");

        let error = ingest_file(
            PathBuf::from("not-read-for-invalid-workers.json").as_path(),
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            0,
        )
        .unwrap_err();

        assert!(matches!(error, EngineError::InvalidWorkers(0)));
    }
}
