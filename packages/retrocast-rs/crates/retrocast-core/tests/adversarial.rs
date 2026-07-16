use std::{collections::BTreeMap, fs};

use proptest::{collection, prelude::*};
use retrocast_core::{
    adapt::{ingest, ingest_file},
    adapters::{self, AiZynthFinderAdapter, adapt_candidates_with_workers},
    analyze::{analyze, summarize_target_results},
    chem,
    error::EngineError,
    io::{read_json_value, read_jsonl_values, write_json, write_json_gz, write_jsonl_gz},
    model::{Candidate, Predictions, Target, Task},
    route::AdaptMode,
    score::{Stocks, score},
};
use serde_json::{Map, Value, json};
use tempfile::tempdir;

fn bounded_string() -> impl Strategy<Value = String> {
    collection::vec(any::<char>(), 0..64).prop_map(|characters| characters.into_iter().collect())
}

fn bounded_json() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(|number| Value::Number(number.into())),
        bounded_string().prop_map(Value::String),
    ];

    leaf.prop_recursive(4, 64, 8, |inner| {
        prop_oneof![
            collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
            collection::btree_map(bounded_string(), inner, 0..8).prop_map(|entries| {
                Value::Object(entries.into_iter().collect::<Map<String, Value>>())
            }),
        ]
    })
}

proptest! {
    #[test]
    fn adapter_candidate_boundaries_are_worker_deterministic_and_respect_limits(
        adapter_index in 0_usize..adapters::BUILT_IN_ADAPTERS.len(),
        payload in bounded_json(),
        limit in 0_usize..16,
    ) {
        let name = adapters::BUILT_IN_ADAPTERS[adapter_index];
        let adapter = adapters::built_in(name).expect("registered built-in adapter");
        let serial = adapt_candidates_with_workers(
            payload.clone(),
            adapter.as_ref(),
            AdaptMode::Strict,
            None,
            Some("generated-source"),
            Some(limit),
            1,
        );
        let parallel = adapt_candidates_with_workers(
            payload,
            adapter.as_ref(),
            AdaptMode::Strict,
            None,
            Some("generated-source"),
            Some(limit),
            4,
        );

        match (serial, parallel) {
            (Ok(serial), Ok(parallel)) => {
                prop_assert!(serial.len() <= limit);
                prop_assert_eq!(serde_json::to_value(&serial).unwrap(), serde_json::to_value(&parallel).unwrap());
                for candidate in serial {
                    prop_assert!(candidate.rank > 0);
                    prop_assert!(candidate.route.is_some() ^ candidate.failure.is_some());
                    prop_assert!(serde_json::from_value::<Candidate>(serde_json::to_value(candidate).unwrap()).is_ok());
                }
            }
            (Err(serial), Err(parallel)) => prop_assert_eq!(serial.to_string(), parallel.to_string()),
            (serial, parallel) => prop_assert!(false, "worker-dependent adapter result for {name}: serial={serial:?}, parallel={parallel:?}"),
        }
    }

    #[test]
    fn malformed_artifact_bytes_are_a_panic_safety_backstop(
        payload in collection::vec(any::<u8>(), 0..4096),
        compressed in any::<bool>(),
    ) {
        let directory = tempdir().unwrap();
        let json_path = directory.path().join(if compressed { "payload.json.gz" } else { "payload.json" });
        fs::write(&json_path, &payload).unwrap();
        let _ = read_json_value(&json_path);

        let jsonl_path = directory.path().join(if compressed { "payload.jsonl.gz" } else { "payload.jsonl" });
        fs::write(&jsonl_path, payload).unwrap();
        let _ = read_jsonl_values(&jsonl_path, false);
    }

    #[test]
    fn generated_artifacts_round_trip_and_compress_deterministically(
        payload in bounded_json(),
        rows in collection::vec(bounded_json(), 0..16),
    ) {
        let directory = tempdir().unwrap();
        let first = directory.path().join("first.json.gz");
        let second = directory.path().join("second.json.gz");
        write_json_gz(&first, &payload).unwrap();
        write_json_gz(&second, &payload).unwrap();

        prop_assert_eq!(read_json_value(&first).unwrap(), payload);
        prop_assert_eq!(fs::read(first).unwrap(), fs::read(second).unwrap());

        let jsonl = directory.path().join("rows.jsonl.gz");
        prop_assert_eq!(write_jsonl_gz(&jsonl, &rows).unwrap(), rows.len());
        prop_assert_eq!(read_jsonl_values(&jsonl, false).unwrap(), rows);
    }

    #[test]
    fn streamed_ingest_differentially_matches_owned_ingest_for_generated_targets(
        ethanol in collection::vec(any::<bool>(), 0..6),
        methanol in collection::vec(any::<bool>(), 0..6),
        ethane in collection::vec(any::<bool>(), 0..6),
        ethanol_key_mode in 0_u8..3,
        methanol_key_mode in 0_u8..3,
        ethane_key_mode in 0_u8..3,
        workers in 1_usize..13,
    ) {
        let task = adversarial_task();
        let mut raw = Map::new();
        insert_generated_target(&mut raw, "ethanol", "CCO", "OCC", &ethanol, ethanol_key_mode);
        insert_generated_target(&mut raw, "methanol", "CO", "OC", &methanol, methanol_key_mode);
        insert_generated_target(&mut raw, "ethane", "CC", "CC", &ethane, ethane_key_mode);
        raw.insert("not-in-task".to_owned(), json!([{"arbitrary": [1, 2, 3]}]));
        let raw = Value::Object(raw);

        let owned = ingest(
            raw.clone(),
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            workers,
        )
        .unwrap();
        let directory = tempdir().unwrap();
        let path = directory.path().join("generated.json.gz");
        write_json(&path, &raw).unwrap();
        let streamed = ingest_file(
            &path,
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            workers,
        )
        .unwrap();

        prop_assert_eq!(serde_json::to_value(streamed).unwrap(), serde_json::to_value(owned).unwrap());
    }
}

#[test]
fn artifact_readers_reject_corruption_and_report_jsonl_rows() {
    let directory = tempdir().unwrap();

    let invalid_gzip = directory.path().join("invalid.json.gz");
    fs::write(&invalid_gzip, b"not gzip").unwrap();
    assert!(read_json_value(&invalid_gzip).is_err());

    let truncated_gzip = directory.path().join("truncated.json.gz");
    write_json_gz(&truncated_gzip, &json!({"targets": [1, 2, 3]})).unwrap();
    let mut bytes = fs::read(&truncated_gzip).unwrap();
    bytes.truncate(bytes.len() / 2);
    fs::write(&truncated_gzip, bytes).unwrap();
    assert!(read_json_value(&truncated_gzip).is_err());

    let invalid_jsonl = directory.path().join("invalid.jsonl");
    fs::write(&invalid_jsonl, b"{\"valid\":true}\nnot-json\n").unwrap();
    let error = read_jsonl_values(&invalid_jsonl, false).unwrap_err();
    assert!(matches!(error, EngineError::Jsonl { line_number: 2, .. }));

    let empty_jsonl = directory.path().join("empty-row.jsonl");
    fs::write(&empty_jsonl, b"{\"valid\":true}\n\n").unwrap();
    let error = read_jsonl_values(&empty_jsonl, false).unwrap_err();
    assert!(matches!(error, EngineError::Jsonl { line_number: 2, .. }));
    assert_eq!(read_jsonl_values(&empty_jsonl, true).unwrap().len(), 1);
}

#[test]
fn artifact_writers_are_deterministic_and_round_trip_hostile_strings() {
    let directory = tempdir().unwrap();
    let first = directory.path().join("first.json.gz");
    let second = directory.path().join("second.json.gz");
    let value = json!({
        "control": "line one\nline two\t\u{0000}",
        "unicode": "λ/🧪/é",
        "quote": "\\\"",
        "nested": [{"z": 1, "a": true}, null]
    });

    write_json_gz(&first, &value).unwrap();
    write_json_gz(&second, &value).unwrap();
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    assert_eq!(read_json_value(&first).unwrap(), value);

    let rows = vec![json!({"b": 2, "a": 1}), json!(["\n", "🧪"]), Value::Null];
    let jsonl = directory.path().join("rows.jsonl.gz");
    assert_eq!(write_jsonl_gz(&jsonl, &rows).unwrap(), rows.len());
    assert_eq!(read_jsonl_values(&jsonl, false).unwrap(), rows);
}

#[test]
fn deeply_nested_json_fails_cleanly_at_the_deserializer_limit() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("deep.json");
    let depth = 256;
    let payload = format!("{}null{}", "[".repeat(depth), "]".repeat(depth));
    fs::write(&path, payload).unwrap();

    assert!(read_json_value(&path).is_err());
}

#[test]
fn full_pipeline_semantics_are_identical_across_worker_counts() {
    let task = adversarial_task();
    let raw = json!({
        "ethanol": [
            {"type": "mol", "smiles": "OCC"},
            {"type": "mol", "smiles": "not-smiles"}
        ],
        "methanol": [{"type": "mol", "smiles": "OC"}],
        "ethane": [{"type": "mol", "smiles": "CC"}]
    });
    let stocks = Stocks::new();
    let mut reference = None;

    for workers in [1, 2, 12] {
        let predictions = ingest(
            raw.clone(),
            &AiZynthFinderAdapter,
            &task,
            AdaptMode::Strict,
            None,
            workers,
        )
        .unwrap();
        let evaluation = score(
            &predictions,
            &task,
            &stocks,
            "full",
            "prefix",
            None,
            workers,
        )
        .unwrap();
        let analysis = analyze(&evaluation, &[1, 10], &[1, 3], 128, 42, workers).unwrap();

        let redundant_options =
            analyze(&evaluation, &[10, 1, 10], &[3, 1, 3], 128, 42, workers).unwrap();
        assert_eq!(
            serde_json::to_value(&redundant_options).unwrap(),
            serde_json::to_value(&analysis).unwrap(),
            "duplicate or reordered metric options changed analysis",
        );

        let mut reordered_candidates = evaluation.clone();
        for target in reordered_candidates.targets.values_mut() {
            target.candidates.reverse();
        }
        let reordered_analysis =
            analyze(&reordered_candidates, &[1, 10], &[1, 3], 128, 42, workers).unwrap();
        assert_eq!(
            serde_json::to_value(&reordered_analysis).unwrap(),
            serde_json::to_value(&analysis).unwrap(),
            "candidate storage order overrode explicit ranks",
        );

        let mut padded_with_failure = evaluation.clone();
        let failure = padded_with_failure.targets["ethanol"]
            .candidates
            .iter()
            .find(|candidate| candidate.failure.is_some())
            .cloned()
            .unwrap();
        let mut later_failure = failure;
        later_failure.rank = usize::MAX;
        padded_with_failure
            .targets
            .get_mut("ethanol")
            .unwrap()
            .candidates
            .push(later_failure);
        let padded_analysis =
            analyze(&padded_with_failure, &[1, 10], &[1, 3], 128, 42, workers).unwrap();
        assert_eq!(
            serde_json::to_value(&padded_analysis).unwrap(),
            serde_json::to_value(&analysis).unwrap(),
            "a later failed candidate changed success metrics",
        );

        let result =
            json!({"predictions": predictions, "evaluation": evaluation, "analysis": analysis});

        if let Some(reference) = &reference {
            assert_eq!(reference, &result, "workers={workers} changed semantics");
        } else {
            reference = Some(result);
        }
    }
}

#[test]
fn zero_workers_is_rejected_before_parallel_work_starts() {
    let task = adversarial_task();
    let error = ingest(
        json!({}),
        &AiZynthFinderAdapter,
        &task,
        AdaptMode::Strict,
        None,
        0,
    )
    .unwrap_err();
    assert!(matches!(error, EngineError::InvalidWorkers(0)));

    let error = score(
        &Predictions::new(),
        &task,
        &Stocks::new(),
        "full",
        "prefix",
        None,
        0,
    )
    .unwrap_err();
    assert!(matches!(error, EngineError::InvalidWorkers(0)));

    let evaluation = score(
        &Predictions::new(),
        &task,
        &Stocks::new(),
        "full",
        "prefix",
        None,
        1,
    )
    .unwrap();
    let error = analyze(&evaluation, &[1], &[1], 8, 42, 0).unwrap_err();
    assert!(matches!(error, EngineError::InvalidWorkers(0)));
}

#[test]
fn zero_bootstrap_resamples_is_rejected_instead_of_panicking() {
    let task = adversarial_task();
    let evaluation = score(
        &Predictions::new(),
        &task,
        &Stocks::new(),
        "full",
        "prefix",
        None,
        1,
    )
    .unwrap();

    let error = analyze(&evaluation, &[1], &[1], 0, 42, 1).unwrap_err();

    assert!(matches!(error, EngineError::InvalidBootstrapResamples(0)));

    let targets = evaluation.targets.into_values().collect::<Vec<_>>();
    let error =
        summarize_target_results(&targets, &[0], &[1], &[1], "task", "full", 0, 42, 1).unwrap_err();
    assert!(matches!(error, EngineError::InvalidBootstrapResamples(0)));
}

fn insert_generated_target(
    raw: &mut Map<String, Value>,
    target_id: &str,
    canonical_smiles: &str,
    equivalent_smiles: &str,
    outcomes: &[bool],
    key_mode: u8,
) {
    let payload = Value::Array(
        outcomes
            .iter()
            .map(|valid| {
                if *valid {
                    json!({"type": "mol", "smiles": equivalent_smiles})
                } else {
                    json!({"type": "mol", "smiles": "not-smiles"})
                }
            })
            .collect(),
    );
    match key_mode {
        0 => {
            raw.insert(target_id.to_owned(), payload);
        }
        1 => {
            raw.insert(canonical_smiles.to_owned(), payload);
        }
        2 => {
            raw.insert(
                canonical_smiles.to_owned(),
                json!([{"type": "mol", "smiles": "wrong-target"}]),
            );
            raw.insert(target_id.to_owned(), payload);
        }
        _ => unreachable!("generated key mode is bounded"),
    }
}

fn adversarial_task() -> Task {
    let targets = [("ethanol", "CCO"), ("methanol", "CO"), ("ethane", "CC")]
        .into_iter()
        .map(|(id, smiles)| {
            let (smiles, inchikey) = chem::normalize(smiles).unwrap();
            (
                id.to_owned(),
                Target {
                    id: id.to_owned(),
                    smiles,
                    inchikey,
                    acceptable_routes: Vec::new(),
                    annotations: Map::new(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    Task {
        name: "adversarial-workers".to_owned(),
        description: String::new(),
        targets,
        default_constraints: Vec::new(),
        constraints: BTreeMap::new(),
        metric_label: None,
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}
