use std::{collections::BTreeMap, fs};

use proptest::{collection, prelude::*};
use retrocast_core::{
    adapt::ingest,
    adapters::{self, AiZynthFinderAdapter},
    analyze::analyze,
    chem,
    error::EngineError,
    io::{
        read_json_value, read_jsonl_values, validate_path_component, write_json_gz, write_jsonl_gz,
    },
    model::{Candidate, Evaluation, Predictions, Target, Task},
    route::AdaptMode,
    route_path::RoutePath,
    schema::{CanonicalSmiles, InchiKey, ReactionSmiles, SchemaVersion},
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
    fn route_paths_round_trip_canonically(
        indices in collection::vec(any::<usize>(), 0..32),
        molecule in any::<bool>(),
    ) {
        let path = if molecule {
            RoutePath::Molecule(indices.into_boxed_slice())
        } else {
            RoutePath::Reaction(indices.into_boxed_slice())
        };
        let encoded = path.to_string();
        let decoded = RoutePath::parse(&encoded).unwrap();

        prop_assert_eq!(&decoded, &path);
        prop_assert_eq!(serde_json::from_str::<RoutePath>(&serde_json::to_string(&path).unwrap()).unwrap(), path);
    }

    #[test]
    fn path_component_validation_matches_its_security_boundary(value in bounded_string()) {
        let unsafe_component = value.is_empty()
            || value == "."
            || value == ".."
            || value.contains('/')
            || value.contains('\\')
            || value.contains('\0');

        prop_assert_eq!(validate_path_component(&value, "component").is_err(), unsafe_component);
    }

    #[test]
    fn scalar_deserializers_never_accept_values_outside_their_wire_invariants(value in bounded_string()) {
        let wire = serde_json::to_string(&value).unwrap();
        let key_is_valid = {
            let bytes = value.as_bytes();
            bytes.len() == 27
                && bytes[14] == b'-'
                && bytes[25] == b'-'
                && bytes.iter().enumerate().all(|(index, byte)| {
                    matches!(index, 14 | 25) || byte.is_ascii_uppercase()
                })
        };

        prop_assert_eq!(serde_json::from_str::<CanonicalSmiles>(&wire).is_ok(), !value.is_empty());
        prop_assert_eq!(serde_json::from_str::<ReactionSmiles>(&wire).is_ok(), !value.is_empty());
        prop_assert_eq!(serde_json::from_str::<InchiKey>(&wire).is_ok(), key_is_valid);
        prop_assert_eq!(serde_json::from_str::<SchemaVersion>(&wire).is_ok(), value == "2");
    }

    #[test]
    fn candidate_deserialization_enforces_exactly_one_ranked_outcome(
        rank in any::<u16>(),
        has_route in any::<bool>(),
        has_failure in any::<bool>(),
    ) {
        let mut candidate = Map::from_iter([("rank".to_owned(), Value::from(rank))]);
        if has_route {
            candidate.insert("route".to_owned(), valid_route());
        }
        if has_failure {
            candidate.insert("failure".to_owned(), json!({"code": "provider.failure"}));
        }

        let parsed = serde_json::from_value::<Candidate>(Value::Object(candidate));
        prop_assert_eq!(parsed.is_ok(), rank > 0 && (has_route ^ has_failure));
    }

    #[test]
    fn every_adapter_handles_bounded_arbitrary_json_without_panicking(payload in bounded_json()) {
        for &name in adapters::BUILT_IN_ADAPTERS {
            let adapter = adapters::built_in(name).expect("registered built-in adapter");
            if let Ok(entries) = adapter.entries(payload.clone(), Some("adversarial-source")) {
                for entry in entries.into_iter().take(32) {
                    let _ = adapter.cast(entry.payload, AdaptMode::Strict, None);
                }
            }
        }
    }

    #[test]
    fn supported_wire_models_reject_arbitrary_json_without_panicking(payload in bounded_json()) {
        let encoded = serde_json::to_vec(&payload).unwrap();
        let _ = serde_json::from_slice::<Task>(&encoded);
        let _ = serde_json::from_slice::<Predictions>(&encoded);
        let _ = serde_json::from_slice::<Evaluation>(&encoded);
    }

    #[test]
    fn artifact_readers_handle_arbitrary_bytes_without_panicking(
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
}

#[test]
fn route_paths_reject_noncanonical_and_overflowing_indices() {
    for value in [
        "",
        "rc:",
        "rc:x:/0",
        "rc:m:",
        "rc:m:0",
        "rc:m://0",
        "rc:m:/00",
        "rc:m:/+1",
        "rc:m:/-1",
        "rc:m:/ 1",
        "rc:m:/1 ",
        "rc:m:/184467440737095516160000000000000000000",
        "rc:m:/0/",
    ] {
        assert!(RoutePath::parse(value).is_err(), "accepted {value:?}");
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
        let analysis = analyze(&evaluation, &[10, 1, 10], &[3, 1, 3], 128, 42, workers).unwrap();
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

fn valid_route() -> Value {
    json!({
        "target": {
            "smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        },
        "schema_version": "2"
    })
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
