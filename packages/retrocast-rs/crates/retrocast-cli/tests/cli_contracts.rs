use std::{collections::BTreeMap, fs, process::Command};

use retrocast_core::{
    chem,
    io::{read_json, write_csv_gz, write_json},
    model::{AnalysisReport, Constraint, Target, Task},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use url::Url;

#[test]
fn evaluate_command_emits_a_verifiable_release_and_verify_detects_tampering() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let raw = root.join("raw");
    let benchmark = root.join("benchmark.json.gz");
    let stock = root.join("test-stock.csv.gz");
    let output = root.join("release");
    fs::create_dir(&raw).unwrap();
    write_json(
        &raw.join("results.json.gz"),
        &json!({"propane": [aizynth_route()]}),
    )
    .unwrap();
    write_json(
        &raw.join("manifest.json"),
        &json!({"directives": {"raw_results_filename": "results.json.gz"}}),
    )
    .unwrap();
    write_json(&benchmark, &task()).unwrap();
    write_csv_gz(
        &stock,
        &[
            vec!["SMILES".to_owned(), "InChIKey".to_owned()],
            vec!["C".to_owned(), inchikey("C")],
            vec!["CC".to_owned(), inchikey("CC")],
        ],
    )
    .unwrap();

    let evaluation = run(&[
        "evaluate",
        "--raw",
        raw.to_str().unwrap(),
        "--benchmark",
        benchmark.to_str().unwrap(),
        "--stock",
        stock.to_str().unwrap(),
        "--output-dir",
        output.to_str().unwrap(),
        "--n-boot",
        "16",
        "--workers",
        "2",
    ]);
    assert!(
        evaluation.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluation.stderr)
    );
    let stats: Value = serde_json::from_slice(&evaluation.stdout).unwrap();
    assert_eq!(stats["targets"], 1);
    assert_eq!(stats["candidates"], 1);
    let report: AnalysisReport = read_json(&output.join("analysis.json.gz")).unwrap();
    assert_eq!(report.metrics["solv_0[test-stock]_rate"].value, 1.0);

    let verify = run(&[
        "verify",
        "--manifest",
        output.join("manifest.json").to_str().unwrap(),
        "--root-dir",
        output.to_str().unwrap(),
    ]);
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let verification: Value = serde_json::from_slice(&verify.stdout).unwrap();
    assert_eq!(verification[0]["is_valid"], true);

    write_json(
        &output.join("evaluation.json.gz"),
        &json!({"tampered": true}),
    )
    .unwrap();
    let tampered = run(&[
        "verify",
        "--manifest",
        output.join("manifest.json").to_str().unwrap(),
        "--root-dir",
        output.to_str().unwrap(),
    ]);
    assert!(!tampered.status.success());
    assert!(String::from_utf8_lossy(&tampered.stderr).contains("manifest verification failed"));
}

#[test]
fn get_data_dry_run_prints_benchmark_dependencies_without_materializing_them() {
    let directory = tempdir().unwrap();
    let remote = directory.path().join("remote");
    let output = directory.path().join("output");
    fs::create_dir(&remote).unwrap();
    let entries = [
        "1-benchmarks/definitions/mkt-cnv-160.json.gz",
        "1-benchmarks/stocks/buyables-stock.csv.gz",
        "2-raw/model/mkt-cnv-160/results.json.gz",
    ];
    fs::write(
        remote.join("SHA256SUMS"),
        entries
            .iter()
            .map(|entry| format!("{}  {entry}\n", hex_hash(entry.as_bytes())))
            .collect::<String>(),
    )
    .unwrap();

    let base_url = Url::from_directory_path(&remote).unwrap();
    let result = run(&[
        "get-data",
        "mkt-cnv-160",
        "--dir",
        output.to_str().unwrap(),
        "--base-url",
        base_url.as_ref(),
        "--dry-run",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let listed = String::from_utf8(result.stdout).unwrap();
    assert!(listed.contains(entries[0]));
    assert!(listed.contains(entries[1]));
    assert!(!listed.contains(entries[2]));
    assert!(output.join("SHA256SUMS").is_file());
    assert!(!output.join(entries[0]).exists());
    assert!(!output.join(entries[1]).exists());
}

fn run(arguments: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_retrocast"))
        .args(arguments)
        .output()
        .unwrap()
}

fn task() -> Task {
    let (smiles, inchikey) = chem::normalize("CCC").unwrap();
    Task {
        name: "cli-contract".to_owned(),
        description: String::new(),
        targets: BTreeMap::from([(
            "propane".to_owned(),
            Target {
                id: "propane".to_owned(),
                smiles,
                inchikey,
                acceptable_routes: Vec::new(),
                annotations: Map::new(),
            },
        )]),
        default_constraints: vec![Constraint {
            kind: "retrocast.stock_termination".to_owned(),
            fields: Map::from_iter([("stock".to_owned(), Value::String("test-stock".to_owned()))]),
        }],
        constraints: BTreeMap::new(),
        metric_label: None,
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

fn aizynth_route() -> Value {
    json!({
        "type": "mol",
        "smiles": "CCC",
        "children": [{
            "type": "reaction",
            "smiles": "CC.C>>CCC",
            "children": [
                {"type": "mol", "smiles": "CC", "in_stock": true, "children": []},
                {"type": "mol", "smiles": "C", "in_stock": true, "children": []}
            ]
        }]
    })
}

fn inchikey(smiles: &str) -> String {
    chem::normalize(smiles).unwrap().1.into_string()
}

fn hex_hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
