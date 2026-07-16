use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use retrocast_core::{
    chem,
    io::{read_json, write_csv_gz, write_json},
    model::{Constraint, Evaluation, Target, Task},
};
use serde_json::{Map, Value, json};

#[test]
fn standalone_binary_runs_the_project_lifecycle() {
    let root = temporary_directory();
    let benchmark_path = root.join("1-benchmarks/definitions/small.json.gz");
    let stock_path = root.join("1-benchmarks/stocks/test-stock.csv.gz");
    let raw_dir = root.join("2-raw/test-model/small");
    std::fs::create_dir_all(&raw_dir).unwrap();

    let (smiles, inchikey) = chem::normalize("CCO").unwrap();
    let task = Task {
        name: "small".to_owned(),
        description: String::new(),
        targets: BTreeMap::from([(
            "ethanol".to_owned(),
            Target {
                id: "ethanol".to_owned(),
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
    };
    write_json(&benchmark_path, &task).unwrap();
    let (_, carbon_key) = chem::normalize("C").unwrap();
    write_csv_gz(
        &stock_path,
        &[
            vec!["SMILES".to_owned(), "InChIKey".to_owned()],
            vec!["C".to_owned(), carbon_key.into_string()],
        ],
    )
    .unwrap();
    write_json(
        &raw_dir.join("results.json.gz"),
        &json!({"ethanol": raw_route()}),
    )
    .unwrap();
    write_json(
        &raw_dir.join("execution_stats.json.gz"),
        &json!({"wall_time": {"ethanol": 12.5}, "cpu_time": {"ethanol": 3.25}}),
    )
    .unwrap();
    write_json(
        &raw_dir.join("manifest.json"),
        &json!({
            "schema_version": "2",
            "directives": {"adapter": "paroutes", "raw_results_filename": "results.json.gz"}
        }),
    )
    .unwrap();

    run(
        &root,
        &["ingest", "--model", "test-model", "--dataset", "small"],
    );
    run(
        &root,
        &["score", "--model", "test-model", "--dataset", "small"],
    );
    run(
        &root,
        &[
            "analyze",
            "--model",
            "test-model",
            "--dataset",
            "small",
            "--n-boot",
            "10",
        ],
    );

    let evaluation_path = root.join("4-scored/small/test-model/test-stock/evaluation.json.gz");
    let evaluation: Evaluation = read_json(&evaluation_path).unwrap();
    let target = &evaluation.targets["ethanol"];
    assert_eq!(target.wall_time, Some(12.5));
    assert_eq!(target.cpu_time, Some(3.25));
    assert!(target.candidates[0].satisfies_task());
    assert!(
        root.join("5-results/small/test-model/test-stock/analysis.json.gz")
            .is_file()
    );
    assert!(
        root.join("5-results/small/test-model/test-stock/report.md")
            .is_file()
    );
    assert!(
        root.join("5-results/small/test-model/test-stock/manifest.json")
            .is_file()
    );

    std::fs::remove_dir_all(root).unwrap();
}

fn run(root: &Path, arguments: &[&str]) {
    let status = Command::new(env!("CARGO_BIN_EXE_retrocast"))
        .arg("--data-dir")
        .arg(root)
        .args(arguments)
        .status()
        .unwrap();
    assert!(status.success(), "retrocast {arguments:?} failed");
}

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("retrocast-project-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn raw_route() -> Value {
    json!({
        "type": "mol",
        "smiles": "CCO",
        "children": [{
            "type": "reaction",
            "smiles": "CCO",
            "metadata": {"ID": "US123;1", "rsmi": "C.CC>>CCO"},
            "children": [
                {"type": "mol", "smiles": "C", "in_stock": true, "children": []},
                {"type": "mol", "smiles": "CC", "children": [{
                    "type": "reaction",
                    "smiles": "CC",
                    "metadata": {"ID": "US123;2", "rsmi": "C>>CC"},
                    "children": [
                        {"type": "mol", "smiles": "C", "in_stock": true, "children": []}
                    ]
                }]}
            ]
        }]
    })
}
