use std::fs;

use retrocast_core::dataset::{
    HostedDataRequest, TrainingDataRequest, download_hosted_data, download_training_data,
};
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use url::Url;

#[test]
fn training_dry_run_resolves_the_published_contract_without_writing_artifacts() {
    let directory = tempdir().unwrap();
    let remote = directory.path().join("remote");
    let output = directory.path().join("output");
    let release = "v2026-07-15";
    let release_root = remote.join("paroutes").join(release);
    fs::create_dir_all(&release_root).unwrap();
    let entries = [
        "route-holdout-n1-n5/training.jsonl.gz",
        "route-holdout-n1-n5/validation.jsonl.gz",
        "route-holdout-n1-n5/manifest.json",
        "n1-routes/all.jsonl.gz",
    ];
    write_checksum_index(&release_root, &entries);

    let paths = download_training_data(&TrainingDataRequest {
        dataset: "paroutes".to_owned(),
        artifact: Some("route-holdout-n1-n5".to_owned()),
        split: None,
        release: release.to_owned(),
        format: Some("jsonl".to_owned()),
        omit: vec!["validation".to_owned()],
        cache_dir: None,
        output_dir: Some(output.clone()),
        base_url: Url::from_directory_path(&remote).unwrap().to_string(),
        dry_run: true,
    })
    .unwrap();

    assert_eq!(
        paths,
        vec![
            output.join(release).join(entries[0]),
            output.join(release).join(entries[2]),
        ]
    );
    assert!(output.join(release).join("SHA256SUMS").is_file());
    assert!(paths.iter().all(|path| !path.exists()));
}

#[test]
fn hosted_benchmark_target_expands_to_its_definition_and_stock_dependency_only() {
    let directory = tempdir().unwrap();
    let remote = directory.path().join("remote");
    let output = directory.path().join("output");
    fs::create_dir_all(&remote).unwrap();
    let entries = [
        "1-benchmarks/definitions/mkt-lin-500.json.gz",
        "1-benchmarks/stocks/buyables-stock.csv.gz",
        "1-benchmarks/definitions/ref-lin-600.json.gz",
        "2-raw/model/mkt-lin-500/results.json.gz",
    ];
    write_checksum_index(&remote, &entries);

    let mut paths = download_hosted_data(&HostedDataRequest {
        target: "mkt-lin-500".to_owned(),
        cache_dir: None,
        output_dir: Some(output.clone()),
        base_url: Url::from_directory_path(&remote).unwrap().to_string(),
        dry_run: true,
    })
    .unwrap();
    paths.sort();

    let mut expected = vec![output.join(entries[0]), output.join(entries[1])];
    expected.sort();
    assert_eq!(paths, expected);
    assert!(paths.iter().all(|path| !path.exists()));
    assert!(output.join("SHA256SUMS").is_file());
}

fn write_checksum_index(root: &std::path::Path, entries: &[&str]) {
    let contents = entries
        .iter()
        .map(|entry| format!("{}  {entry}", hex_hash(entry.as_bytes())))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(root.join("SHA256SUMS"), format!("{contents}\n")).unwrap();
}

fn hex_hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
