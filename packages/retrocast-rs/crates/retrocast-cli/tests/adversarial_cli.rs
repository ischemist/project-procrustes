use std::{
    fs,
    process::{Command, Output},
};

use tempfile::tempdir;

#[test]
fn hostile_project_selectors_are_rejected_without_creating_outputs() {
    let directory = tempdir().unwrap();
    for (flag, value) in [
        ("--model", "../escape"),
        ("--model", "folder\\escape"),
        ("--dataset", "../escape"),
        ("--dataset", "/absolute"),
        ("--dataset", "a/b"),
    ] {
        let mut arguments = vec![
            "--data-dir",
            directory.path().to_str().unwrap(),
            "ingest",
            "--model",
            "model",
            "--dataset",
            "dataset",
        ];
        let position = arguments
            .iter()
            .position(|argument| *argument == flag)
            .unwrap();
        arguments[position + 1] = value;
        assert_clean_failure(&arguments);
    }

    assert!(!directory.path().join("3-processed").exists());
}

#[test]
fn malformed_artifacts_and_invalid_execution_options_fail_cleanly() {
    let directory = tempdir().unwrap();
    let input = directory.path().join("input.json");
    let output = directory.path().join("output.json");
    fs::write(&input, b"[]").unwrap();

    assert_clean_failure(&[
        "adapt",
        "--input",
        input.to_str().unwrap(),
        "--adapter",
        "aizynthfinder",
        "--output",
        output.to_str().unwrap(),
        "--workers",
        "0",
    ]);
    assert!(!output.exists());

    assert_clean_failure(&[
        "adapt",
        "--input",
        input.to_str().unwrap(),
        "--adapter",
        "does-not-exist",
        "--output",
        output.to_str().unwrap(),
    ]);
    assert_clean_failure(&[
        "adapt",
        "--input",
        input.to_str().unwrap(),
        "--adapter",
        "aizynthfinder",
        "--output",
        output.to_str().unwrap(),
        "--mode",
        "anything-goes",
    ]);

    let corrupt = directory.path().join("corrupt.json.gz");
    fs::write(&corrupt, b"not a gzip stream").unwrap();
    assert_clean_failure(&[
        "adapt",
        "--input",
        corrupt.to_str().unwrap(),
        "--adapter",
        "aizynthfinder",
        "--output",
        output.to_str().unwrap(),
    ]);
    assert!(!output.exists());

    let evaluation = directory.path().join("evaluation.json");
    fs::write(
        &evaluation,
        br#"{
            "task":{"name":"empty","targets":{}},
            "tiers":[0],
            "metric_label":"task",
            "acceptable_match_level":"full",
            "acceptable_route_match":"prefix",
            "targets":{}
        }"#,
    )
    .unwrap();
    assert_clean_failure(&[
        "analyze",
        "--evaluation",
        evaluation.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--n-boot",
        "0",
    ]);
    assert!(!output.exists());
}

#[test]
fn pipeline_rejects_manifest_path_traversal_before_reading_external_inputs() {
    let directory = tempdir().unwrap();
    let raw = directory.path().join("raw");
    fs::create_dir(&raw).unwrap();
    fs::write(
        raw.join("manifest.json"),
        br#"{"directives":{"raw_results_filename":"../../outside.json"}}"#,
    )
    .unwrap();
    let output = directory.path().join("output");

    assert_clean_failure(&[
        "pipeline",
        "--raw",
        raw.to_str().unwrap(),
        "--benchmark",
        directory.path().join("missing-task.json").to_str().unwrap(),
        "--stock",
        directory.path().join("missing-stock.csv").to_str().unwrap(),
        "--output-dir",
        output.to_str().unwrap(),
    ]);
    assert!(!output.exists());
}

#[test]
fn partial_file_mode_arguments_are_rejected_without_side_effects() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("output.json");
    assert_clean_failure(&[
        "ingest",
        "--input",
        "only-one-argument.json",
        "--output",
        output.to_str().unwrap(),
    ]);
    assert_clean_failure(&[
        "score",
        "--candidates",
        "only-one-argument.json",
        "--output",
        output.to_str().unwrap(),
    ]);
    assert_clean_failure(&["analyze", "--output", output.to_str().unwrap()]);
    assert!(!output.exists());
}

#[test]
fn adapter_listing_is_complete_unique_and_deterministic() {
    let first = run(&["list-adapters"]);
    let second = run(&["list-adapters"]);
    assert!(first.status.success());
    assert_eq!(first.stdout, second.stdout);

    let output = String::from_utf8(first.stdout).unwrap();
    let canonical = output
        .lines()
        .filter(|line| !line.contains("deprecated alias"))
        .collect::<Vec<_>>();
    assert_eq!(canonical.len(), 13);
    let mut unique = canonical.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), canonical.len());
}

fn assert_clean_failure(arguments: &[&str]) {
    let output = run(arguments);
    assert!(
        !output.status.success(),
        "retrocast {arguments:?} unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.is_empty(),
        "retrocast {arguments:?} failed silently"
    );
    assert!(
        !stderr.contains("panicked at") && !stderr.contains("stack backtrace:"),
        "retrocast {arguments:?} panicked:\n{stderr}"
    );
}

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_retrocast"))
        .args(arguments)
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap()
}
