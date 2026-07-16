use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

use retrocast_core::{
    chem,
    curation::{excise_reactions, generate_pruned_routes},
    dataset::{DatasetError, TrainingSetRequest, download_training_set},
    io::{read_json, write_csv_gz, write_json, write_jsonl_gz},
    model::{AnalysisReport, Evaluation, Molecule, Reaction, Route, Target, Task},
    pipeline::{PipelineOptions, run_pipeline},
    provenance::{
        ContentType, Manifest, ManifestOutput, VerificationCategory, VerificationLevel,
        create_manifest, file_hash, verify_manifest,
    },
    route::AdaptMode,
    route_path::RoutePath,
    route_view::InchiKeyLevel,
    schema::ReactionSmiles,
    training::{
        TrainingSetBuildConfig, adapt_training_routes, audit_route_release,
        audit_single_step_release, build_test_reaction_records, build_test_route_records,
        build_training_reaction_release, build_training_route_release,
    },
};
use serde_json::{Map, Value, json};
use tempfile::tempdir;
use url::Url;

#[test]
fn curation_excision_and_pruning_preserve_valid_route_boundaries() {
    let route = two_step_route();
    let root_reaction = RoutePath::target().produced_by().unwrap();
    let intermediate = root_reaction.reactant(0).unwrap();
    let intermediate_reaction = intermediate.produced_by().unwrap();

    let root_signature = route
        .reaction_at(&root_reaction)
        .unwrap()
        .signature(InchiKeyLevel::Full);
    let intermediate_signature = route
        .reaction_at(&intermediate_reaction)
        .unwrap()
        .signature(InchiKeyLevel::Full);

    let root_excised = excise_reactions(&route, &HashSet::from([root_signature]));
    assert_eq!(root_excised.len(), 1);
    assert_eq!(root_excised[0].target.smiles, "CC");
    assert_eq!(root_excised[0].depth(), 1);
    assert_eq!(root_excised[0].annotations["source"], "curation-contract");

    let intermediate_excised = excise_reactions(&route, &HashSet::from([intermediate_signature]));
    assert_eq!(intermediate_excised.len(), 1);
    assert_eq!(intermediate_excised[0].target.smiles, "CCC");
    assert_eq!(intermediate_excised[0].depth(), 1);
    let excised_leaves: HashSet<_> = intermediate_excised[0]
        .leaves()
        .into_iter()
        .map(|leaf| leaf.value.smiles.to_string())
        .collect();
    assert_eq!(
        excised_leaves,
        HashSet::from(["C".to_owned(), "CC".to_owned()])
    );

    let stock = HashSet::from([inchikey("C"), inchikey("CC")]);
    let mut pruned = generate_pruned_routes(&route, &stock);
    pruned.sort_by_key(Route::depth);
    assert_eq!(pruned.len(), 2);
    assert_eq!(pruned.iter().map(Route::depth).collect::<Vec<_>>(), [1, 2]);
    assert!(pruned.iter().all(|candidate| {
        candidate
            .leaves()
            .iter()
            .all(|leaf| stock.contains(leaf.value.inchikey.as_str()))
    }));
}

#[test]
fn provenance_verification_distinguishes_chain_breaks_from_disk_tampering() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let source = root.join("0-assets/source.json");
    let parent_output = root.join("3-processed/demo/candidates.json");
    let child_output = root.join("4-scored/demo/evaluation.json");
    write_json(&source, &json!({"input": true})).unwrap();
    write_json(&parent_output, &json!({"candidate": 1})).unwrap();
    write_json(&child_output, &json!({"score": 1})).unwrap();

    let parent = create_manifest(
        "ingest:v2",
        std::slice::from_ref(&source),
        &[manifest_output(&parent_output, json!({"candidate": 1}))],
        root,
        Map::new(),
        Map::new(),
        Map::new(),
        Map::new(),
        None,
        false,
    )
    .unwrap();
    let parent_manifest = parent_output.parent().unwrap().join("manifest.json");
    write_json(&parent_manifest, &parent).unwrap();

    let mut child = create_manifest(
        "score:v2",
        std::slice::from_ref(&parent_output),
        &[manifest_output(&child_output, json!({"score": 1}))],
        root,
        Map::new(),
        Map::new(),
        Map::new(),
        Map::new(),
        None,
        false,
    )
    .unwrap();
    let child_manifest = child_output.parent().unwrap().join("manifest.json");
    write_json(&child_manifest, &child).unwrap();

    let valid = verify_manifest(&child_manifest, root, true, false, false);
    assert!(
        valid.is_valid,
        "unexpected verification issues: {:?}",
        valid.issues
    );
    assert!(valid.issues.iter().any(|issue| {
        issue.category == Some(VerificationCategory::Graph)
            && issue.level == VerificationLevel::Pass
            && issue.message.contains("2 manifests")
    }));

    write_json(&parent_output, &json!({"candidate": "tampered"})).unwrap();
    let tampered = verify_manifest(&child_manifest, root, true, false, false);
    assert!(!tampered.is_valid);
    assert!(tampered.issues.iter().any(|issue| {
        issue.category == Some(VerificationCategory::Phase2)
            && issue.message.contains("HASH MISMATCH")
    }));

    write_json(&parent_output, &json!({"candidate": 1})).unwrap();
    child.source_files[0].file_hash = "0".repeat(64);
    write_json(&child_manifest, &child).unwrap();
    let broken_chain = verify_manifest(&child_manifest, root, true, false, false);
    assert!(!broken_chain.is_valid);
    assert!(broken_chain.issues.iter().any(|issue| {
        issue.category == Some(VerificationCategory::Phase1)
            && issue
                .message
                .contains("Hash mismatch between parent and child")
    }));
}

#[test]
fn standalone_pipeline_writes_a_self_describing_scored_release() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let raw_dir = root.join("raw");
    let task_path = root.join("benchmark.json.gz");
    let stock_path = root.join("contract-stock.csv.gz");
    let stats_path = root.join("execution-stats.json.gz");
    let output = root.join("pipeline-output");

    fs::create_dir_all(&raw_dir).unwrap();
    write_json(
        &raw_dir.join("results.json.gz"),
        &json!({"propane": [aizynth_raw_route()]}),
    )
    .unwrap();
    write_json(
        &raw_dir.join("manifest.json"),
        &json!({"directives": {"raw_results_filename": "results.json.gz"}}),
    )
    .unwrap();
    write_json(&task_path, &task_for("propane", "CCC")).unwrap();
    write_json(
        &stats_path,
        &json!({"wall_time": {"propane": 2.5}, "cpu_time": {"propane": 1.25}}),
    )
    .unwrap();
    write_csv_gz(
        &stock_path,
        &[
            vec!["SMILES".to_owned(), "InChIKey".to_owned()],
            vec!["C".to_owned(), inchikey("C")],
            vec!["CC".to_owned(), inchikey("CC")],
        ],
    )
    .unwrap();

    let pipeline_stats = run_pipeline(
        &raw_dir,
        &task_path,
        &stock_path,
        None,
        Some(&stats_path),
        &output,
        &PipelineOptions {
            adapter: "aizynthfinder",
            mode: AdaptMode::Strict,
            max_candidates: None,
            workers: 2,
            match_level: "full",
            acceptable_route_match: "prefix",
            ks: &[1, 5],
            prefix_depths: &[1, 2],
            n_boot: 32,
            seed: 7,
        },
    )
    .unwrap();

    assert_eq!(pipeline_stats.targets, 1);
    assert_eq!(pipeline_stats.candidates, 1);
    assert_eq!(pipeline_stats.workers, 2);
    assert!(pipeline_stats.total_seconds > 0.0);

    let evaluation: Evaluation = read_json(&output.join("evaluation.json.gz")).unwrap();
    let target = &evaluation.targets["propane"];
    assert_eq!(target.wall_time, Some(2.5));
    assert_eq!(target.cpu_time, Some(1.25));
    assert!(target.candidates[0].satisfies_task());
    assert_eq!(
        evaluation.task.default_constraints[0].fields["stock"],
        "contract-stock"
    );

    let analysis: AnalysisReport = read_json(&output.join("analysis.json.gz")).unwrap();
    assert_eq!(analysis.bootstrap_resamples, 32);
    assert_eq!(analysis.runtime.total_wall_time, Some(2.5));

    let manifest: Manifest = read_json(&output.join("manifest.json")).unwrap();
    assert_eq!(manifest.action, "pipeline:v2");
    assert_eq!(manifest.parameters["workers"], 2);
    assert_eq!(manifest.statistics["targets"], 1);
    assert_eq!(manifest.output_files().count(), 4);
    assert!(verify_manifest(&output.join("manifest.json"), &output, false, false, false).is_valid);
}

#[test]
fn training_release_builders_preserve_occurrences_and_enforce_audit_boundaries() {
    let raw = paroutes_raw_route();
    let adaptation = adapt_training_routes(json!([raw.clone(), raw]), "all").unwrap();
    assert_eq!(adaptation.stats.raw_routes, 2);
    assert_eq!(adaptation.stats.adapted_routes, 2);
    assert_eq!(adaptation.routes[0].source.raw_index, 0);
    assert_eq!(adaptation.routes[1].source.raw_index, 1);

    let test_routes = build_test_route_records("n1", &adaptation.routes, "paroutes");
    let test_reactions = build_test_reaction_records("n1", &test_routes, "paroutes").unwrap();
    assert_eq!(test_routes.len(), 2);
    assert_eq!(test_reactions.len(), 4);
    assert_eq!(
        test_reactions
            .iter()
            .flat_map(|record| record.sources.iter().map(|source| source.step_index))
            .collect::<Vec<_>>(),
        [Some(1), Some(2), Some(1), Some(2)]
    );

    let route_config = TrainingSetBuildConfig {
        holdout_mode: "route".to_owned(),
        val_fraction: 0.0,
        seed: 20_260_502,
        route_prefix: "paroutes".to_owned(),
    };
    let route_release = build_training_route_release(
        &adaptation.routes,
        serde_json::to_value(&adaptation.stats).unwrap(),
        &BTreeMap::new(),
        BTreeMap::new(),
        &route_config,
    )
    .unwrap();
    assert_eq!(route_release.records.len(), 1);
    assert_eq!(route_release.records[0].sources.len(), 2);
    assert_eq!(
        route_release.summary["postprocessing"]["chemical_duplicates_removed"],
        1
    );

    let training_routes = route_release.records.clone();
    audit_route_release(
        &route_release.release_name,
        &route_release.records,
        &training_routes,
        &[],
    )
    .unwrap();

    let reaction_config = TrainingSetBuildConfig {
        holdout_mode: "reaction".to_owned(),
        ..route_config
    };
    let reactions =
        build_training_reaction_release(&route_release.records, &reaction_config).unwrap();
    assert_eq!(reactions.records.len(), 2);
    assert!(
        reactions
            .records
            .iter()
            .all(|record| record.split == "training")
    );
    let parent_ids = HashSet::from([route_release.records[0].id.clone()]);
    let audit = audit_single_step_release(
        &reactions.release_name,
        &reactions.records,
        &reactions.records,
        &[],
        reactions.records.len(),
        reactions.records.len(),
        0,
        &parent_ids,
    )
    .unwrap();
    assert_eq!(audit["parent_routes"], 1);
    assert_eq!(audit["cross_split_reaction_identity_overlap"], 0);

    let missing_parent = audit_single_step_release(
        &reactions.release_name,
        &reactions.records,
        &reactions.records,
        &[],
        reactions.records.len(),
        reactions.records.len(),
        0,
        &HashSet::new(),
    )
    .unwrap_err();
    assert_eq!(
        missing_parent.code,
        "workflow.single_step_release_audit_failed"
    );
    assert_eq!(
        missing_parent.context["failures"]["missing_parent_route_sources"],
        2
    );

    let malformed = adapt_training_routes(json!([{"bad": "route"}]), "all").unwrap_err();
    assert_eq!(malformed.code, "adapter.schema_invalid");
}

#[test]
fn dataset_download_repairs_corrupt_cache_and_rejects_bad_remote_bytes() {
    let directory = tempdir().unwrap();
    let remote = directory.path().join("remote");
    let cache = directory.path().join("cache");
    let release = "v2026-05-12";
    let artifact = "route-holdout-n1-n5";
    let relative_artifact = format!("{artifact}/training.jsonl.gz");
    let relative_manifest = format!("{artifact}/manifest.json");
    let remote_release = remote.join("paroutes").join(release);
    let remote_artifact = remote_release.join(&relative_artifact);
    let remote_manifest = remote_release.join(&relative_manifest);
    fs::create_dir_all(remote_artifact.parent().unwrap()).unwrap();
    write_jsonl_gz(
        &remote_artifact,
        &[json!({"id": "route-1", "split": "training"})],
    )
    .unwrap();
    write_json(&remote_manifest, &json!({"action": "training-release"})).unwrap();
    let published_bytes = fs::read(&remote_artifact).unwrap();
    fs::write(
        remote_release.join("SHA256SUMS"),
        format!(
            "{}  {relative_artifact}\n{}  {relative_manifest}\n",
            file_hash(&remote_artifact).unwrap(),
            file_hash(&remote_manifest).unwrap(),
        ),
    )
    .unwrap();
    fs::write(
        remote.join("paroutes/latest.json"),
        format!(r#"{{"dataset":"paroutes","latest_release":"{release}"}}"#),
    )
    .unwrap();

    let request = TrainingSetRequest {
        dataset: "paroutes".to_owned(),
        artifact: artifact.to_owned(),
        split: "training".to_owned(),
        release: "latest".to_owned(),
        format: "jsonl".to_owned(),
        cache_dir: Some(cache.clone()),
        output_dir: None,
        base_url: Url::from_directory_path(&remote).unwrap().to_string(),
    };

    let downloaded = download_training_set(&request).unwrap();
    assert_eq!(
        downloaded,
        cache
            .join("paroutes")
            .join(release)
            .join(&relative_artifact)
    );
    assert_eq!(fs::read(&downloaded).unwrap(), published_bytes);
    assert!(downloaded.parent().unwrap().join("manifest.json").is_file());

    fs::write(&downloaded, b"corrupt cache").unwrap();
    assert_eq!(download_training_set(&request).unwrap(), downloaded);
    assert_eq!(
        fs::read(&downloaded).unwrap(),
        fs::read(&remote_artifact).unwrap()
    );

    fs::write(&remote_artifact, b"bad remote bytes").unwrap();
    fs::write(&downloaded, b"force redownload").unwrap();
    let error = download_training_set(&request).unwrap_err();
    assert!(matches!(
        error,
        DatasetError::Policy { ref code, .. } if code == "dataset.hash_mismatch"
    ));
    assert!(!downloaded.exists());
    assert!(
        !cache
            .join("paroutes")
            .join(release)
            .join("SHA256SUMS")
            .exists()
    );
}

fn manifest_output(path: &Path, value: Value) -> ManifestOutput {
    ManifestOutput {
        label: None,
        path: path.to_path_buf(),
        value,
        content_type: ContentType::Unknown,
        content_hash: None,
    }
}

fn task_for(id: &str, smiles: &str) -> Task {
    let (smiles, inchikey) = chem::normalize(smiles).unwrap();
    Task {
        name: "workflow-contract".to_owned(),
        description: String::new(),
        targets: BTreeMap::from([(
            id.to_owned(),
            Target {
                id: id.to_owned(),
                smiles,
                inchikey,
                acceptable_routes: Vec::new(),
                annotations: Map::new(),
            },
        )]),
        default_constraints: Vec::new(),
        constraints: BTreeMap::new(),
        metric_label: None,
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

fn two_step_route() -> Route {
    let intermediate = molecule(
        "CC",
        Some(reaction(
            vec![molecule("C", None), molecule("C", None)],
            "C.C>>CC",
        )),
    );
    Route {
        target: molecule(
            "CCC",
            Some(reaction(
                vec![intermediate, molecule("C", None)],
                "CC.C>>CCC",
            )),
        ),
        annotations: Map::from_iter([(
            "source".to_owned(),
            Value::String("curation-contract".to_owned()),
        )]),
        schema_version: Default::default(),
    }
}

fn molecule(smiles: &str, product_of: Option<Reaction>) -> Molecule {
    let (smiles, inchikey) = chem::normalize(smiles).unwrap();
    Molecule {
        smiles,
        inchikey,
        product_of: product_of.map(Box::new),
        annotations: Map::new(),
    }
}

fn reaction(reactants: Vec<Molecule>, mapped_smiles: &str) -> Reaction {
    Reaction {
        reactants,
        mapped_reaction_smiles: Some(ReactionSmiles::try_from(mapped_smiles).unwrap()),
        template: None,
        reagents: None,
        solvents: None,
        annotations: Map::new(),
    }
}

fn inchikey(smiles: &str) -> String {
    chem::normalize(smiles).unwrap().1.into_string()
}

fn aizynth_raw_route() -> Value {
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

fn paroutes_raw_route() -> Value {
    json!({
        "type": "mol",
        "smiles": "CCO",
        "children": [{
            "type": "reaction",
            "smiles": "CCO",
            "metadata": {"ID": "US123;1", "rsmi": "C.CC>>CCO"},
            "children": [
                {"type": "mol", "smiles": "C", "in_stock": true, "children": []},
                {
                    "type": "mol",
                    "smiles": "CC",
                    "children": [{
                        "type": "reaction",
                        "smiles": "CC",
                        "metadata": {"ID": "US123;2", "rsmi": "C>>CC"},
                        "children": [
                            {"type": "mol", "smiles": "C", "in_stock": true, "children": []}
                        ]
                    }]
                }
            ]
        }]
    })
}
