use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use flate2::{Compression, write::GzEncoder};
use serde::Serialize;
use serde_json::Value;

use crate::{
    adapt::process_file_targets,
    adapters,
    analyze::{analyze_prepared_targets, prepare_target_analysis},
    error::{EngineError, Result},
    io::{read_json, read_stock, write_json},
    model::{ExecutionStats, Task},
    provenance::{ContentType, ManifestOutput, create_manifest},
    route::AdaptMode,
    schema::SchemaVersion,
    score::TargetScorer,
};

#[cfg(test)]
use crate::{adapt::ingest_file, analyze::analyze, score::score_owned};

#[derive(Debug, Serialize)]
pub struct EvaluationRunStats {
    pub engine: &'static str,
    pub workers: usize,
    pub targets: usize,
    pub candidates: usize,
    pub ingest_seconds: f64,
    pub score_seconds: f64,
    pub analyze_seconds: f64,
    pub total_seconds: f64,
    pub targets_per_second: f64,
    pub candidates_per_second: f64,
}

pub struct EvaluationOptions<'a> {
    pub adapter: &'a str,
    pub mode: AdaptMode,
    pub max_candidates: Option<usize>,
    pub workers: usize,
    pub match_level: &'a str,
    pub acceptable_route_match: &'a str,
    pub ks: &'a [usize],
    pub prefix_depths: &'a [usize],
    pub n_boot: usize,
    pub seed: u64,
}

#[cfg(test)]
fn evaluate_materialized_files(
    raw_path: &Path,
    benchmark_path: &Path,
    stock_path: &Path,
    stock_name: Option<&str>,
    execution_stats_path: Option<&Path>,
    output_dir: &Path,
    options: &EvaluationOptions<'_>,
) -> Result<EvaluationRunStats> {
    let started = Instant::now();
    let raw_path = resolve_raw_path(raw_path)?;
    let mut task: Task = read_json(benchmark_path)?;
    let stock_name = stock_name.map(str::to_owned).unwrap_or_else(|| {
        stock_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("stock")
            .trim_end_matches(".csv.gz")
            .trim_end_matches(".txt.gz")
            .trim_end_matches(".txt")
            .to_owned()
    });
    bind_stock_constraint(&mut task, &stock_name);
    let stats: Option<ExecutionStats> = execution_stats_path.map(read_json).transpose()?;

    let ingest_started = Instant::now();
    let adapter = adapters::built_in(options.adapter).ok_or_else(|| crate::error::EngineError::UnknownAdapter {
        name: options.adapter.to_owned(),
        available: "aizynthfinder, askcos, directmultistep, dreamretroer, molbuilder, multistepttl, paroutes, retrochimera, retrostar, synllama, synplanner, syntheseus, ursa".to_owned(),
    })?;
    let predictions = ingest_file(
        &raw_path,
        adapter.as_ref(),
        &task,
        options.mode,
        options.max_candidates,
        options.workers,
    )?;
    let ingest_seconds = ingest_started.elapsed().as_secs_f64();
    let targets = task.targets.len();
    let candidates = predictions.values().map(Vec::len).sum();
    write_json(&output_dir.join("candidates.json.gz"), &predictions)?;

    let score_started = Instant::now();
    let stocks = read_stock(stock_path, &stock_name)?;
    let evaluation = score_owned(
        predictions,
        task,
        &stocks,
        options.match_level,
        options.acceptable_route_match,
        stats.as_ref(),
        options.workers,
    )?;
    let score_seconds = score_started.elapsed().as_secs_f64();
    write_json(&output_dir.join("evaluation.json.gz"), &evaluation)?;

    let analyze_started = Instant::now();
    let report = analyze(
        &evaluation,
        options.ks,
        options.prefix_depths,
        options.n_boot,
        options.seed,
        options.workers,
    )?;
    let analyze_seconds = analyze_started.elapsed().as_secs_f64();
    drop(evaluation);
    write_json(&output_dir.join("analysis.json.gz"), &report)?;

    let total_seconds = started.elapsed().as_secs_f64();
    let stats = EvaluationRunStats {
        engine: "rust",
        workers: options.workers,
        targets,
        candidates,
        ingest_seconds,
        score_seconds,
        analyze_seconds,
        total_seconds,
        targets_per_second: targets as f64 / total_seconds,
        candidates_per_second: candidates as f64 / total_seconds,
    };
    write_json(&output_dir.join("evaluation-run.json"), &stats)?;
    write_evaluation_manifest(
        &raw_path,
        benchmark_path,
        stock_path,
        execution_stats_path,
        output_dir,
        &stats,
        options,
    )?;
    Ok(stats)
}

/// Evaluate planner output without retaining corpus-wide route graphs.
///
/// Each worker adapts and scores one target, writes its candidate and
/// evaluation output fragments, and keeps only compact analysis contributions.
/// Sorted fragment assembly preserves artifact ordering without using JSON as
/// transport between scoring and analysis.
pub fn evaluate_files(
    raw_path: &Path,
    benchmark_path: &Path,
    stock_path: &Path,
    stock_name: Option<&str>,
    execution_stats_path: Option<&Path>,
    output_dir: &Path,
    options: &EvaluationOptions<'_>,
) -> Result<EvaluationRunStats> {
    let started = Instant::now();
    let raw_path = resolve_raw_path(raw_path)?;
    let mut task: Task = read_json(benchmark_path)?;
    let stock_name = stock_name.map(str::to_owned).unwrap_or_else(|| {
        stock_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("stock")
            .trim_end_matches(".csv.gz")
            .trim_end_matches(".txt.gz")
            .trim_end_matches(".txt")
            .to_owned()
    });
    bind_stock_constraint(&mut task, &stock_name);
    let execution_stats: Option<ExecutionStats> =
        execution_stats_path.map(read_json).transpose()?;
    let stocks = read_stock(stock_path, &stock_name)?;
    let scorer = TargetScorer::new(&stocks, options.match_level, options.acceptable_route_match);
    let adapter = adapters::built_in(options.adapter).ok_or_else(|| {
        crate::error::EngineError::UnknownAdapter {
            name: options.adapter.to_owned(),
            available: "aizynthfinder, askcos, directmultistep, dreamretroer, molbuilder, multistepttl, paroutes, retrochimera, retrostar, synllama, synplanner, syntheseus, ursa".to_owned(),
        }
    })?;

    let staging = EvaluationStaging::create(output_dir)?;
    let candidate_fragments = staging.path.join("candidates");
    let evaluation_fragments = staging.path.join("evaluation");
    std::fs::create_dir_all(&candidate_fragments)?;
    std::fs::create_dir_all(&evaluation_fragments)?;
    let target_indices: BTreeMap<_, _> = task
        .targets
        .keys()
        .enumerate()
        .map(|(index, target_id)| (target_id.clone(), index))
        .collect();
    let metric_label = task.derived_metric_label();
    let tiers = [0];

    let ingest_started = Instant::now();
    let processed = process_file_targets(
        &raw_path,
        adapter.as_ref(),
        &task,
        options.mode,
        options.max_candidates,
        options.workers,
        |target_id, target, candidates| {
            let index = target_indices[target_id];
            write_json(&fragment_path(&candidate_fragments, index), &candidates)?;
            let candidate_count = candidates.len();
            let constraints = task.effective_constraints(target_id);
            let result = scorer.score_owned(
                target,
                constraints,
                candidates,
                execution_stats
                    .as_ref()
                    .and_then(|stats| stats.wall_time.get(target_id).copied()),
                execution_stats
                    .as_ref()
                    .and_then(|stats| stats.cpu_time.get(target_id).copied()),
            )?;
            write_json(&fragment_path(&evaluation_fragments, index), &result)?;
            let prepared = prepare_target_analysis(
                &result,
                &tiers,
                options.ks,
                options.prefix_depths,
                &metric_label,
                options.match_level,
            );
            Ok((candidate_count, prepared))
        },
    )?;
    let ingest_seconds = ingest_started.elapsed().as_secs_f64();

    let score_started = Instant::now();
    let target_ids = task.targets.keys().cloned().collect::<Vec<_>>();
    write_fragment_map(
        &output_dir.join("candidates.json.gz"),
        &target_ids,
        &candidate_fragments,
    )?;
    let candidates = processed.values().map(|(count, _)| *count).sum();
    write_evaluation_fragments(
        &output_dir.join("evaluation.json.gz"),
        &task,
        &metric_label,
        options.match_level,
        options.acceptable_route_match,
        &target_ids,
        &evaluation_fragments,
    )?;
    let score_seconds = score_started.elapsed().as_secs_f64();

    let analyze_started = Instant::now();
    let prepared = processed
        .into_values()
        .map(|(_, prepared)| prepared)
        .collect::<Vec<_>>();
    let report =
        analyze_prepared_targets(&prepared, options.n_boot, options.seed, options.workers)?;
    let analyze_seconds = analyze_started.elapsed().as_secs_f64();
    write_json(&output_dir.join("analysis.json.gz"), &report)?;

    let targets = target_ids.len();
    let total_seconds = started.elapsed().as_secs_f64();
    let stats = EvaluationRunStats {
        engine: "rust",
        workers: options.workers,
        targets,
        candidates,
        ingest_seconds,
        score_seconds,
        analyze_seconds,
        total_seconds,
        targets_per_second: targets as f64 / total_seconds,
        candidates_per_second: candidates as f64 / total_seconds,
    };
    write_json(&output_dir.join("evaluation-run.json"), &stats)?;
    write_evaluation_manifest(
        &raw_path,
        benchmark_path,
        stock_path,
        execution_stats_path,
        output_dir,
        &stats,
        options,
    )?;
    Ok(stats)
}

fn fragment_path(directory: &Path, index: usize) -> PathBuf {
    directory.join(format!("{index:08}.json"))
}

fn write_fragment_map(path: &Path, target_ids: &[String], fragments: &Path) -> Result<()> {
    let mut writer = gzip_json_writer(path)?;
    write_fragment_map_body(&mut writer, target_ids, fragments)?;
    writer.finish()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_evaluation_fragments(
    path: &Path,
    task: &Task,
    metric_label: &str,
    match_level: &str,
    acceptable_route_match: &str,
    target_ids: &[String],
    fragments: &Path,
) -> Result<()> {
    let mut writer = gzip_json_writer(path)?;
    writer.write_all(b"{\"task\":")?;
    serde_json::to_writer(&mut writer, task)?;
    writer.write_all(b",\"tiers\":[0],\"metric_label\":")?;
    serde_json::to_writer(&mut writer, metric_label)?;
    writer.write_all(b",\"acceptable_match_level\":")?;
    serde_json::to_writer(&mut writer, match_level)?;
    writer.write_all(b",\"acceptable_route_match\":")?;
    serde_json::to_writer(&mut writer, acceptable_route_match)?;
    writer.write_all(b",\"targets\":")?;
    write_fragment_map_body(&mut writer, target_ids, fragments)?;
    writer.write_all(b",\"schema_version\":")?;
    serde_json::to_writer(&mut writer, &SchemaVersion::V2)?;
    writer.write_all(b"}")?;
    writer.finish()?;
    Ok(())
}

fn write_fragment_map_body(
    writer: &mut impl Write,
    target_ids: &[String],
    fragments: &Path,
) -> Result<()> {
    writer.write_all(b"{")?;
    for (index, target_id) in target_ids.iter().enumerate() {
        if index != 0 {
            writer.write_all(b",")?;
        }
        serde_json::to_writer(&mut *writer, target_id)?;
        writer.write_all(b":")?;
        std::io::copy(&mut File::open(fragment_path(fragments, index))?, writer)?;
    }
    writer.write_all(b"}")?;
    Ok(())
}

fn gzip_json_writer(path: &Path) -> Result<GzEncoder<BufWriter<File>>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(GzEncoder::new(
        BufWriter::new(File::create(path)?),
        Compression::default(),
    ))
}

struct EvaluationStaging {
    path: PathBuf,
}

impl EvaluationStaging {
    fn create(output_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(output_dir)?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before Unix epoch")
            .as_nanos();
        let path = output_dir.join(format!(
            ".retrocast-evaluate-staging-{}-{nonce}",
            std::process::id(),
        ));
        std::fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for EvaluationStaging {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn resolve_raw_path(path: &Path) -> Result<PathBuf> {
    if !path.is_dir() {
        return Ok(path.to_owned());
    }
    let manifest_path = path.join("manifest.json");
    let filename = if manifest_path.is_file() {
        let manifest: Value = read_json(&manifest_path)?;
        manifest
            .get("directives")
            .and_then(|directives| directives.get("raw_results_filename"))
            .and_then(Value::as_str)
            .unwrap_or("results.json.gz")
            .to_owned()
    } else {
        "results.json.gz".to_owned()
    };
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename.contains('/')
        || filename.contains('\\')
        || filename.contains('\0')
    {
        return Err(EngineError::AdapterSchema(format!(
            "unsafe raw_results_filename directive {filename:?}"
        )));
    }
    Ok(path.join(filename))
}

#[allow(clippy::too_many_arguments)]
fn write_evaluation_manifest(
    raw_path: &Path,
    benchmark_path: &Path,
    stock_path: &Path,
    execution_stats_path: Option<&Path>,
    output_dir: &Path,
    stats: &EvaluationRunStats,
    options: &EvaluationOptions<'_>,
) -> Result<()> {
    let candidates_path = output_dir.join("candidates.json.gz");
    let evaluation_path = output_dir.join("evaluation.json.gz");
    let analysis_path = output_dir.join("analysis.json.gz");
    let stats_path = output_dir.join("evaluation-run.json");
    let outputs = [
        ManifestOutput {
            label: None,
            path: candidates_path,
            value: Value::Null,
            content_type: ContentType::Unknown,
            content_hash: None,
        },
        ManifestOutput {
            label: None,
            path: evaluation_path,
            value: Value::Null,
            content_type: ContentType::Unknown,
            content_hash: None,
        },
        ManifestOutput {
            label: None,
            path: analysis_path,
            value: Value::Null,
            content_type: ContentType::Unknown,
            content_hash: None,
        },
        ManifestOutput {
            label: None,
            path: stats_path,
            value: serde_json::to_value(stats)?,
            content_type: ContentType::Unknown,
            content_hash: None,
        },
    ];
    let mut sources = vec![
        raw_path.to_owned(),
        benchmark_path.to_owned(),
        stock_path.to_owned(),
    ];
    if let Some(path) = execution_stats_path {
        sources.push(path.to_owned());
    }
    let manifest = create_manifest(
        "evaluate:v2",
        &sources,
        &outputs,
        output_dir,
        serde_json::Map::from_iter([
            (
                "adapter".to_owned(),
                Value::String(options.adapter.to_owned()),
            ),
            (
                "mode".to_owned(),
                Value::String(options.mode.as_str().to_owned()),
            ),
            ("workers".to_owned(), Value::from(options.workers)),
            (
                "match_level".to_owned(),
                Value::String(options.match_level.to_owned()),
            ),
            (
                "acceptable_route_match".to_owned(),
                Value::String(options.acceptable_route_match.to_owned()),
            ),
            ("n_boot".to_owned(), Value::from(options.n_boot)),
            ("seed".to_owned(), Value::from(options.seed)),
        ]),
        serde_json::Map::from_iter([
            ("targets".to_owned(), Value::from(stats.targets)),
            ("candidates".to_owned(), Value::from(stats.candidates)),
        ]),
        serde_json::Map::new(),
        serde_json::Map::new(),
        None,
        false,
    )?;
    write_json(&output_dir.join("manifest.json"), &manifest)
}

pub fn bind_stock_constraint(task: &mut Task, stock_name: &str) {
    task.default_constraints
        .retain(|constraint| constraint.kind != "retrocast.stock_termination");
    task.default_constraints.push(crate::model::Constraint {
        kind: "retrocast.stock_termination".to_owned(),
        fields: serde_json::Map::from_iter([(
            "stock".to_owned(),
            Value::String(stock_name.to_owned()),
        )]),
    });
    for constraints in task.constraints.values_mut() {
        constraints.retain(|constraint| constraint.kind != "retrocast.stock_termination");
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{EvaluationOptions, evaluate_files, evaluate_materialized_files};
    use crate::{
        io::{read_json, write_json},
        route::AdaptMode,
    };

    #[test]
    fn evaluate_files_matches_materialized_artifacts() {
        let temporary = tempfile::tempdir().unwrap();
        let raw_path = temporary.path().join("raw.json.gz");
        let benchmark_path = temporary.path().join("benchmark.json");
        let stock_path = temporary.path().join("stock.txt");
        write_json(
            &raw_path,
            &json!({
                "methanol": [{"type": "mol", "smiles": "OC"}],
                "ethanol": [{"type": "mol", "smiles": "OCC"}]
            }),
        )
        .unwrap();
        write_json(
            &benchmark_path,
            &json!({
                "name": "evaluation-test",
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
            }),
        )
        .unwrap();
        std::fs::write(&stock_path, "CCO\nCO\n").unwrap();
        let options = EvaluationOptions {
            adapter: "aizynthfinder",
            mode: AdaptMode::Strict,
            max_candidates: None,
            workers: 2,
            match_level: "full",
            acceptable_route_match: "prefix",
            ks: &[1, 3],
            prefix_depths: &[1, 2],
            n_boot: 100,
            seed: 42,
        };
        let materialized = temporary.path().join("materialized");
        let optimized = temporary.path().join("optimized");

        evaluate_materialized_files(
            &raw_path,
            &benchmark_path,
            &stock_path,
            Some("stock"),
            None,
            &materialized,
            &options,
        )
        .unwrap();
        evaluate_files(
            &raw_path,
            &benchmark_path,
            &stock_path,
            Some("stock"),
            None,
            &optimized,
            &options,
        )
        .unwrap();

        for artifact in [
            "candidates.json.gz",
            "evaluation.json.gz",
            "analysis.json.gz",
        ] {
            assert_eq!(
                read_json::<serde_json::Value>(&materialized.join(artifact)).unwrap(),
                read_json::<serde_json::Value>(&optimized.join(artifact)).unwrap(),
                "{artifact} differs",
            );
        }
        for artifact in ["candidates.json.gz", "analysis.json.gz"] {
            assert_eq!(
                std::fs::read(materialized.join(artifact)).unwrap(),
                std::fs::read(optimized.join(artifact)).unwrap(),
                "compressed {artifact} differs",
            );
        }
        assert!(std::fs::read_dir(&optimized).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with('.')
        }));
    }
}
