use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use serde::Serialize;
use serde_json::Value;

use crate::{
    adapt::ingest_file,
    adapters,
    analyze::analyze,
    error::{EngineError, Result},
    io::{read_json, read_stock, write_json},
    model::{ExecutionStats, Task},
    provenance::{ContentType, ManifestOutput, create_manifest},
    route::AdaptMode,
    score::score_owned,
};

#[derive(Debug, Serialize)]
pub struct PipelineStats {
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

pub struct PipelineOptions<'a> {
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

pub fn run_pipeline(
    raw_path: &Path,
    benchmark_path: &Path,
    stock_path: &Path,
    stock_name: Option<&str>,
    execution_stats_path: Option<&Path>,
    output_dir: &Path,
    options: &PipelineOptions<'_>,
) -> Result<PipelineStats> {
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
    let stats = PipelineStats {
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
    write_json(&output_dir.join("pipeline-stats.json"), &stats)?;
    write_pipeline_manifest(
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
fn write_pipeline_manifest(
    raw_path: &Path,
    benchmark_path: &Path,
    stock_path: &Path,
    execution_stats_path: Option<&Path>,
    output_dir: &Path,
    stats: &PipelineStats,
    options: &PipelineOptions<'_>,
) -> Result<()> {
    let candidates_path = output_dir.join("candidates.json.gz");
    let evaluation_path = output_dir.join("evaluation.json.gz");
    let analysis_path = output_dir.join("analysis.json.gz");
    let stats_path = output_dir.join("pipeline-stats.json");
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
        "pipeline:v2",
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
