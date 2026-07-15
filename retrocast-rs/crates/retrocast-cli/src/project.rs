use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, bail};
use retrocast_core::{
    adapt, adapters, analyze,
    io::{read_json, read_stock, write_json},
    model::{AnalysisReport, Evaluation, ExecutionStats, Predictions, Task},
    provenance::{ContentType, ManifestOutput, create_manifest},
    route::AdaptMode,
    score::{self, Stocks},
};
use serde_json::{Map, Value, json};

const DEFAULT_DATA_DIR: &str = "data/retrocast";

#[derive(Clone, Debug)]
pub struct ProjectPaths {
    pub data_dir: PathBuf,
    pub benchmarks: PathBuf,
    pub stocks: PathBuf,
    pub raw: PathBuf,
    pub processed: PathBuf,
    pub scored: PathBuf,
    pub results: PathBuf,
}

impl ProjectPaths {
    pub fn resolve(cli_data_dir: Option<PathBuf>, config_path: &Path) -> anyhow::Result<Self> {
        let config_data_dir = read_config_data_dir(config_path)?;
        let data_dir = cli_data_dir
            .or_else(|| env::var_os("RETROCAST_DATA_DIR").map(PathBuf::from))
            .or(config_data_dir)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DATA_DIR));
        Ok(Self {
            benchmarks: data_dir.join("1-benchmarks/definitions"),
            stocks: data_dir.join("1-benchmarks/stocks"),
            raw: data_dir.join("2-raw"),
            processed: data_dir.join("3-processed"),
            scored: data_dir.join("4-scored"),
            results: data_dir.join("5-results"),
            data_dir,
        })
    }

    pub fn print_config(&self) {
        println!("RetroCast Schema V2 Configuration");
        println!("data_dir\t{}", self.data_dir.display());
        for (name, path) in self.entries() {
            let state = if path.exists() { "exists" } else { "missing" };
            println!("{name}\t{}\t{state}", path.display());
        }
    }

    pub fn list_raw_manifests(&self) -> anyhow::Result<()> {
        println!("model\tbenchmark\tadapter\traw_file");
        if !self.raw.exists() {
            return Ok(());
        }
        for model_dir in sorted_directories(&self.raw)? {
            for benchmark_dir in sorted_directories(&model_dir)? {
                let manifest_path = benchmark_dir.join("manifest.json");
                if !manifest_path.is_file() {
                    continue;
                }
                let model = safe_name(&file_name(&model_dir)?, "model")?;
                let benchmark = safe_name(&file_name(&benchmark_dir)?, "benchmark")?;
                let adapter = manifest_directive(&manifest_path, "adapter")?.unwrap_or_default();
                let raw_file = manifest_directive(&manifest_path, "raw_results_filename")?
                    .unwrap_or_else(|| "results.json.gz".to_owned());
                println!("{model}\t{benchmark}\t{adapter}\t{raw_file}");
            }
        }
        Ok(())
    }

    fn entries(&self) -> [(&str, &Path); 5] {
        [
            ("benchmarks", &self.benchmarks),
            ("raw", &self.raw),
            ("processed", &self.processed),
            ("scored", &self.scored),
            ("results", &self.results),
        ]
    }
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub model: Option<String>,
    pub all_models: bool,
    pub dataset: Option<String>,
    pub all_datasets: bool,
}

#[derive(Clone, Debug)]
pub struct ProjectIngestOptions<'a> {
    pub adapter: Option<&'a str>,
    pub mode: AdaptMode,
    pub max_candidates: Option<usize>,
    pub workers: usize,
}

#[derive(Clone, Debug)]
pub struct ProjectScoreOptions<'a> {
    pub match_level: &'a str,
    pub acceptable_route_match: &'a str,
    pub workers: usize,
}

#[derive(Clone, Debug)]
pub struct ProjectAnalyzeOptions<'a> {
    pub stock: Option<&'a str>,
    pub ks: &'a [usize],
    pub prefix_depths: &'a [usize],
    pub n_boot: usize,
    pub seed: u64,
    pub workers: usize,
}

pub fn ingest_project(
    paths: &ProjectPaths,
    selection: &Selection,
    options: &ProjectIngestOptions<'_>,
) -> anyhow::Result<()> {
    let models = resolve_models(paths, selection, Stage::Ingest)?;
    let datasets = resolve_datasets(paths, selection)?;
    for model in models {
        for dataset in &datasets {
            let raw_dir = paths.raw.join(&model).join(dataset);
            if !raw_dir.is_dir() {
                eprintln!("skipping {model}/{dataset}: raw directory is missing");
                continue;
            }
            let manifest_path = raw_dir.join("manifest.json");
            let adapter_name = options
                .adapter
                .map(str::to_owned)
                .or(manifest_directive(&manifest_path, "adapter")?)
                .with_context(|| {
                    format!("{model}/{dataset} has no adapter directive or --adapter")
                })?;
            let raw_filename = manifest_directive(&manifest_path, "raw_results_filename")?
                .unwrap_or_else(|| "results.json.gz".to_owned());
            safe_name(&raw_filename, "raw results filename")?;
            let raw_path = raw_dir.join(raw_filename);
            let task_path = paths.benchmarks.join(format!("{dataset}.json.gz"));
            let raw: Value = read_json(&raw_path)?;
            let task: Task = read_json(&task_path)?;
            let adapter = adapters::built_in(&adapter_name)
                .with_context(|| format!("unknown RetroCast adapter {adapter_name:?}"))?;
            let predictions = adapt::ingest(
                raw,
                adapter.as_ref(),
                &task,
                options.mode,
                options.max_candidates,
                options.workers,
            )?;
            let output = paths
                .processed
                .join(dataset)
                .join(&model)
                .join("candidates.json.gz");
            write_json(&output, &predictions)?;
            write_stage_manifest(
                "ingest:v2",
                &[raw_path, task_path],
                &output,
                &predictions,
                ContentType::Predictions,
                &paths.data_dir,
                json!({
                    "model": model,
                    "benchmark": dataset,
                    "adapter": adapter_name,
                    "mode": options.mode.as_str(),
                    "max_candidates": options.max_candidates,
                    "workers": options.workers,
                }),
                json!({
                    "targets": predictions.len(),
                    "candidates": candidate_count(&predictions),
                }),
            )?;
            println!("ingested {model}/{dataset} -> {}", output.display());
        }
    }
    Ok(())
}

pub fn score_project(
    paths: &ProjectPaths,
    selection: &Selection,
    options: &ProjectScoreOptions<'_>,
) -> anyhow::Result<()> {
    let models = resolve_models(paths, selection, Stage::Score)?;
    let datasets = resolve_datasets(paths, selection)?;
    for model in models {
        for dataset in &datasets {
            let task_path = paths.benchmarks.join(format!("{dataset}.json.gz"));
            let candidates_path = paths
                .processed
                .join(dataset)
                .join(&model)
                .join("candidates.json.gz");
            if !candidates_path.is_file() {
                eprintln!("skipping {model}/{dataset}: processed candidates are missing");
                continue;
            }
            let task: Task = read_json(&task_path)?;
            let predictions: Predictions = read_json(&candidates_path)?;
            let (stocks, stock_paths) = load_task_stocks(&task, paths)?;
            let execution_stats_path = paths
                .raw
                .join(&model)
                .join(dataset)
                .join("execution_stats.json.gz");
            let execution_stats: Option<ExecutionStats> = if execution_stats_path.is_file() {
                Some(read_json(&execution_stats_path)?)
            } else {
                None
            };
            let evaluation = score::score(
                &predictions,
                &task,
                &stocks,
                options.match_level,
                options.acceptable_route_match,
                execution_stats.as_ref(),
                options.workers,
            )?;
            let label = task.derived_metric_label();
            let output = paths
                .scored
                .join(dataset)
                .join(&model)
                .join(&label)
                .join("evaluation.json.gz");
            write_json(&output, &evaluation)?;
            let mut sources = vec![task_path, candidates_path];
            sources.extend(stock_paths);
            if execution_stats_path.is_file() {
                sources.push(execution_stats_path);
            }
            write_stage_manifest(
                "score:v2",
                &sources,
                &output,
                &evaluation,
                ContentType::Unknown,
                &paths.data_dir,
                json!({
                    "model": model,
                    "benchmark": dataset,
                    "stock": label,
                    "match_level": options.match_level,
                    "acceptable_route_match": options.acceptable_route_match,
                    "workers": options.workers,
                }),
                json!({
                    "targets": evaluation.targets.len(),
                    "candidates": evaluation.targets.values().map(|target| target.candidates.len()).sum::<usize>(),
                }),
            )?;
            println!("scored {model}/{dataset}/{label} -> {}", output.display());
        }
    }
    Ok(())
}

pub fn analyze_project(
    paths: &ProjectPaths,
    selection: &Selection,
    options: &ProjectAnalyzeOptions<'_>,
) -> anyhow::Result<()> {
    let models = resolve_models(paths, selection, Stage::Analyze)?;
    let datasets = resolve_datasets(paths, selection)?;
    for model in models {
        for dataset in &datasets {
            let scored_dir = paths.scored.join(dataset).join(&model);
            if !scored_dir.is_dir() {
                eprintln!("skipping {model}/{dataset}: scored evaluations are missing");
                continue;
            }
            let labels = if let Some(stock) = options.stock {
                vec![safe_name(stock, "stock")?]
            } else {
                sorted_directories(&scored_dir)?
                    .into_iter()
                    .map(|path| file_name(&path))
                    .collect::<anyhow::Result<Vec<_>>>()?
            };
            for label in labels {
                let evaluation_path = scored_dir.join(&label).join("evaluation.json.gz");
                if !evaluation_path.is_file() {
                    eprintln!("skipping {model}/{dataset}/{label}: evaluation is missing");
                    continue;
                }
                let mut evaluation: Evaluation = read_json(&evaluation_path)?;
                let execution_stats_path = paths
                    .raw
                    .join(&model)
                    .join(dataset)
                    .join("execution_stats.json.gz");
                if execution_stats_path.is_file() {
                    let stats: ExecutionStats = read_json(&execution_stats_path)?;
                    merge_execution_stats(&mut evaluation, &stats);
                }
                let report = analyze::analyze(
                    &evaluation,
                    options.ks,
                    options.prefix_depths,
                    options.n_boot,
                    options.seed,
                    options.workers,
                )?;
                let output_dir = paths.results.join(dataset).join(&model).join(&label);
                let analysis_path = output_dir.join("analysis.json.gz");
                let markdown_path = output_dir.join("report.md");
                write_json(&analysis_path, &report)?;
                write_markdown_report(&markdown_path, &model, dataset, &label, &report)?;
                let mut sources = vec![evaluation_path];
                if execution_stats_path.is_file() {
                    sources.push(execution_stats_path);
                }
                write_stage_manifest_many(
                    "analyze:v2",
                    &sources,
                    &[
                        (
                            &analysis_path,
                            serde_json::to_value(&report)?,
                            ContentType::Unknown,
                        ),
                        (&markdown_path, Value::Null, ContentType::Unknown),
                    ],
                    &paths.data_dir,
                    json!({
                        "model": model,
                        "benchmark": dataset,
                        "stock": label,
                        "top_k": options.ks,
                        "prefix_depth": options.prefix_depths,
                        "n_boot": options.n_boot,
                        "seed": options.seed,
                        "workers": options.workers,
                    }),
                    json!({
                        "n_metrics": report.metrics.len(),
                        "n_strata": report.by_stratum.len(),
                    }),
                )?;
                println!(
                    "analyzed {model}/{dataset}/{label} -> {}",
                    analysis_path.display()
                );
            }
        }
    }
    Ok(())
}

pub fn write_sidecar_manifest<T: serde::Serialize>(
    action: &str,
    sources: &[PathBuf],
    output: &Path,
    value: &T,
    content_type: ContentType,
    parameters: Value,
    statistics: Value,
) -> anyhow::Result<()> {
    let root = output.parent().unwrap_or_else(|| Path::new("."));
    let manifest = create_manifest(
        action,
        sources,
        &[ManifestOutput {
            label: None,
            path: output.to_owned(),
            value: serde_json::to_value(value)?,
            content_type,
            content_hash: None,
        }],
        root,
        object(parameters)?,
        object(statistics)?,
        Map::new(),
        Map::new(),
        None,
        false,
    )?;
    write_json(&sidecar_manifest_path(output), &manifest)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum Stage {
    Ingest,
    Score,
    Analyze,
}

fn resolve_models(
    paths: &ProjectPaths,
    selection: &Selection,
    stage: Stage,
) -> anyhow::Result<Vec<String>> {
    if let Some(model) = &selection.model {
        return Ok(vec![safe_name(model, "model")?]);
    }
    if !selection.all_models {
        bail!("project mode requires --model or --all-models");
    }
    let base = match stage {
        Stage::Ingest => &paths.raw,
        Stage::Score => &paths.processed,
        Stage::Analyze => &paths.scored,
    };
    if !base.is_dir() {
        return Ok(Vec::new());
    }
    if matches!(stage, Stage::Ingest) {
        return sorted_directories(base)?
            .into_iter()
            .map(|path| safe_name(&file_name(&path)?, "model"))
            .collect();
    }
    let mut models = BTreeSet::new();
    for dataset_dir in sorted_directories(base)? {
        for model_dir in sorted_directories(&dataset_dir)? {
            models.insert(safe_name(&file_name(&model_dir)?, "model")?);
        }
    }
    Ok(models.into_iter().collect())
}

fn resolve_datasets(paths: &ProjectPaths, selection: &Selection) -> anyhow::Result<Vec<String>> {
    if let Some(dataset) = &selection.dataset {
        return Ok(vec![safe_name(dataset, "dataset")?]);
    }
    if !selection.all_datasets {
        bail!("project mode requires --dataset or --all-datasets");
    }
    if !paths.benchmarks.is_dir() {
        return Ok(Vec::new());
    }
    let mut datasets = Vec::new();
    for entry in std::fs::read_dir(&paths.benchmarks)? {
        let path = entry?.path();
        let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(dataset) = filename.strip_suffix(".json.gz") {
            datasets.push(safe_name(dataset, "dataset")?);
        }
    }
    datasets.sort();
    Ok(datasets)
}

fn load_task_stocks(task: &Task, paths: &ProjectPaths) -> anyhow::Result<(Stocks, Vec<PathBuf>)> {
    let names: BTreeSet<String> = task
        .targets
        .keys()
        .flat_map(|target_id| task.effective_constraints(target_id))
        .filter(|constraint| constraint.kind == "retrocast.stock_termination")
        .filter_map(|constraint| constraint.fields.get("stock")?.as_str().map(str::to_owned))
        .collect();
    let mut stocks = BTreeMap::new();
    let mut sources = Vec::new();
    for name in names {
        safe_name(&name, "stock")?;
        let path = paths.stocks.join(format!("{name}.csv.gz"));
        stocks.extend(read_stock(&path, &name)?);
        sources.push(path);
    }
    Ok((stocks, sources))
}

fn merge_execution_stats(evaluation: &mut Evaluation, stats: &ExecutionStats) {
    for (target_id, target) in &mut evaluation.targets {
        if target.wall_time.is_none() {
            target.wall_time = stats.wall_time.get(target_id).copied();
        }
        if target.cpu_time.is_none() {
            target.cpu_time = stats.cpu_time.get(target_id).copied();
        }
    }
}

fn write_markdown_report(
    path: &Path,
    model: &str,
    dataset: &str,
    stock: &str,
    report: &AnalysisReport,
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = format!(
        "# Evaluation Report: {model} / {dataset} / {stock}\n\n| Metric | Value | Count | 95% CI |\n| --- | ---: | ---: | --- |\n"
    );
    for (name, summary) in &report.metrics {
        let interval = match (summary.ci_low, summary.ci_high) {
            (Some(low), Some(high)) => format!("{low:.6}–{high:.6}"),
            _ => "—".to_owned(),
        };
        text.push_str(&format!(
            "| `{name}` | {:.6} | {} | {interval} |\n",
            summary.value, summary.count
        ));
    }
    std::fs::write(path, text)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_stage_manifest<T: serde::Serialize>(
    action: &str,
    sources: &[PathBuf],
    output: &Path,
    value: &T,
    content_type: ContentType,
    root: &Path,
    parameters: Value,
    statistics: Value,
) -> anyhow::Result<()> {
    write_stage_manifest_many(
        action,
        sources,
        &[(output, serde_json::to_value(value)?, content_type)],
        root,
        parameters,
        statistics,
    )
}

fn write_stage_manifest_many(
    action: &str,
    sources: &[PathBuf],
    outputs: &[(&Path, Value, ContentType)],
    root: &Path,
    parameters: Value,
    statistics: Value,
) -> anyhow::Result<()> {
    let outputs = outputs
        .iter()
        .map(|(path, value, content_type)| ManifestOutput {
            label: None,
            path: path.to_path_buf(),
            value: value.clone(),
            content_type: *content_type,
            content_hash: None,
        })
        .collect::<Vec<_>>();
    let manifest = create_manifest(
        action,
        sources,
        &outputs,
        root,
        object(parameters)?,
        object(statistics)?,
        Map::new(),
        Map::new(),
        None,
        false,
    )?;
    let manifest_path = outputs
        .first()
        .and_then(|output| output.path.parent())
        .context("stage manifest requires at least one output")?
        .join("manifest.json");
    write_json(&manifest_path, &manifest)?;
    Ok(())
}

fn object(value: Value) -> anyhow::Result<Map<String, Value>> {
    value
        .as_object()
        .cloned()
        .context("manifest metadata must be a JSON object")
}

fn candidate_count(predictions: &Predictions) -> usize {
    predictions.values().map(Vec::len).sum()
}

fn manifest_directive(path: &Path, key: &str) -> anyhow::Result<Option<String>> {
    if !path.is_file() {
        return Ok(None);
    }
    let manifest: Value = read_json(path)?;
    Ok(manifest
        .get("directives")
        .and_then(|directives| directives.get(key))
        .and_then(Value::as_str)
        .map(str::to_owned))
}

fn read_config_data_dir(path: &Path) -> anyhow::Result<Option<PathBuf>> {
    if !path.is_file() {
        return Ok(None);
    }
    let payload = std::fs::read_to_string(path)?;
    let config: Value = serde_yaml::from_str(&payload)
        .with_context(|| format!("failed to parse config file {}", path.display()))?;
    Ok(config
        .get("data_dir")
        .and_then(Value::as_str)
        .map(PathBuf::from))
}

fn sorted_directories(path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut directories = std::fs::read_dir(path)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    directories.sort();
    Ok(directories)
}

fn file_name(path: &Path) -> anyhow::Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .with_context(|| format!("path has no UTF-8 filename: {}", path.display()))
}

fn safe_name(value: &str, label: &str) -> anyhow::Result<String> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || value.contains('\0')
    {
        bail!("unsafe {label}: {value:?}");
    }
    Ok(value.to_owned())
}

fn sidecar_manifest_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    for suffix in [".jsonl.gz", ".json.gz", ".csv.gz", ".txt.gz"] {
        if let Some(stem) = name.strip_suffix(suffix) {
            return output.with_file_name(format!("{stem}.manifest.json"));
        }
    }
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(name);
    output.with_file_name(format!("{stem}.manifest.json"))
}
