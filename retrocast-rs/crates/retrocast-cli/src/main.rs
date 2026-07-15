use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use retrocast_core::{
    adapt::ingest,
    adapters,
    analyze::analyze,
    dataset::{
        HostedDataRequest, TrainingDataRequest, download_hosted_data, download_training_data,
    },
    io::{read_json, read_stock, write_json},
    model::{Candidate, Evaluation, ExecutionStats, Predictions, Task},
    pipeline::{PipelineOptions, bind_stock_constraint, run_pipeline},
    provenance::{ContentType, verify_manifest},
    route::AdaptMode,
    score::score,
};
use serde_json::{Value, json};

mod project;

use project::{
    ProjectAnalyzeOptions, ProjectIngestOptions, ProjectPaths, ProjectScoreOptions, Selection,
    analyze_project, ingest_project, score_project, write_sidecar_manifest,
};

#[derive(Parser)]
#[command(
    name = "retrocast",
    version,
    about = "Ingest, score, and analyze retrosynthesis plans"
)]
struct Cli {
    /// Path to the project configuration file.
    #[arg(long, global = true, default_value = "retrocast-config.yaml")]
    config: PathBuf,
    /// Override the project data directory (or use RETROCAST_DATA_DIR).
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show the resolved project data directory and stage paths.
    Config,
    /// List raw model manifests in the project data directory.
    List,
    /// List the built-in planner adapters available in this executable.
    ListAdapters,
    /// Download published RetroCast benchmarks, stocks, or workflow artifacts.
    GetData {
        target: String,
        #[arg(long = "dir")]
        output_dir: Option<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long, default_value = "https://files.ischemist.com/retrocast/data")]
        base_url: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// Download published PaRoutes training releases.
    GetTrainingData {
        artifact_or_release: Option<String>,
        #[arg(long)]
        split: Option<String>,
        #[arg(long)]
        format: Option<String>,
        #[arg(long, value_delimiter = ',')]
        omit: Vec<String>,
        #[arg(long, default_value = "latest")]
        release: String,
        #[arg(long, default_value = "paroutes")]
        dataset: String,
        #[arg(long = "dir")]
        output_dir: Option<PathBuf>,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(
            long,
            default_value = "https://files.ischemist.com/retrocast/training-sets"
        )]
        base_url: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// Adapt one planner payload into ranked canonical candidates.
    Adapt {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        adapter: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "strict")]
        mode: String,
        #[arg(long)]
        max_candidates: Option<usize>,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Group adapted candidates by benchmark target identity.
    Collect {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        benchmark: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Adapt and collect planner outputs against a benchmark task.
    Ingest {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        adapter: Option<String>,
        #[arg(long)]
        benchmark: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        all_models: bool,
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long)]
        all_datasets: bool,
        #[arg(long, default_value = "strict")]
        mode: String,
        #[arg(long)]
        max_candidates: Option<usize>,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Score collected candidates against validity and task constraints.
    #[command(alias = "score-file")]
    Score {
        #[arg(long, visible_alias = "routes")]
        candidates: Option<PathBuf>,
        #[arg(long)]
        benchmark: Option<PathBuf>,
        #[arg(long)]
        stock: Option<PathBuf>,
        #[arg(long)]
        stock_name: Option<String>,
        #[arg(long)]
        model_name: Option<String>,
        #[arg(long)]
        ignore_stereo: bool,
        #[arg(long)]
        execution_stats: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        all_models: bool,
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long)]
        all_datasets: bool,
        #[arg(long, default_value = "full")]
        match_level: String,
        #[arg(long, default_value = "prefix")]
        acceptable_route_match: String,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Aggregate a scored evaluation into metrics and bootstrap intervals.
    Analyze {
        #[arg(long)]
        evaluation: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        all_models: bool,
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long)]
        all_datasets: bool,
        #[arg(long)]
        stock: Option<String>,
        #[arg(long, value_delimiter = ',', default_value = "1,3,5,10,20,50,100")]
        ks: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
        prefix_depths: Vec<usize>,
        #[arg(long, default_value_t = 10_000)]
        n_boot: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Verify artifact hashes and, optionally, the complete provenance chain.
    Verify {
        #[arg(long, alias = "target")]
        manifest: Option<PathBuf>,
        #[arg(long)]
        root_dir: Option<PathBuf>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        deep: bool,
        #[arg(long)]
        output_only: bool,
        #[arg(long)]
        strict: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run ingest, score, and analysis in one process.
    Pipeline {
        #[arg(long)]
        raw: PathBuf,
        #[arg(long)]
        benchmark: PathBuf,
        #[arg(long)]
        stock: PathBuf,
        #[arg(long)]
        stock_name: Option<String>,
        #[arg(long)]
        execution_stats: Option<PathBuf>,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value = "aizynthfinder")]
        adapter: String,
        #[arg(long, default_value_t = 1)]
        workers: usize,
        #[arg(long, default_value = "strict")]
        mode: String,
        #[arg(long)]
        max_candidates: Option<usize>,
        #[arg(long, default_value = "full")]
        match_level: String,
        #[arg(long, default_value = "prefix")]
        acceptable_route_match: String,
        #[arg(long, value_delimiter = ',', default_value = "1,3,5,10,20,50,100")]
        ks: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
        prefix_depths: Vec<usize>,
        #[arg(long, default_value_t = 10_000)]
        n_boot: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config;
    let data_dir = cli.data_dir;
    match cli.command {
        Command::Config => ProjectPaths::resolve(data_dir, &config_path)?.print_config(),
        Command::List => ProjectPaths::resolve(data_dir, &config_path)?.list_raw_manifests()?,
        Command::ListAdapters => {
            for adapter in adapters::BUILT_IN_ADAPTERS {
                println!("{adapter}");
            }
            for (alias, canonical) in adapters::DEPRECATED_ADAPTER_ALIASES {
                println!("{alias} -> {canonical} (deprecated alias)");
            }
        }
        Command::GetData {
            target,
            output_dir,
            cache_dir,
            base_url,
            dry_run,
        } => {
            let paths = download_hosted_data(&HostedDataRequest {
                target,
                cache_dir,
                output_dir,
                base_url,
                dry_run,
            })?;
            for path in paths {
                println!("{}", path.display());
            }
        }
        Command::GetTrainingData {
            artifact_or_release,
            split,
            format,
            omit,
            mut release,
            dataset,
            output_dir,
            cache_dir,
            base_url,
            dry_run,
        } => {
            let artifact = match artifact_or_release {
                Some(value) if is_date_release(&value) => {
                    release = value;
                    None
                }
                value => value,
            };
            if artifact.is_none() && release == "latest" {
                anyhow::bail!("specify a training artifact or an explicit release");
            }
            let paths = download_training_data(&TrainingDataRequest {
                dataset,
                artifact,
                split,
                release,
                format,
                omit,
                cache_dir,
                output_dir,
                base_url,
                dry_run,
            })?;
            for path in paths {
                println!("{}", path.display());
            }
        }
        Command::Adapt {
            input,
            adapter,
            output,
            mode,
            max_candidates,
            workers,
        } => {
            let raw: Value = read_json(&input)?;
            let mode = AdaptMode::parse(&mode)?;
            let resolved = resolve_adapter(&adapter)?;
            let candidates = adapters::adapt_candidates_with_workers(
                raw,
                resolved.as_ref(),
                mode,
                None,
                None,
                max_candidates,
                workers,
            )?;
            write_json(&output, &candidates)?;
            write_sidecar_manifest(
                "[cli]adapt",
                &[input],
                &output,
                &candidates,
                ContentType::Unknown,
                json!({
                    "adapter": adapter,
                    "mode": mode.as_str(),
                    "max_candidates": max_candidates,
                    "workers": workers,
                }),
                json!({"candidates": candidates.len()}),
            )?;
        }
        Command::Collect {
            input,
            benchmark,
            output,
        } => {
            let candidates: Vec<Candidate> = read_json(&input)?;
            let task: Task = read_json(&benchmark)?;
            let predictions = retrocast_core::adapt::collect_candidates(candidates, &task);
            write_json(&output, &predictions)?;
            write_sidecar_manifest(
                "[cli]collect",
                &[input, benchmark],
                &output,
                &predictions,
                ContentType::Predictions,
                json!({}),
                json!({
                    "targets": predictions.len(),
                    "candidates": predictions.values().map(Vec::len).sum::<usize>(),
                }),
            )?;
        }
        Command::Ingest {
            input,
            adapter,
            benchmark,
            output,
            model,
            all_models,
            dataset,
            all_datasets,
            mode,
            max_candidates,
            workers,
        } => {
            let mode = AdaptMode::parse(&mode)?;
            match (input, benchmark, output) {
                (Some(input), Some(benchmark), Some(output)) => {
                    let adapter = adapter.context("file-mode ingest requires --adapter")?;
                    let raw: Value = read_json(&input)?;
                    let task: Task = read_json(&benchmark)?;
                    let resolved = resolve_adapter(&adapter)?;
                    let predictions =
                        ingest(raw, resolved.as_ref(), &task, mode, max_candidates, workers)?;
                    write_json(&output, &predictions)?;
                    write_sidecar_manifest(
                        "[cli]ingest",
                        &[input, benchmark],
                        &output,
                        &predictions,
                        ContentType::Predictions,
                        json!({
                            "adapter": adapter,
                            "mode": mode.as_str(),
                            "max_candidates": max_candidates,
                            "workers": workers,
                        }),
                        json!({
                            "targets": predictions.len(),
                            "candidates": predictions.values().map(Vec::len).sum::<usize>(),
                        }),
                    )?;
                }
                (None, None, None) => ingest_project(
                    &ProjectPaths::resolve(data_dir, &config_path)?,
                    &Selection {
                        model,
                        all_models,
                        dataset,
                        all_datasets,
                    },
                    &ProjectIngestOptions {
                        adapter: adapter.as_deref(),
                        mode,
                        max_candidates,
                        workers,
                    },
                )?,
                _ => anyhow::bail!(
                    "ingest requires all of --input/--benchmark/--output for file mode, or none for project mode"
                ),
            }
        }
        Command::Score {
            candidates,
            benchmark,
            stock,
            stock_name,
            model_name,
            ignore_stereo,
            execution_stats,
            output,
            model,
            all_models,
            dataset,
            all_datasets,
            match_level,
            acceptable_route_match,
            workers,
        } => match (candidates, benchmark, stock, output) {
            (Some(candidates), Some(benchmark), Some(stock), Some(output)) => {
                let predictions: Predictions = read_json(&candidates)?;
                let mut task: Task = read_json(&benchmark)?;
                let stock_name = stock_name.unwrap_or_else(|| infer_stock_name(&stock));
                bind_stock_constraint(&mut task, &stock_name);
                let stocks = read_stock(&stock, &stock_name)?;
                let stats: Option<ExecutionStats> =
                    execution_stats.as_deref().map(read_json).transpose()?;
                let match_level = if ignore_stereo {
                    "no_stereo"
                } else {
                    &match_level
                };
                let evaluation = score(
                    &predictions,
                    &task,
                    &stocks,
                    match_level,
                    &acceptable_route_match,
                    stats.as_ref(),
                    workers,
                )?;
                write_json(&output, &evaluation)?;
                let mut sources = vec![benchmark, candidates, stock];
                if let Some(execution_stats) = execution_stats {
                    sources.push(execution_stats);
                }
                write_sidecar_manifest(
                    "[cli]score-file",
                    &sources,
                    &output,
                    &evaluation,
                    ContentType::Unknown,
                    json!({
                        "stock": stock_name,
                        "model_name": model_name,
                        "ignore_stereo": ignore_stereo,
                        "acceptable_route_match": acceptable_route_match,
                        "workers": workers,
                    }),
                    json!({
                        "targets": evaluation.targets.len(),
                        "candidates": evaluation.targets.values().map(|target| target.candidates.len()).sum::<usize>(),
                    }),
                )?;
            }
            (None, None, None, None) => {
                if stock_name.is_some() || model_name.is_some() || execution_stats.is_some() {
                    anyhow::bail!(
                        "--stock-name, --model-name, and --execution-stats are file-mode score options"
                    );
                }
                let match_level = if ignore_stereo {
                    "no_stereo"
                } else {
                    &match_level
                };
                score_project(
                    &ProjectPaths::resolve(data_dir, &config_path)?,
                    &Selection {
                        model,
                        all_models,
                        dataset,
                        all_datasets,
                    },
                    &ProjectScoreOptions {
                        match_level,
                        acceptable_route_match: &acceptable_route_match,
                        workers,
                    },
                )?;
            }
            _ => anyhow::bail!(
                "score requires all of --candidates/--benchmark/--stock/--output for file mode, or none for project mode"
            ),
        },
        Command::Analyze {
            evaluation,
            output,
            model,
            all_models,
            dataset,
            all_datasets,
            stock,
            ks,
            prefix_depths,
            n_boot,
            seed,
            workers,
        } => match (evaluation, output) {
            (Some(evaluation_path), Some(output)) => {
                let evaluation: Evaluation = read_json(&evaluation_path)?;
                let report = analyze(&evaluation, &ks, &prefix_depths, n_boot, seed, workers)?;
                write_json(&output, &report)?;
                write_sidecar_manifest(
                    "[cli]analyze",
                    &[evaluation_path],
                    &output,
                    &report,
                    ContentType::Unknown,
                    json!({
                        "top_k": ks,
                        "prefix_depth": prefix_depths,
                        "n_boot": n_boot,
                        "seed": seed,
                        "workers": workers,
                    }),
                    json!({
                        "n_metrics": report.metrics.len(),
                        "n_strata": report.by_stratum.len(),
                    }),
                )?;
            }
            (None, None) => analyze_project(
                &ProjectPaths::resolve(data_dir, &config_path)?,
                &Selection {
                    model,
                    all_models,
                    dataset,
                    all_datasets,
                },
                &ProjectAnalyzeOptions {
                    stock: stock.as_deref(),
                    ks: &ks,
                    prefix_depths: &prefix_depths,
                    n_boot,
                    seed,
                    workers,
                },
            )?,
            _ => anyhow::bail!(
                "analyze requires both --evaluation and --output for file mode, or neither for project mode"
            ),
        },
        Command::Verify {
            manifest,
            root_dir,
            all,
            deep,
            output_only,
            strict,
            output,
        } => {
            let paths = ProjectPaths::resolve(data_dir, &config_path)?;
            let root_dir = root_dir.unwrap_or_else(|| paths.data_dir.clone());
            let manifests = if all {
                find_manifests(&paths.data_dir)?
            } else {
                vec![manifest.context("verify requires --manifest/--target or --all")?]
            };
            let reports = manifests
                .iter()
                .map(|manifest| verify_manifest(manifest, &root_dir, deep, output_only, !strict))
                .collect::<Vec<_>>();
            if let Some(output) = output {
                write_json(&output, &reports)?;
            } else {
                println!("{}", serde_json::to_string_pretty(&reports)?);
            }
            if reports.iter().any(|report| !report.is_valid) {
                anyhow::bail!("manifest verification failed");
            }
        }
        Command::Pipeline {
            raw,
            benchmark,
            stock,
            stock_name,
            execution_stats,
            output_dir,
            adapter,
            workers,
            mode,
            max_candidates,
            match_level,
            acceptable_route_match,
            ks,
            prefix_depths,
            n_boot,
            seed,
        } => {
            let mode = AdaptMode::parse(&mode)?;
            let stats = run_pipeline(
                &raw,
                &benchmark,
                &stock,
                stock_name.as_deref(),
                execution_stats.as_deref(),
                &output_dir,
                &PipelineOptions {
                    adapter: &adapter,
                    mode,
                    max_candidates,
                    workers,
                    match_level: &match_level,
                    acceptable_route_match: &acceptable_route_match,
                    ks: &ks,
                    prefix_depths: &prefix_depths,
                    n_boot,
                    seed,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
    }
    Ok(())
}

fn resolve_adapter(name: &str) -> anyhow::Result<Box<dyn adapters::Adapter>> {
    adapters::built_in(name).with_context(|| format!("unknown RetroCast adapter {name:?}"))
}

fn infer_stock_name(path: &std::path::Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("stock")
        .trim_end_matches(".csv.gz")
        .trim_end_matches(".txt.gz")
        .trim_end_matches(".txt")
        .to_owned()
}

fn is_date_release(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 11
        && bytes[0] == b'v'
        && bytes[5] == b'-'
        && bytes[8] == b'-'
        && bytes[1..5].iter().all(u8::is_ascii_digit)
        && bytes[6..8].iter().all(u8::is_ascii_digit)
        && bytes[9..11].iter().all(u8::is_ascii_digit)
}

fn find_manifests(root: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    fn visit(path: &std::path::Path, manifests: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if !path.is_dir() {
            return Ok(());
        }
        for entry in std::fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_dir() {
                visit(&path, manifests)?;
            } else if path.file_name().is_some_and(|name| name == "manifest.json") {
                manifests.push(path);
            }
        }
        Ok(())
    }

    let mut manifests = Vec::new();
    visit(root, &mut manifests)?;
    manifests.sort();
    Ok(manifests)
}

#[cfg(test)]
mod tests {
    use super::is_date_release;

    #[test]
    fn recognizes_only_canonical_date_release_syntax() {
        assert!(is_date_release("v2026-07-14"));
        assert!(!is_date_release("version-0714"));
        assert!(!is_date_release("v2026-7-14"));
        assert!(!is_date_release("v2026-07-14-extra"));
    }
}
