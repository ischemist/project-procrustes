use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyValueError},
    prelude::*,
};
use retrocast_core::{
    adapt, adapters, analyze,
    chem::{self, InchiKeyLevel as ChemInchiKeyLevel},
    embedding::{EmbeddingOptions, find_route_embeddings, route_embeds_at},
    io,
    model::{Evaluation, ExecutionStats, Predictions, Task},
    pipeline::{PipelineOptions, run_pipeline},
    route::AdaptMode,
    route_path::RoutePath,
    route_view::InchiKeyLevel,
    score,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Opaque ownership of ingested candidates while a Python pipeline remains native.
///
/// Python can materialize the value for inspection, but scoring can consume this
/// handle directly without serializing and reparsing the candidate graph.
#[pyclass(frozen, module = "retrocast._native")]
struct NativePredictions {
    value: Arc<Predictions>,
}

#[pymethods]
impl NativePredictions {
    fn json(&self) -> PyResult<String> {
        to_json(self.value.as_ref())
    }

    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let value = Arc::clone(&self.value);
        py.detach(move || io::write_json(&path, value.as_ref()))
            .map_err(python_error)
    }
}

/// Opaque ownership of a scored evaluation for direct native analysis.
#[pyclass(frozen, module = "retrocast._native")]
struct NativeEvaluation {
    value: Arc<Evaluation>,
}

#[pymethods]
impl NativeEvaluation {
    fn json(&self) -> PyResult<String> {
        to_json(self.value.as_ref())
    }

    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let value = Arc::clone(&self.value);
        py.detach(move || io::write_json(&path, value.as_ref()))
            .map_err(python_error)
    }

    fn metric_label(&self) -> String {
        self.value.metric_label.clone()
    }
}

#[derive(serde::Serialize)]
struct AdapterBoundaryResult<T: serde::Serialize> {
    value: Option<T>,
    failure: Option<retrocast_core::model::FailureRecord>,
}

impl<T: serde::Serialize> AdapterBoundaryResult<T> {
    fn success(value: T) -> Self {
        Self {
            value: Some(value),
            failure: None,
        }
    }

    fn failure(
        adapter: &str,
        error: retrocast_core::error::EngineError,
        target: Option<&retrocast_core::model::Target>,
    ) -> Self {
        Self {
            value: None,
            failure: Some(adapters::boundary_failure(adapter, error, target)),
        }
    }
}

fn python_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn artifact_read_error(error: retrocast_core::error::EngineError) -> PyErr {
    match error {
        retrocast_core::error::EngineError::Json(error) if error.is_data() => {
            PyValueError::new_err(error.to_string())
        }
        retrocast_core::error::EngineError::Io(error) => PyOSError::new_err(error.to_string()),
        error => python_error(error),
    }
}

fn chemistry_error(error: retrocast_core::error::EngineError) -> PyErr {
    match error {
        retrocast_core::error::EngineError::InvalidSmiles { .. }
        | retrocast_core::error::EngineError::InvalidInchiKey { .. }
        | retrocast_core::error::EngineError::InchiKeyUpscale { .. } => {
            PyValueError::new_err(error.to_string())
        }
        _ => python_error(error),
    }
}

fn from_json<T: serde::de::DeserializeOwned>(value: &str) -> PyResult<T> {
    serde_json::from_str(value).map_err(python_error)
}

fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (raw_json, adapter, mode="strict", target_json=None, source_key=None, max_candidates=None, workers=1))]
#[allow(clippy::too_many_arguments)]
fn adapt_candidates_json(
    py: Python<'_>,
    raw_json: &str,
    adapter: &str,
    mode: &str,
    target_json: Option<&str>,
    source_key: Option<&str>,
    max_candidates: Option<usize>,
    workers: usize,
) -> PyResult<String> {
    let raw: Value = from_json(raw_json)?;
    let target: Option<retrocast_core::model::Target> = target_json.map(from_json).transpose()?;
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let resolved = adapters::built_in(adapter)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RetroCast adapter {adapter:?}")))?;
    let result = py
        .detach(|| {
            adapters::adapt_candidates_with_workers(
                raw,
                resolved.as_ref(),
                mode,
                target.as_ref(),
                source_key,
                max_candidates,
                workers,
            )
        })
        .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (raw_json, adapter, mode="strict", target_json=None))]
fn adapt_route_json(
    py: Python<'_>,
    raw_json: &str,
    adapter: &str,
    mode: &str,
    target_json: Option<&str>,
) -> PyResult<String> {
    let raw: Value = from_json(raw_json)?;
    let target: Option<retrocast_core::model::Target> = target_json.map(from_json).transpose()?;
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let resolved = adapters::built_in(adapter)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RetroCast adapter {adapter:?}")))?;
    let route = py
        .detach(|| resolved.cast(raw, mode, target.as_ref()))
        .map_err(python_error)?;
    to_json(&route)
}

#[pyfunction]
#[pyo3(signature = (raw_json, adapter, source_key=None))]
fn adapter_entries_json(
    py: Python<'_>,
    raw_json: &str,
    adapter: &str,
    source_key: Option<&str>,
) -> PyResult<String> {
    let raw: Value = from_json(raw_json)?;
    let resolved = adapters::built_in(adapter)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RetroCast adapter {adapter:?}")))?;
    let result = match py.detach(|| resolved.entries(raw, source_key)) {
        Ok(entries) => AdapterBoundaryResult::success(entries),
        Err(error) => AdapterBoundaryResult::failure(adapter, error, None),
    };
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (raw_json, adapter, mode="strict", target_json=None))]
fn adapter_cast_result_json(
    py: Python<'_>,
    raw_json: &str,
    adapter: &str,
    mode: &str,
    target_json: Option<&str>,
) -> PyResult<String> {
    let raw: Value = from_json(raw_json)?;
    let target: Option<retrocast_core::model::Target> = target_json.map(from_json).transpose()?;
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let resolved = adapters::built_in(adapter)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RetroCast adapter {adapter:?}")))?;
    let result = match py.detach(|| resolved.cast(raw, mode, target.as_ref())) {
        Ok(route) => AdapterBoundaryResult::success(route),
        Err(error) => AdapterBoundaryResult::failure(adapter, error, target.as_ref()),
    };
    to_json(&result)
}

#[pyfunction]
fn dms_route_length_json(raw_json: &str) -> PyResult<usize> {
    let raw: Value = from_json(raw_json)?;
    adapters::dms_route_length(&raw).map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (synthesis_string, mode="strict"))]
fn synllama_precursor_map_json(synthesis_string: &str, mode: &str) -> PyResult<String> {
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let result = match adapters::parse_synllama_synthesis(synthesis_string, mode) {
        Ok(precursor_map) => AdapterBoundaryResult::success(precursor_map),
        Err(error) => AdapterBoundaryResult::failure("synllama", error, None),
    };
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (route_string, adapter, mode="strict"))]
fn reaction_string_parse_json(route_string: &str, adapter: &str, mode: &str) -> PyResult<String> {
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let result = match adapters::parse_reaction_string(route_string, mode, adapter) {
        Ok(parsed) => AdapterBoundaryResult::success(parsed),
        Err(error) => AdapterBoundaryResult::failure(adapter, error, None),
    };
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (raw_json, source_key=None))]
fn synplanner_entries_json(raw_json: &str, source_key: Option<&str>) -> PyResult<String> {
    let raw: Value = from_json(raw_json)?;
    let result = match adapters::extract_synplanner_entries(raw, source_key) {
        Ok(batch) => AdapterBoundaryResult::success(batch),
        Err(error) => AdapterBoundaryResult::failure("synplanner", error, None),
    };
    to_json(&result)
}

#[pyfunction]
fn paroutes_condition_stats_json(route_json: &str, statistics_json: &str) -> PyResult<String> {
    let route: Value = from_json(route_json)?;
    let mut statistics: adapters::ConditionSlotParseStatistics = from_json(statistics_json)?;
    adapters::analyze_condition_slots(&route, &mut statistics);
    to_json(&statistics)
}

#[pyfunction]
fn candidate_statistics_json(candidates_json: &str) -> PyResult<String> {
    let candidates: Vec<retrocast_core::model::Candidate> = from_json(candidates_json)?;
    to_json(&retrocast_core::stats::candidate_statistics(&candidates))
}

#[pyfunction]
fn collected_candidate_statistics_json(predictions_json: &str) -> PyResult<String> {
    let predictions: Predictions = from_json(predictions_json)?;
    to_json(&retrocast_core::stats::collected_candidate_statistics(
        &predictions,
    ))
}

#[pyfunction]
fn collected_candidate_statistics_native(
    predictions: PyRef<'_, NativePredictions>,
) -> PyResult<String> {
    to_json(&retrocast_core::stats::collected_candidate_statistics(
        predictions.value.as_ref(),
    ))
}

#[pyfunction]
fn candidate_run_manifest_json(statistics_json: &str) -> PyResult<String> {
    let statistics: retrocast_core::stats::CandidateRunStatistics = from_json(statistics_json)?;
    to_json(&statistics.manifest())
}

#[pyfunction]
fn evaluation_statistics_json(evaluation_json: &str) -> PyResult<String> {
    let evaluation: Evaluation = from_json(evaluation_json)?;
    to_json(&retrocast_core::stats::evaluation_statistics(&evaluation))
}

#[pyfunction]
fn evaluation_statistics_native(evaluation: PyRef<'_, NativeEvaluation>) -> PyResult<String> {
    to_json(&retrocast_core::stats::evaluation_statistics(
        evaluation.value.as_ref(),
    ))
}

#[pyfunction]
fn bootstrap_distribution_json(values_json: &str, n_boot: usize, seed: i64) -> PyResult<String> {
    let values: Vec<f64> = from_json(values_json)?;
    to_json(&retrocast_core::stats::bootstrap_distribution(
        &values,
        n_boot,
        seed as u64,
    ))
}

#[pyfunction]
fn reliability_flag_json(n: usize, probability: f64) -> PyResult<String> {
    to_json(&retrocast_core::stats::reliability_flag(n, probability))
}

#[pyfunction]
#[pyo3(signature = (values_json, n_boot, seed, alpha=0.05, reliability=true))]
fn summarize_values_json(
    values_json: &str,
    n_boot: usize,
    seed: i64,
    alpha: f64,
    reliability: bool,
) -> PyResult<String> {
    if n_boot == 0 {
        return Err(PyValueError::new_err("n_boot must be positive"));
    }
    let values: Vec<f64> = from_json(values_json)?;
    to_json(&retrocast_core::stats::summarize_values(
        &values,
        n_boot,
        seed as u64,
        alpha,
        reliability,
    ))
}

#[pyfunction]
fn probabilistic_ranking_json(
    model_values_json: &str,
    n_boot: usize,
    seed: i64,
) -> PyResult<String> {
    let model_values: BTreeMap<String, Vec<f64>> = from_json(model_values_json)?;
    to_json(&retrocast_core::stats::probabilistic_ranking(
        &model_values,
        n_boot,
        seed as u64,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn paired_difference_json(
    values_a_json: &str,
    values_b_json: &str,
    model_a: &str,
    model_b: &str,
    metric: &str,
    n_boot: usize,
    seed: i64,
) -> PyResult<String> {
    let values_a: Vec<f64> = from_json(values_a_json)?;
    let values_b: Vec<f64> = from_json(values_b_json)?;
    if values_a.is_empty() || values_a.len() != values_b.len() {
        return Err(PyValueError::new_err(
            "paired samples must be non-empty and have equal length",
        ));
    }
    if n_boot == 0 {
        return Err(PyValueError::new_err("n_boot must be positive"));
    }
    to_json(&retrocast_core::stats::paired_difference(
        &values_a,
        &values_b,
        model_a,
        model_b,
        metric,
        n_boot,
        seed as u64,
    ))
}

#[pyfunction]
#[pyo3(signature = (targets_json, tiers_json, ks_json, prefix_depths_json, metric_label="task", match_level="full", n_boot=10000, seed=42, workers=1))]
#[allow(clippy::too_many_arguments)]
fn summarize_targets_json(
    py: Python<'_>,
    targets_json: &str,
    tiers_json: &str,
    ks_json: &str,
    prefix_depths_json: &str,
    metric_label: &str,
    match_level: &str,
    n_boot: usize,
    seed: i64,
    workers: usize,
) -> PyResult<String> {
    let targets: Vec<retrocast_core::model::TargetResult> = from_json(targets_json)?;
    let tiers: Vec<u8> = from_json(tiers_json)?;
    let ks: Vec<usize> = from_json(ks_json)?;
    let prefix_depths: Vec<usize> = from_json(prefix_depths_json)?;
    let result = py
        .detach(|| {
            retrocast_core::analyze::summarize_target_results(
                &targets,
                &tiers,
                &ks,
                &prefix_depths,
                metric_label,
                match_level,
                n_boot,
                seed as u64,
                workers,
            )
        })
        .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (route_json, constraints_json, stocks_json, match_level="full"))]
fn check_task_constraints_json(
    route_json: &str,
    constraints_json: &str,
    stocks_json: &str,
    match_level: &str,
) -> PyResult<String> {
    let route: retrocast_core::model::Route = from_json(route_json)?;
    let constraints: Vec<retrocast_core::model::Constraint> = from_json(constraints_json)?;
    let stocks: retrocast_core::score::Stocks = from_json(stocks_json)?;
    let result =
        retrocast_core::score::check_task_constraints(&route, &constraints, &stocks, match_level)
            .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (target_json, metric, k=None))]
fn target_metric_json(target_json: &str, metric: &str, k: Option<usize>) -> PyResult<f64> {
    let target: retrocast_core::model::TargetResult = from_json(target_json)?;
    match metric {
        "is_solvable" => Ok(retrocast_core::stats::target_is_solvable(&target)),
        "top_k" => Ok(retrocast_core::stats::target_top_k(
            &target,
            k.ok_or_else(|| PyValueError::new_err("top_k metric requires k"))?,
        )),
        _ => Err(PyValueError::new_err(format!(
            "unknown target metric {metric:?}"
        ))),
    }
}

#[pyfunction]
fn collect_candidates_json(candidates_json: &str, task_json: &str) -> PyResult<String> {
    let candidates: Vec<retrocast_core::model::Candidate> = from_json(candidates_json)?;
    let task: Task = from_json(task_json)?;
    to_json(&retrocast_core::adapt::collect_candidates(
        candidates, &task,
    ))
}

#[pyfunction]
fn collect_routes_json(routes_json: &str, task_json: &str) -> PyResult<String> {
    let routes: Vec<retrocast_core::model::Route> = from_json(routes_json)?;
    let task: Task = from_json(task_json)?;
    to_json(&retrocast_core::adapt::collect_routes(routes, &task))
}

#[pyfunction]
fn route_path_parse(value: &str) -> PyResult<(String, Vec<usize>)> {
    let path = RoutePath::parse(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let kind = if path.is_molecule() { "m" } else { "r" };
    Ok((kind.to_owned(), path.indices().to_vec()))
}

#[pyfunction]
#[pyo3(signature = (value, operation, index=None))]
fn route_path_transform(value: &str, operation: &str, index: Option<i64>) -> PyResult<String> {
    let path = RoutePath::parse(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let transformed = match operation {
        "produced_by" => path.produced_by(),
        "product" => path.product(),
        "reactant" => path.reactant(
            usize::try_from(
                index.ok_or_else(|| PyValueError::new_err("reactant operation requires index"))?,
            )
            .map_err(|_| PyValueError::new_err("reactant index must be non-negative"))?,
        ),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown route path operation {operation:?}"
            )));
        }
    }
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(transformed.to_string())
}

#[pyfunction]
fn reactant_order_json(reactants_json: &str) -> PyResult<String> {
    let reactants: Vec<retrocast_core::model::Molecule> = from_json(reactants_json)?;
    to_json(&retrocast_core::route::reactant_order(&reactants))
}

#[pyfunction]
#[pyo3(signature = (route_json, node_kind, level="full", path=None, depth=None, fields_json="[]"))]
#[allow(clippy::too_many_arguments)]
fn route_identity_json(
    route_json: &str,
    node_kind: &str,
    level: &str,
    path: Option<&str>,
    depth: Option<i64>,
    fields_json: &str,
) -> PyResult<String> {
    let route: retrocast_core::model::Route = from_json(route_json)?;
    let level: InchiKeyLevel = serde_json::from_value(serde_json::json!(level))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let depth = depth
        .map(|value| {
            usize::try_from(value).map_err(|_| PyValueError::new_err("depth must be non-negative"))
        })
        .transpose()?;
    let field_names: Vec<String> = from_json(fields_json)?;
    let mut unknown = Vec::new();
    let fields = field_names
        .iter()
        .filter_map(
            |field| match retrocast_core::route_view::ReactionContentField::parse(field) {
                Some(field) => Some(field),
                None => {
                    unknown.push(field.clone());
                    None
                }
            },
        )
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        unknown.sort();
        return Err(PyValueError::new_err(format!(
            "unknown reaction content fields: {}",
            unknown.join(", ")
        )));
    }
    let has_content = !fields.is_empty();
    let (key, signature) = match node_kind {
        "route" => {
            let key = if has_content {
                route.content_key(level, &fields, depth)
            } else {
                route.key(level, depth)
            };
            let signature = if has_content {
                route.content_signature(level, &fields, depth)
            } else {
                route.signature(level, depth)
            };
            (key, signature)
        }
        "molecule" => {
            let path = RoutePath::parse(
                path.ok_or_else(|| PyValueError::new_err("molecule identity requires path"))?,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let view = route
                .molecule_at(&path)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let key = if has_content {
                view.content_subtree_key(level, &fields, depth)
            } else {
                view.subtree_key(level, depth)
            };
            let signature = if has_content {
                view.content_subtree_signature(level, &fields, depth)
            } else {
                view.subtree_signature(level, depth)
            };
            (key, signature)
        }
        "reaction" => {
            let path = RoutePath::parse(
                path.ok_or_else(|| PyValueError::new_err("reaction identity requires path"))?,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let view = route
                .reaction_at(&path)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let key = if has_content {
                view.content_key(level, &fields)
            } else {
                view.key(level)
            };
            let signature = if has_content {
                view.content_signature(level, &fields)
            } else {
                view.signature(level)
            };
            (key, signature)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown route identity node kind {node_kind:?}"
            )));
        }
    };
    to_json(&serde_json::json!({"key": key, "signature": signature}))
}

#[pyfunction]
#[pyo3(signature = (route_json, molecule_path=None))]
fn route_structure_json(route_json: &str, molecule_path: Option<&str>) -> PyResult<String> {
    let route: retrocast_core::model::Route = from_json(route_json)?;
    let (molecule_paths, leaf_paths, depth) = if let Some(path) = molecule_path {
        let path =
            RoutePath::parse(path).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let view = route
            .molecule_at(&path)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        (
            view.molecules()
                .into_iter()
                .map(|node| node.path.to_string())
                .collect::<Vec<_>>(),
            view.leaves()
                .into_iter()
                .map(|node| node.path.to_string())
                .collect::<Vec<_>>(),
            view.depth(),
        )
    } else {
        (
            route
                .molecules()
                .into_iter()
                .map(|node| node.path.to_string())
                .collect::<Vec<_>>(),
            route
                .leaves()
                .into_iter()
                .map(|node| node.path.to_string())
                .collect::<Vec<_>>(),
            route.depth(),
        )
    };
    let reaction_paths = route
        .reactions()
        .into_iter()
        .map(|node| node.path.to_string())
        .collect::<Vec<_>>();
    to_json(&serde_json::json!({
        "molecule_paths": molecule_paths,
        "leaf_paths": leaf_paths,
        "reaction_paths": reaction_paths,
        "depth": depth,
        "is_convergent": route.is_convergent(),
    }))
}

#[pyfunction]
#[pyo3(signature = (raw_json, adapter, task_json, mode="strict", max_candidates=None, workers=1))]
fn ingest_json(
    py: Python<'_>,
    raw_json: &str,
    adapter: &str,
    task_json: &str,
    mode: &str,
    max_candidates: Option<usize>,
    workers: usize,
) -> PyResult<String> {
    let raw: Value = from_json(raw_json)?;
    let task: Task = from_json(task_json)?;
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let resolved = adapters::built_in(adapter)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RetroCast adapter {adapter:?}")))?;
    let result = py
        .detach(|| adapt::ingest(raw, resolved.as_ref(), &task, mode, max_candidates, workers))
        .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (raw_json, adapter, task_json, mode="strict", max_candidates=None, workers=1))]
fn ingest_native(
    py: Python<'_>,
    raw_json: &str,
    adapter: &str,
    task_json: &str,
    mode: &str,
    max_candidates: Option<usize>,
    workers: usize,
) -> PyResult<NativePredictions> {
    let raw: Value = from_json(raw_json)?;
    let task: Task = from_json(task_json)?;
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let resolved = adapters::built_in(adapter)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RetroCast adapter {adapter:?}")))?;
    let value = py
        .detach(|| adapt::ingest(raw, resolved.as_ref(), &task, mode, max_candidates, workers))
        .map_err(python_error)?;
    Ok(NativePredictions {
        value: Arc::new(value),
    })
}

#[pyfunction]
fn load_predictions_native(py: Python<'_>, path: PathBuf) -> PyResult<NativePredictions> {
    let value = py
        .detach(move || io::read_json::<Predictions>(&path))
        .map_err(artifact_read_error)?;
    Ok(NativePredictions {
        value: Arc::new(value),
    })
}

#[pyfunction]
fn load_evaluation_native(py: Python<'_>, path: PathBuf) -> PyResult<NativeEvaluation> {
    let value = py
        .detach(move || io::read_json::<Evaluation>(&path))
        .map_err(artifact_read_error)?;
    Ok(NativeEvaluation {
        value: Arc::new(value),
    })
}

#[pyfunction]
#[pyo3(signature = (raw_path, adapter, task_path, mode="strict", max_candidates=None, workers=1))]
fn ingest_file_native(
    py: Python<'_>,
    raw_path: PathBuf,
    adapter: &str,
    task_path: PathBuf,
    mode: &str,
    max_candidates: Option<usize>,
    workers: usize,
) -> PyResult<NativePredictions> {
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let resolved = adapters::built_in(adapter)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown RetroCast adapter {adapter:?}")))?;
    let value = py
        .detach(move || {
            let task: Task = io::read_json(&task_path)?;
            adapt::ingest_file(
                &raw_path,
                resolved.as_ref(),
                &task,
                mode,
                max_candidates,
                workers,
            )
        })
        .map_err(python_error)?;
    Ok(NativePredictions {
        value: Arc::new(value),
    })
}

#[pyfunction]
#[pyo3(signature = (predictions_json, task_json, stocks_json, match_level="full", acceptable_route_match="prefix", execution_stats_json=None, workers=1))]
#[allow(clippy::too_many_arguments)]
fn score_json(
    py: Python<'_>,
    predictions_json: &str,
    task_json: &str,
    stocks_json: &str,
    match_level: &str,
    acceptable_route_match: &str,
    execution_stats_json: Option<&str>,
    workers: usize,
) -> PyResult<String> {
    let predictions: Predictions = from_json(predictions_json)?;
    let task: Task = from_json(task_json)?;
    let stocks = from_json(stocks_json)?;
    let stats: Option<ExecutionStats> = execution_stats_json.map(from_json).transpose()?;
    let result = py
        .detach(|| {
            score::score(
                &predictions,
                &task,
                &stocks,
                match_level,
                acceptable_route_match,
                stats.as_ref(),
                workers,
            )
        })
        .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (predictions, task_json, stocks_json, match_level="full", acceptable_route_match="prefix", execution_stats_json=None, workers=1))]
#[allow(clippy::too_many_arguments)]
fn score_native(
    py: Python<'_>,
    predictions: PyRef<'_, NativePredictions>,
    task_json: &str,
    stocks_json: &str,
    match_level: &str,
    acceptable_route_match: &str,
    execution_stats_json: Option<&str>,
    workers: usize,
) -> PyResult<NativeEvaluation> {
    let predictions = Arc::clone(&predictions.value);
    let task: Task = from_json(task_json)?;
    let stocks = from_json(stocks_json)?;
    let stats: Option<ExecutionStats> = execution_stats_json.map(from_json).transpose()?;
    let match_level = match_level.to_owned();
    let acceptable_route_match = acceptable_route_match.to_owned();
    let value = py
        .detach(move || {
            score::score(
                predictions.as_ref(),
                &task,
                &stocks,
                &match_level,
                &acceptable_route_match,
                stats.as_ref(),
                workers,
            )
        })
        .map_err(python_error)?;
    Ok(NativeEvaluation {
        value: Arc::new(value),
    })
}

#[pyfunction]
#[pyo3(signature = (predictions_path, task_path, stocks_dir, execution_stats_path=None, match_level="full", acceptable_route_match="prefix", workers=1))]
#[allow(clippy::too_many_arguments)]
fn score_project_native(
    py: Python<'_>,
    predictions_path: PathBuf,
    task_path: PathBuf,
    stocks_dir: PathBuf,
    execution_stats_path: Option<PathBuf>,
    match_level: &str,
    acceptable_route_match: &str,
    workers: usize,
) -> PyResult<(NativeEvaluation, String, Vec<PathBuf>)> {
    let match_level = match_level.to_owned();
    let acceptable_route_match = acceptable_route_match.to_owned();
    let (value, label, stock_paths) = py
        .detach(move || {
            let predictions: Predictions = io::read_json(&predictions_path)?;
            let task: Task = io::read_json(&task_path)?;
            let stock_names: BTreeSet<String> = task
                .targets
                .keys()
                .flat_map(|target_id| task.effective_constraints(target_id))
                .filter(|constraint| constraint.kind == "retrocast.stock_termination")
                .filter_map(|constraint| {
                    constraint.fields.get("stock")?.as_str().map(str::to_owned)
                })
                .collect();
            let mut stocks = BTreeMap::new();
            let mut stock_paths = Vec::new();
            for name in stock_names {
                let path = stocks_dir.join(format!("{name}.csv.gz"));
                stocks.extend(io::read_stock(&path, &name)?);
                stock_paths.push(path);
            }
            let stats: Option<ExecutionStats> = execution_stats_path
                .as_deref()
                .map(io::read_json)
                .transpose()?;
            let label = task.derived_metric_label();
            let evaluation = score::score_owned(
                predictions,
                task,
                &stocks,
                &match_level,
                &acceptable_route_match,
                stats.as_ref(),
                workers,
            )?;
            Ok::<_, retrocast_core::error::EngineError>((evaluation, label, stock_paths))
        })
        .map_err(python_error)?;
    Ok((
        NativeEvaluation {
            value: Arc::new(value),
        },
        label,
        stock_paths,
    ))
}

#[pyfunction]
#[pyo3(signature = (evaluation_json, ks, prefix_depths, n_boot=10000, seed=42, workers=1))]
fn analyze_json(
    py: Python<'_>,
    evaluation_json: &str,
    ks: Vec<usize>,
    prefix_depths: Vec<usize>,
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> PyResult<String> {
    let evaluation: Evaluation = from_json(evaluation_json)?;
    let result = py
        .detach(|| analyze::analyze(&evaluation, &ks, &prefix_depths, n_boot, seed, workers))
        .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (evaluation, ks, prefix_depths, n_boot=10000, seed=42, workers=1))]
fn analyze_native(
    py: Python<'_>,
    evaluation: PyRef<'_, NativeEvaluation>,
    ks: Vec<usize>,
    prefix_depths: Vec<usize>,
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> PyResult<String> {
    let evaluation = Arc::clone(&evaluation.value);
    let result = py
        .detach(move || {
            analyze::analyze(
                evaluation.as_ref(),
                &ks,
                &prefix_depths,
                n_boot,
                seed,
                workers,
            )
        })
        .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (evaluation_path, ks, prefix_depths, execution_stats_path=None, n_boot=10000, seed=42, workers=1))]
#[allow(clippy::too_many_arguments)]
fn analyze_file_json(
    py: Python<'_>,
    evaluation_path: PathBuf,
    ks: Vec<usize>,
    prefix_depths: Vec<usize>,
    execution_stats_path: Option<PathBuf>,
    n_boot: usize,
    seed: u64,
    workers: usize,
) -> PyResult<String> {
    let result = py
        .detach(move || {
            let mut evaluation: Evaluation = io::read_json(&evaluation_path)?;
            if let Some(path) = execution_stats_path {
                let stats: ExecutionStats = io::read_json(&path)?;
                for (target_id, target) in &mut evaluation.targets {
                    if target.wall_time.is_none() {
                        target.wall_time = stats.wall_time.get(target_id).copied();
                    }
                    if target.cpu_time.is_none() {
                        target.cpu_time = stats.cpu_time.get(target_id).copied();
                    }
                }
            }
            analyze::analyze(&evaluation, &ks, &prefix_depths, n_boot, seed, workers)
        })
        .map_err(python_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (raw_path, benchmark_path, stock_path, output_dir, stock_name=None, execution_stats_path=None, adapter="aizynthfinder", workers=1, mode="strict", max_candidates=None, match_level="full", acceptable_route_match="prefix", ks=vec![1, 3, 5, 10, 20, 50, 100], prefix_depths=vec![1, 2, 3], n_boot=10000, seed=42))]
#[allow(clippy::too_many_arguments)]
fn run_pipeline_json(
    py: Python<'_>,
    raw_path: PathBuf,
    benchmark_path: PathBuf,
    stock_path: PathBuf,
    output_dir: PathBuf,
    stock_name: Option<String>,
    execution_stats_path: Option<PathBuf>,
    adapter: &str,
    workers: usize,
    mode: &str,
    max_candidates: Option<usize>,
    match_level: &str,
    acceptable_route_match: &str,
    ks: Vec<usize>,
    prefix_depths: Vec<usize>,
    n_boot: usize,
    seed: u64,
) -> PyResult<String> {
    let mode = AdaptMode::parse(mode).map_err(python_error)?;
    let adapter = adapter.to_owned();
    let match_level = match_level.to_owned();
    let acceptable_route_match = acceptable_route_match.to_owned();
    let stats = py
        .detach(move || {
            run_pipeline(
                &raw_path,
                &benchmark_path,
                &stock_path,
                stock_name.as_deref(),
                execution_stats_path.as_deref(),
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
            )
        })
        .map_err(python_error)?;
    to_json(&stats)
}

#[pyfunction]
#[pyo3(signature = (manifest_path, root_dir, deep=false, output_only=false, lenient=true))]
fn verify_manifest_json(
    py: Python<'_>,
    manifest_path: &str,
    root_dir: &str,
    deep: bool,
    output_only: bool,
    lenient: bool,
) -> PyResult<String> {
    let report = py.detach(|| {
        retrocast_core::provenance::verify_manifest(
            Path::new(manifest_path),
            Path::new(root_dir),
            deep,
            output_only,
            lenient,
        )
    });
    to_json(&report)
}

#[derive(serde::Deserialize)]
struct CreateManifestRequest {
    action: String,
    sources: Vec<PathBuf>,
    outputs: Vec<retrocast_core::provenance::ManifestOutput>,
    root_dir: PathBuf,
    #[serde(default)]
    parameters: serde_json::Map<String, Value>,
    #[serde(default)]
    statistics: serde_json::Map<String, Value>,
    #[serde(default)]
    directives: serde_json::Map<String, Value>,
    #[serde(default)]
    summary: serde_json::Map<String, Value>,
    release_name: Option<String>,
    #[serde(default)]
    keyed_output_files: bool,
}

#[pyfunction]
fn create_manifest_json(request_json: &str) -> PyResult<String> {
    let request: CreateManifestRequest = from_json(request_json)?;
    let manifest = retrocast_core::provenance::create_manifest(
        request.action,
        &request.sources,
        &request.outputs,
        &request.root_dir,
        request.parameters,
        request.statistics,
        request.directives,
        request.summary,
        request.release_name,
        request.keyed_output_files,
    )
    .map_err(python_error)?;
    to_json(&manifest)
}

#[pyfunction]
fn hash_file(path: &str) -> PyResult<String> {
    retrocast_core::provenance::file_hash(Path::new(path)).map_err(|error| match error {
        retrocast_core::error::EngineError::Io(error) => PyOSError::new_err(error.to_string()),
        error => python_error(error),
    })
}

#[pyfunction]
fn hash_json(value_json: &str) -> PyResult<String> {
    let value: Value = from_json(value_json)?;
    retrocast_core::provenance::content_hash(&value).map_err(python_error)
}

fn artifact_write_result(result: retrocast_core::error::Result<()>) -> PyResult<()> {
    result.map_err(|error| match error {
        retrocast_core::error::EngineError::Io(error) => PyOSError::new_err(error.to_string()),
        error => python_error(error),
    })
}

#[pyfunction]
fn write_json_gz_json(path: &str, value_json: &str) -> PyResult<()> {
    let value = from_json(value_json)?;
    artifact_write_result(retrocast_core::io::write_json_gz(Path::new(path), &value))
}

#[pyfunction]
fn write_jsonl_gz_json(path: &str, rows_json: &str) -> PyResult<usize> {
    let rows: Vec<Value> = from_json(rows_json)?;
    retrocast_core::io::write_jsonl_gz(Path::new(path), &rows).map_err(python_error)
}

#[pyfunction]
fn write_lines_gz(path: &str, lines: Vec<String>) -> PyResult<usize> {
    retrocast_core::io::write_lines_gz(Path::new(path), &lines).map_err(python_error)
}

#[pyfunction]
fn write_csv_gz(path: &str, rows: Vec<Vec<String>>) -> PyResult<usize> {
    retrocast_core::io::write_csv_gz(Path::new(path), &rows).map_err(python_error)
}

#[pyfunction]
fn read_json_json(path: &str) -> PyResult<String> {
    let value =
        retrocast_core::io::read_json_value(Path::new(path)).map_err(artifact_read_error)?;
    to_json(&value)
}

#[pyfunction]
#[pyo3(signature = (path, skip_empty=true))]
fn read_jsonl_json(path: &str, skip_empty: bool) -> PyResult<String> {
    let values = retrocast_core::io::read_jsonl_values(Path::new(path), skip_empty)
        .map_err(artifact_read_error)?;
    to_json(&values)
}

#[pyfunction]
fn read_lines_gz(path: &str) -> PyResult<Vec<String>> {
    retrocast_core::io::read_lines_gz(Path::new(path)).map_err(artifact_read_error)
}

fn dataset_error(error: retrocast_core::dataset::DatasetError) -> PyErr {
    let payload = serde_json::to_string(&error)
        .unwrap_or_else(|_| format!("{{\"kind\":\"unknown\",\"message\":{error:?}}}"));
    PyRuntimeError::new_err(format!("__retrocast_dataset__{payload}"))
}

#[pyfunction]
fn dataset_build_url(base_url: &str, segments: Vec<String>) -> String {
    retrocast_core::dataset::build_url(base_url, &segments)
}

#[pyfunction]
fn dataset_load_json_url_json(url: &str) -> PyResult<String> {
    let value = retrocast_core::dataset::load_json_url(url).map_err(dataset_error)?;
    to_json(&value)
}

#[pyfunction]
fn dataset_download_url_to_path(url: &str, destination: &str) -> PyResult<()> {
    retrocast_core::dataset::download_url_to_path(url, Path::new(destination))
        .map_err(dataset_error)
}

#[pyfunction]
fn dataset_load_sha256sums_json(path: &str) -> PyResult<String> {
    let values =
        retrocast_core::dataset::load_sha256sums(Path::new(path)).map_err(dataset_error)?;
    to_json(&values)
}

#[pyfunction]
fn dataset_download_training_set_json(request_json: &str) -> PyResult<String> {
    let request = from_json(request_json)?;
    let path = retrocast_core::dataset::download_training_set(&request).map_err(dataset_error)?;
    Ok(path.to_string_lossy().into_owned())
}

#[pyfunction]
fn dataset_download_training_data_json(request_json: &str) -> PyResult<Vec<String>> {
    let request = from_json(request_json)?;
    let paths = retrocast_core::dataset::download_training_data(&request).map_err(dataset_error)?;
    Ok(paths
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect())
}

#[pyfunction]
fn dataset_download_hosted_data_json(request_json: &str) -> PyResult<Vec<String>> {
    let request = from_json(request_json)?;
    let paths = retrocast_core::dataset::download_hosted_data(&request).map_err(dataset_error)?;
    Ok(paths
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect())
}

#[pyfunction]
fn dataset_download_hosted_file(request_json: &str) -> PyResult<String> {
    let request = from_json(request_json)?;
    let path = retrocast_core::dataset::download_hosted_file(&request).map_err(dataset_error)?;
    Ok(path.to_string_lossy().into_owned())
}

#[pyfunction]
fn dataset_validate_training_request(
    dataset: &str,
    artifact: &str,
    split: &str,
    format: &str,
) -> PyResult<()> {
    retrocast_core::dataset::validate_training_request(dataset, artifact, split, format)
        .map_err(dataset_error)
}

#[pyfunction]
fn dataset_resolve_release(dataset: &str, release: &str, base_url: &str) -> PyResult<String> {
    retrocast_core::dataset::resolve_release(dataset, release, base_url).map_err(dataset_error)
}

#[pyfunction]
fn dataset_training_filename(artifact: &str, split: &str, format: &str) -> PyResult<String> {
    retrocast_core::dataset::training_filename(artifact, split, format).map_err(dataset_error)
}

#[pyfunction]
fn dataset_training_root(
    dataset: &str,
    release: &str,
    cache_dir: Option<&str>,
    output_dir: Option<&str>,
) -> String {
    retrocast_core::dataset::training_root(
        dataset,
        release,
        cache_dir.map(Path::new),
        output_dir.map(Path::new),
    )
    .to_string_lossy()
    .into_owned()
}

#[pyfunction]
fn dataset_hosted_root(cache_dir: Option<&str>, output_dir: Option<&str>) -> String {
    retrocast_core::dataset::hosted_root(cache_dir.map(Path::new), output_dir.map(Path::new))
        .to_string_lossy()
        .into_owned()
}

#[pyfunction]
#[pyo3(signature = (key, artifact=None, split=None, format=None, omit=Vec::new()))]
fn dataset_training_file_matches(
    key: &str,
    artifact: Option<&str>,
    split: Option<&str>,
    format: Option<&str>,
    omit: Vec<String>,
) -> bool {
    retrocast_core::dataset::training_file_matches(key, artifact, split, format, &omit)
}

#[pyfunction]
fn dataset_resolve_expected(path: &str, url: &str, key: &str) -> PyResult<String> {
    retrocast_core::dataset::resolve_expected(Path::new(path), url, key).map_err(dataset_error)
}

fn inchi_level(value: &str) -> PyResult<InchiKeyLevel> {
    match value {
        "full" => Ok(InchiKeyLevel::Full),
        "no_stereo" => Ok(InchiKeyLevel::NoStereo),
        "connectivity" => Ok(InchiKeyLevel::Connectivity),
        _ => Err(PyRuntimeError::new_err(format!(
            "unsupported InChIKey match level {value:?}"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (query_json, container_json, match_level="full", allow_leaf_extension=true))]
fn find_route_embeddings_json(
    py: Python<'_>,
    query_json: &str,
    container_json: &str,
    match_level: &str,
    allow_leaf_extension: bool,
) -> PyResult<String> {
    let query = from_json(query_json)?;
    let container = from_json(container_json)?;
    let options = EmbeddingOptions {
        match_level: inchi_level(match_level)?,
        allow_leaf_extension,
    };
    let matches = py
        .detach(|| find_route_embeddings(&query, &container, options))
        .map_err(python_error)?;
    to_json(&matches)
}

#[pyfunction]
#[pyo3(signature = (query_json, query_path, container_json, container_path, match_level="full", allow_leaf_extension=true))]
fn route_embeds_at_json(
    py: Python<'_>,
    query_json: &str,
    query_path: &str,
    container_json: &str,
    container_path: &str,
    match_level: &str,
    allow_leaf_extension: bool,
) -> PyResult<String> {
    let query: retrocast_core::model::Route = from_json(query_json)?;
    let container: retrocast_core::model::Route = from_json(container_json)?;
    let query_path = RoutePath::parse(query_path).map_err(python_error)?;
    let container_path = RoutePath::parse(container_path).map_err(python_error)?;
    let query_view = query.molecule_at(&query_path).map_err(python_error)?;
    let container_view = container
        .molecule_at(&container_path)
        .map_err(python_error)?;
    let options = EmbeddingOptions {
        match_level: inchi_level(match_level)?,
        allow_leaf_extension,
    };
    let matched = py.detach(|| route_embeds_at(query_view, container_view, options));
    to_json(&matched)
}

#[pyfunction]
fn subtree_reaction_count_json(route_json: &str, molecule_path: &str) -> PyResult<usize> {
    let route: retrocast_core::model::Route = from_json(route_json)?;
    let path = RoutePath::parse(molecule_path).map_err(python_error)?;
    let molecule = route.molecule_at(&path).map_err(python_error)?;
    Ok(retrocast_core::embedding::subtree_reaction_count(&molecule))
}

#[pyfunction]
fn excise_reactions_json(route_json: &str, excluded: Vec<String>) -> PyResult<String> {
    let route = from_json(route_json)?;
    let routes =
        retrocast_core::curation::excise_reactions(&route, &excluded.into_iter().collect());
    to_json(&routes)
}

#[pyfunction]
fn deduplicate_routes_json(routes_json: &str) -> PyResult<String> {
    let routes: Vec<retrocast_core::model::Route> = from_json(routes_json)?;
    to_json(&retrocast_core::curation::deduplicate_routes(routes))
}

#[pyfunction]
fn filter_by_route_type_json(task_json: &str, route_type: &str) -> PyResult<String> {
    let task: Task = from_json(task_json)?;
    let convergent = match route_type {
        "linear" => false,
        "convergent" => true,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown route type: {route_type}"
            )));
        }
    };
    to_json(&retrocast_core::curation::filter_by_route_type(
        &task, convergent,
    ))
}

#[pyfunction]
fn clean_and_prioritize_pools_json(primary_json: &str, secondary_json: &str) -> PyResult<String> {
    let primary = from_json(primary_json)?;
    let secondary = from_json(secondary_json)?;
    to_json(&retrocast_core::curation::clean_and_prioritize_pools(
        primary, secondary,
    ))
}

#[pyfunction]
fn generate_pruned_routes_json(route_json: &str, stock: Vec<String>) -> PyResult<String> {
    let route = from_json(route_json)?;
    let stock: HashSet<_> = stock.into_iter().collect();
    to_json(&retrocast_core::curation::generate_pruned_routes(
        &route, &stock,
    ))
}

#[pyfunction]
fn route_is_convergent_json(route_json: &str) -> PyResult<bool> {
    let route: retrocast_core::model::Route = from_json(route_json)?;
    Ok(route.is_convergent())
}

#[pyfunction]
fn sample_indices(population_size: usize, sample_size: usize, seed: &str) -> PyResult<Vec<usize>> {
    retrocast_core::sampling::sample_indices(population_size, sample_size, seed)
        .map_err(python_error)
}

#[pyfunction]
fn sample_stratified_priority_indices(
    grouped_pool_sizes: Vec<Vec<usize>>,
    target_counts: Vec<usize>,
    seed: &str,
) -> PyResult<Vec<Vec<(usize, usize)>>> {
    let selected = retrocast_core::sampling::sample_stratified_priority_indices(
        &grouped_pool_sizes,
        &target_counts,
        seed,
    )
    .map_err(python_error)?;
    Ok(selected
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|coordinate| (coordinate.pool, coordinate.index))
                .collect()
        })
        .collect())
}

#[pyfunction]
fn training_validation_indices_json(
    routes_json: &str,
    validation_fraction: f64,
    seed: &str,
) -> PyResult<Vec<usize>> {
    let routes: Vec<retrocast_core::model::Route> = from_json(routes_json)?;
    retrocast_core::training::validation_indices(&routes, validation_fraction, seed)
        .map_err(python_error)
}

fn training_error(error: retrocast_core::training::TrainingError) -> PyErr {
    let payload = serde_json::to_string(&error).unwrap_or_else(|_| {
        format!("{{\"code\":\"workflow.training_release_error\",\"message\":{error:?}}}")
    });
    PyRuntimeError::new_err(format!("__retrocast_training__{payload}"))
}

#[pyfunction]
#[pyo3(signature = (dataset, routes_json, route_prefix="paroutes"))]
fn build_test_route_records_json(
    dataset: &str,
    routes_json: &str,
    route_prefix: &str,
) -> PyResult<String> {
    let routes: Vec<retrocast_core::training::AdaptedTrainingRoute> = from_json(routes_json)?;
    to_json(&retrocast_core::training::build_test_route_records(
        dataset,
        &routes,
        route_prefix,
    ))
}

#[pyfunction]
fn adapt_training_routes_json(raw_json: &str, dataset: &str) -> PyResult<String> {
    let raw: serde_json::Value = from_json(raw_json)?;
    let result =
        retrocast_core::training::adapt_training_routes(raw, dataset).map_err(training_error)?;
    to_json(&result)
}

#[pyfunction]
#[pyo3(signature = (dataset, route_records_json, route_prefix="paroutes"))]
fn build_test_reaction_records_json(
    dataset: &str,
    route_records_json: &str,
    route_prefix: &str,
) -> PyResult<String> {
    let records: Vec<retrocast_core::training::TestRouteRecord> = from_json(route_records_json)?;
    let result =
        retrocast_core::training::build_test_reaction_records(dataset, &records, route_prefix)
            .map_err(training_error)?;
    to_json(&result)
}

#[pyfunction]
fn build_training_reaction_release_json(
    route_records_json: &str,
    config_json: &str,
) -> PyResult<String> {
    let records: Vec<retrocast_core::training::TrainingRouteRecord> =
        from_json(route_records_json)?;
    let config: retrocast_core::training::TrainingSetBuildConfig = from_json(config_json)?;
    let result = retrocast_core::training::build_training_reaction_release(&records, &config)
        .map_err(training_error)?;
    to_json(&result)
}

#[pyfunction]
fn build_training_route_release_json(
    all_routes_json: &str,
    all_adaptation_json: &str,
    holdout_routes_json: &str,
    holdout_adaptation_json: &str,
    config_json: &str,
) -> PyResult<String> {
    let all_routes: Vec<retrocast_core::training::AdaptedTrainingRoute> =
        from_json(all_routes_json)?;
    let all_adaptation: serde_json::Value = from_json(all_adaptation_json)?;
    let holdout_routes: std::collections::BTreeMap<
        String,
        Vec<retrocast_core::training::AdaptedTrainingRoute>,
    > = from_json(holdout_routes_json)?;
    let holdout_adaptation: std::collections::BTreeMap<String, serde_json::Value> =
        from_json(holdout_adaptation_json)?;
    let config: retrocast_core::training::TrainingSetBuildConfig = from_json(config_json)?;
    let result = retrocast_core::training::build_training_route_release(
        &all_routes,
        all_adaptation,
        &holdout_routes,
        holdout_adaptation,
        &config,
    )
    .map_err(training_error)?;
    to_json(&result)
}

#[pyfunction]
fn audit_route_release_json(
    release_name: &str,
    all_json: &str,
    training_json: &str,
    validation_json: &str,
) -> PyResult<()> {
    let all: Vec<retrocast_core::training::TrainingRouteRecord> = from_json(all_json)?;
    let training: Vec<retrocast_core::training::TrainingRouteRecord> = from_json(training_json)?;
    let validation: Vec<retrocast_core::training::TrainingRouteRecord> =
        from_json(validation_json)?;
    retrocast_core::training::audit_route_release(release_name, &all, &training, &validation)
        .map_err(training_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn audit_single_step_release_json(
    release_name: &str,
    all_json: &str,
    training_json: &str,
    validation_json: &str,
    all_rsmi_count: usize,
    training_rsmi_count: usize,
    validation_rsmi_count: usize,
    parent_route_ids: Vec<String>,
) -> PyResult<String> {
    let all: Vec<retrocast_core::training::TrainingReactionRecord> = from_json(all_json)?;
    let training: Vec<retrocast_core::training::TrainingReactionRecord> = from_json(training_json)?;
    let validation: Vec<retrocast_core::training::TrainingReactionRecord> =
        from_json(validation_json)?;
    let result = retrocast_core::training::audit_single_step_release(
        release_name,
        &all,
        &training,
        &validation,
        all_rsmi_count,
        training_rsmi_count,
        validation_rsmi_count,
        &parent_route_ids.into_iter().collect(),
    )
    .map_err(training_error)?;
    to_json(&result)
}

#[pyfunction]
fn build_route_embedding_audit_json(
    release_name: &str,
    containers_json: &str,
    query_sources_json: &str,
    options_json: &str,
) -> PyResult<String> {
    let containers: Vec<retrocast_core::training::TrainingRouteRecord> =
        from_json(containers_json)?;
    let query_sources: Vec<retrocast_core::embedding_audit::QuerySource> =
        from_json(query_sources_json)?;
    let options: retrocast_core::embedding_audit::RouteEmbeddingAuditOptions =
        from_json(options_json)?;
    let audit = retrocast_core::embedding_audit::build_route_embedding_audit(
        release_name,
        &containers,
        &query_sources,
        options,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    to_json(&audit)
}

#[pyfunction]
fn engine_info() -> (&'static str, &'static str, String) {
    (
        retrocast_core::VERSION,
        "RDKit C++",
        retrocast_core::chem::version(),
    )
}

#[pyfunction]
#[pyo3(signature = (smiles, remove_mapping=false, ignore_stereo=false))]
fn canonicalize_smiles(
    py: Python<'_>,
    smiles: &str,
    remove_mapping: bool,
    ignore_stereo: bool,
) -> PyResult<String> {
    py.detach(|| chem::canonicalize(smiles, remove_mapping, ignore_stereo))
        .map(|result| result.to_string())
        .map_err(chemistry_error)
}

#[pyfunction]
#[pyo3(signature = (smiles, level="full"))]
fn get_inchi_key(py: Python<'_>, smiles: &str, level: &str) -> PyResult<String> {
    let level = parse_chem_level(level)?;
    py.detach(|| chem::inchi_key(smiles, level))
        .map_err(chemistry_error)
}

#[pyfunction]
fn reduce_inchi_key(inchikey: &str, level: &str) -> PyResult<String> {
    chem::reduce_inchi_key(inchikey, parse_chem_level(level)?).map_err(chemistry_error)
}

fn parse_chem_level(level: &str) -> PyResult<ChemInchiKeyLevel> {
    match level {
        "full" => Ok(ChemInchiKeyLevel::Full),
        "no_stereo" => Ok(ChemInchiKeyLevel::NoStereo),
        "connectivity" => Ok(ChemInchiKeyLevel::Connectivity),
        value => Err(PyValueError::new_err(format!(
            "unknown inchikey level: {value}"
        ))),
    }
}

#[pyfunction]
fn molecular_descriptors(py: Python<'_>, smiles: &str) -> PyResult<(u32, f64, u32)> {
    py.detach(|| chem::descriptors(smiles))
        .map(|result| {
            (
                result.heavy_atom_count,
                result.molecular_weight,
                result.chiral_center_count,
            )
        })
        .map_err(chemistry_error)
}

#[pymodule]
#[pyo3(name = "_native")]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePredictions>()?;
    module.add_class::<NativeEvaluation>()?;
    module.add_function(wrap_pyfunction!(canonicalize_smiles, module)?)?;
    module.add_function(wrap_pyfunction!(get_inchi_key, module)?)?;
    module.add_function(wrap_pyfunction!(reduce_inchi_key, module)?)?;
    module.add_function(wrap_pyfunction!(molecular_descriptors, module)?)?;
    module.add_function(wrap_pyfunction!(adapt_candidates_json, module)?)?;
    module.add_function(wrap_pyfunction!(adapt_route_json, module)?)?;
    module.add_function(wrap_pyfunction!(adapter_entries_json, module)?)?;
    module.add_function(wrap_pyfunction!(adapter_cast_result_json, module)?)?;
    module.add_function(wrap_pyfunction!(dms_route_length_json, module)?)?;
    module.add_function(wrap_pyfunction!(synllama_precursor_map_json, module)?)?;
    module.add_function(wrap_pyfunction!(reaction_string_parse_json, module)?)?;
    module.add_function(wrap_pyfunction!(synplanner_entries_json, module)?)?;
    module.add_function(wrap_pyfunction!(paroutes_condition_stats_json, module)?)?;
    module.add_function(wrap_pyfunction!(candidate_statistics_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        collected_candidate_statistics_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        collected_candidate_statistics_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(candidate_run_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(evaluation_statistics_json, module)?)?;
    module.add_function(wrap_pyfunction!(evaluation_statistics_native, module)?)?;
    module.add_function(wrap_pyfunction!(bootstrap_distribution_json, module)?)?;
    module.add_function(wrap_pyfunction!(reliability_flag_json, module)?)?;
    module.add_function(wrap_pyfunction!(summarize_values_json, module)?)?;
    module.add_function(wrap_pyfunction!(probabilistic_ranking_json, module)?)?;
    module.add_function(wrap_pyfunction!(paired_difference_json, module)?)?;
    module.add_function(wrap_pyfunction!(summarize_targets_json, module)?)?;
    module.add_function(wrap_pyfunction!(check_task_constraints_json, module)?)?;
    module.add_function(wrap_pyfunction!(target_metric_json, module)?)?;
    module.add_function(wrap_pyfunction!(collect_candidates_json, module)?)?;
    module.add_function(wrap_pyfunction!(collect_routes_json, module)?)?;
    module.add_function(wrap_pyfunction!(route_path_parse, module)?)?;
    module.add_function(wrap_pyfunction!(route_path_transform, module)?)?;
    module.add_function(wrap_pyfunction!(reactant_order_json, module)?)?;
    module.add_function(wrap_pyfunction!(route_identity_json, module)?)?;
    module.add_function(wrap_pyfunction!(route_structure_json, module)?)?;
    module.add_function(wrap_pyfunction!(ingest_json, module)?)?;
    module.add_function(wrap_pyfunction!(ingest_native, module)?)?;
    module.add_function(wrap_pyfunction!(ingest_file_native, module)?)?;
    module.add_function(wrap_pyfunction!(load_predictions_native, module)?)?;
    module.add_function(wrap_pyfunction!(load_evaluation_native, module)?)?;
    module.add_function(wrap_pyfunction!(score_json, module)?)?;
    module.add_function(wrap_pyfunction!(score_native, module)?)?;
    module.add_function(wrap_pyfunction!(score_project_native, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_json, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_native, module)?)?;
    module.add_function(wrap_pyfunction!(analyze_file_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_pipeline_json, module)?)?;
    module.add_function(wrap_pyfunction!(verify_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(create_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(hash_file, module)?)?;
    module.add_function(wrap_pyfunction!(hash_json, module)?)?;
    module.add_function(wrap_pyfunction!(write_json_gz_json, module)?)?;
    module.add_function(wrap_pyfunction!(write_jsonl_gz_json, module)?)?;
    module.add_function(wrap_pyfunction!(write_lines_gz, module)?)?;
    module.add_function(wrap_pyfunction!(write_csv_gz, module)?)?;
    module.add_function(wrap_pyfunction!(read_json_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_jsonl_json, module)?)?;
    module.add_function(wrap_pyfunction!(read_lines_gz, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_build_url, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_load_json_url_json, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_download_url_to_path, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_load_sha256sums_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        dataset_download_training_set_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        dataset_download_training_data_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(dataset_download_hosted_data_json, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_download_hosted_file, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_validate_training_request, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_resolve_release, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_training_filename, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_training_root, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_hosted_root, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_training_file_matches, module)?)?;
    module.add_function(wrap_pyfunction!(dataset_resolve_expected, module)?)?;
    module.add_function(wrap_pyfunction!(find_route_embeddings_json, module)?)?;
    module.add_function(wrap_pyfunction!(route_embeds_at_json, module)?)?;
    module.add_function(wrap_pyfunction!(subtree_reaction_count_json, module)?)?;
    module.add_function(wrap_pyfunction!(excise_reactions_json, module)?)?;
    module.add_function(wrap_pyfunction!(deduplicate_routes_json, module)?)?;
    module.add_function(wrap_pyfunction!(filter_by_route_type_json, module)?)?;
    module.add_function(wrap_pyfunction!(clean_and_prioritize_pools_json, module)?)?;
    module.add_function(wrap_pyfunction!(generate_pruned_routes_json, module)?)?;
    module.add_function(wrap_pyfunction!(route_is_convergent_json, module)?)?;
    module.add_function(wrap_pyfunction!(sample_indices, module)?)?;
    module.add_function(wrap_pyfunction!(
        sample_stratified_priority_indices,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(training_validation_indices_json, module)?)?;
    module.add_function(wrap_pyfunction!(adapt_training_routes_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_test_route_records_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_test_reaction_records_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_training_reaction_release_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_training_route_release_json, module)?)?;
    module.add_function(wrap_pyfunction!(audit_route_release_json, module)?)?;
    module.add_function(wrap_pyfunction!(audit_single_step_release_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_route_embedding_audit_json, module)?)?;
    module.add_function(wrap_pyfunction!(engine_info, module)?)?;
    Ok(())
}
