mod askcos;
mod bipartite;
mod common;
mod dms;
mod molbuilder;
mod multistepttl;
mod paroutes;
mod reaction_string;
mod retrochimera;
mod synllama;
mod ursa;

use rayon::prelude::*;
use serde_json::{Map, Value, json};

pub use askcos::AskcosAdapter;
pub use bipartite::{
    SynPlannerAdapter, SynPlannerEntryBatch, SynPlannerSkippedRoute, SyntheseusAdapter,
    extract_synplanner_entries,
};
pub use dms::{DirectMultiStepAdapter, route_length as dms_route_length};
pub use molbuilder::MolBuilderAdapter;
pub use multistepttl::MultiStepTtlAdapter;
pub use paroutes::{ConditionSlotParseStatistics, PaRoutesAdapter, analyze_condition_slots};
pub use reaction_string::{DreamRetroErAdapter, RetroStarAdapter, parse_reaction_string};
pub use retrochimera::RetroChimeraAdapter;
pub use synllama::{SynLlamaAdapter, parse_synthesis_string as parse_synllama_synthesis};
pub use ursa::UrsaAdapter;

use crate::{
    error::{EngineError, Result},
    model::{Candidate, FailureRecord, RawNode, Route, Target},
    route::{AdaptMode, adapt_route},
    with_pool,
};

pub const BUILT_IN_ADAPTERS: &[&str] = &[
    "aizynthfinder",
    "askcos",
    "directmultistep",
    "dreamretroer",
    "molbuilder",
    "multistepttl",
    "paroutes",
    "retrochimera",
    "retrostar",
    "synllama",
    "synplanner",
    "syntheseus",
    "ursa",
];

pub const DEPRECATED_ADAPTER_ALIASES: &[(&str, &str)] = &[
    ("aizynth", "aizynthfinder"),
    ("dreamretro", "dreamretroer"),
    ("retro-star", "retrostar"),
];

#[derive(Clone, Debug, serde::Serialize)]
pub struct RawRouteEntry {
    pub payload: Value,
    pub source_key: Option<String>,
    pub source_row_index: Option<usize>,
    pub source_record_id: Option<String>,
    pub target_hint_id: Option<String>,
    pub target_hint_smiles: Option<String>,
    pub source_order: Option<usize>,
}

impl RawRouteEntry {
    pub fn new(payload: Value) -> Self {
        Self {
            payload,
            source_key: None,
            source_row_index: None,
            source_record_id: None,
            target_hint_id: None,
            target_hint_smiles: None,
            source_order: None,
        }
    }
}

pub trait Adapter: Send + Sync {
    fn name(&self) -> &'static str;

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>>;

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route>;

    #[allow(clippy::too_many_arguments)]
    fn candidates(
        &self,
        payload: Value,
        mode: AdaptMode,
        target: Option<&Target>,
        source_key: Option<&str>,
        max_candidates: Option<usize>,
        workers: usize,
    ) -> Result<Vec<Candidate>> {
        adapt_entries_with_workers(
            payload,
            self,
            mode,
            target,
            source_key,
            max_candidates,
            workers,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AiZynthFinderAdapter;

impl Adapter for AiZynthFinderAdapter {
    fn name(&self) -> &'static str {
        "aizynth"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let Value::Array(routes) = payload else {
            return Err(EngineError::AdapterSchema(
                "route payload must be a list".to_owned(),
            ));
        };
        Ok(routes
            .into_iter()
            .enumerate()
            .map(|(index, payload)| RawRouteEntry {
                payload,
                source_key: source_key.map(str::to_owned),
                source_row_index: None,
                source_record_id: None,
                target_hint_id: None,
                target_hint_smiles: None,
                source_order: Some(index + 1),
            })
            .collect())
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        let node: RawNode = serde_json::from_value(raw_route)?;
        let route = adapt_route(&node, mode)?;
        if let Some(target) = target {
            let (expected, _) = crate::chem::normalize(&target.smiles)?;
            if route.target.smiles != expected {
                return Err(EngineError::TargetMismatch {
                    adapter: self.name(),
                    target_id: target.id.clone(),
                    expected: expected.to_string(),
                    actual: route.target.smiles.to_string(),
                });
            }
        }
        Ok(route)
    }
}

pub fn adapt_candidates(
    payload: Value,
    adapter: &dyn Adapter,
    mode: AdaptMode,
    target: Option<&Target>,
    source_key: Option<&str>,
    max_candidates: Option<usize>,
) -> Result<Vec<Candidate>> {
    adapt_candidates_with_workers(
        payload,
        adapter,
        mode,
        target,
        source_key,
        max_candidates,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn adapt_candidates_with_workers(
    payload: Value,
    adapter: &dyn Adapter,
    mode: AdaptMode,
    target: Option<&Target>,
    source_key: Option<&str>,
    max_candidates: Option<usize>,
    workers: usize,
) -> Result<Vec<Candidate>> {
    adapter.candidates(payload, mode, target, source_key, max_candidates, workers)
}

#[allow(clippy::too_many_arguments)]
fn adapt_entries_with_workers<A: Adapter + ?Sized>(
    payload: Value,
    adapter: &A,
    mode: AdaptMode,
    target: Option<&Target>,
    source_key: Option<&str>,
    max_candidates: Option<usize>,
    workers: usize,
) -> Result<Vec<Candidate>> {
    let mut entries = adapter.entries(payload, source_key)?;
    if let Some(limit) = max_candidates {
        entries.truncate(limit);
    }

    if workers == 1 {
        return Ok(entries
            .into_iter()
            .enumerate()
            .map(|(index, entry)| candidate_from_entry(adapter, index, entry, mode, target))
            .collect());
    }

    let candidates = with_pool(workers, || {
        entries
            .into_par_iter()
            .enumerate()
            .map(|(index, entry)| candidate_from_entry(adapter, index, entry, mode, target))
            .collect()
    })?;
    Ok(candidates)
}

fn candidate_from_entry<A: Adapter + ?Sized>(
    adapter: &A,
    index: usize,
    mut entry: RawRouteEntry,
    mode: AdaptMode,
    target: Option<&Target>,
) -> Candidate {
    let rank = entry.source_order.unwrap_or(index + 1);
    let raw_route = std::mem::take(&mut entry.payload);
    let inferred_target = if target.is_none() {
        target_from_entry(&entry)
    } else {
        None
    };
    let entry_target = target.or(inferred_target.as_ref());
    candidate_from_result(
        adapter.name(),
        rank,
        adapter.cast(raw_route, mode, entry_target),
        entry_target,
        &entry,
    )
}

pub(super) fn candidate_from_result(
    adapter_name: &str,
    rank: usize,
    route: Result<Route>,
    target: Option<&Target>,
    entry: &RawRouteEntry,
) -> Candidate {
    match route {
        Ok(route) => Candidate {
            rank,
            route: Some(route),
            failure: None,
        },
        Err(error) => Candidate {
            rank,
            route: None,
            failure: Some(failure_record(adapter_name, error, target, entry)),
        },
    }
}

fn target_from_entry(entry: &RawRouteEntry) -> Option<Target> {
    let hinted_smiles = entry.target_hint_smiles.as_deref()?;
    let (smiles, inchikey) = crate::chem::normalize(hinted_smiles).ok()?;
    Some(Target {
        id: entry
            .target_hint_id
            .clone()
            .or_else(|| entry.source_key.clone())
            .unwrap_or_else(|| smiles.to_string()),
        smiles,
        inchikey,
        acceptable_routes: Vec::new(),
        annotations: Map::new(),
    })
}

fn failure_record(
    adapter: &str,
    error: EngineError,
    target: Option<&Target>,
    entry: &RawRouteEntry,
) -> FailureRecord {
    let context = failure_context(adapter, &error);
    FailureRecord {
        code: error_code(&error).to_owned(),
        message: Some(failure_message(&error)),
        target_id: target
            .map(|target| target.id.clone())
            .or_else(|| entry.target_hint_id.clone()),
        target_smiles: target.map(|target| target.smiles.clone()),
        target_inchikey: target.map(|target| target.inchikey.clone()),
        context,
    }
}

pub fn boundary_failure(
    adapter: &str,
    error: EngineError,
    target: Option<&Target>,
) -> FailureRecord {
    failure_record(adapter, error, target, &RawRouteEntry::new(Value::Null))
}

fn failure_message(error: &EngineError) -> String {
    match error {
        EngineError::Chemistry { smiles, .. } | EngineError::InvalidSmiles { smiles, .. } => {
            format!("Invalid SMILES string: {smiles}")
        }
        EngineError::TargetMismatch {
            adapter,
            target_id,
            expected,
            actual,
        } => format!(
            "{} produced mismatched SMILES for target {target_id}. expected canonical: {expected}, but adapter produced: {actual}",
            adapter_display_name(adapter)
        ),
        _ => error.to_string(),
    }
}

fn adapter_display_name(adapter: &str) -> &str {
    match adapter {
        "aizynth" => "AiZynthFinder",
        "askcos" => "ASKCOS",
        "dms" => "DMS",
        "directmultistep" => "DirectMultiStep",
        "dreamretro" => "DreamRetro",
        "dreamretroer" => "DreamRetroEr",
        "molbuilder" => "MolBuilder",
        "multistepttl" => "MultiStepTTL",
        "paroutes" => "PaRoutes",
        "retrochimera" => "RetroChimera",
        "retrostar" => "RetroStar",
        "synllama" => "SynLlama",
        "synplanner" => "SynPlanner",
        "syntheseus" => "Syntheseus",
        "ursa" => "URSA",
        other => other,
    }
}

fn failure_context(adapter: &str, error: &EngineError) -> Map<String, Value> {
    match error {
        EngineError::Chemistry { smiles, .. } | EngineError::InvalidSmiles { smiles, .. } => {
            Map::from_iter([
                ("operation".to_owned(), json!("canonicalize_smiles")),
                ("smiles".to_owned(), json!(smiles)),
            ])
        }
        EngineError::TargetMismatch {
            adapter,
            target_id,
            expected,
            actual,
        } => Map::from_iter([
            ("adapter".to_owned(), json!(adapter)),
            ("target_id".to_owned(), json!(target_id)),
            ("expected_smiles".to_owned(), json!(expected)),
            ("actual_smiles".to_owned(), json!(actual)),
        ]),
        EngineError::AdapterLogic {
            adapter: "askcos",
            code: "adapter.unsupported_feature",
            ..
        } => Map::from_iter([
            ("adapter".to_owned(), json!("askcos")),
            ("feature".to_owned(), json!("full_graph")),
        ]),
        EngineError::AdapterLogic { adapter, .. } => {
            Map::from_iter([("adapter".to_owned(), json!(adapter))])
        }
        EngineError::AdapterSchemaContext { context, .. } => context.clone(),
        _ => Map::from_iter([("adapter".to_owned(), json!(adapter))]),
    }
}

pub(crate) fn error_code(error: &EngineError) -> &'static str {
    match error {
        EngineError::Chemistry { .. } | EngineError::InvalidSmiles { .. } => "chem.invalid_smiles",
        EngineError::TargetMismatch { .. } => "adapter.target_mismatch",
        EngineError::AdapterLogic { code, .. } => code,
        EngineError::RouteShape(_) => "adapter.logic_error",
        _ => "adapter.schema_invalid",
    }
}

pub fn built_in(name: &str) -> Option<Box<dyn Adapter>> {
    match normalize_slug(name).as_str() {
        "aizynth" | "aizynthfinder" => Some(Box::new(AiZynthFinderAdapter)),
        "askcos" => Some(Box::new(AskcosAdapter::default())),
        "askcosfullgraph" => Some(Box::new(AskcosAdapter::full_graph())),
        "directmultistep" | "dms" => Some(Box::new(DirectMultiStepAdapter)),
        "multistepttl" => Some(Box::new(MultiStepTtlAdapter)),
        "molbuilder" => Some(Box::new(MolBuilderAdapter)),
        "paroutes" => Some(Box::new(PaRoutesAdapter)),
        "dreamretroer" => Some(Box::new(DreamRetroErAdapter)),
        "retrostar" => Some(Box::new(RetroStarAdapter)),
        "retrochimera" => Some(Box::new(RetroChimeraAdapter)),
        "synllama" => Some(Box::new(SynLlamaAdapter)),
        "synplanner" => Some(Box::new(SynPlannerAdapter)),
        "syntheseus" => Some(Box::new(SyntheseusAdapter)),
        "ursa" => Some(Box::new(UrsaAdapter)),
        _ => None,
    }
}

pub fn normalize_slug(name: &str) -> String {
    match name.trim().to_lowercase().as_str() {
        "retro-star" => "retrostar".to_owned(),
        "dreamretro" => "dreamretroer".to_owned(),
        "aizynth" => "aizynthfinder".to_owned(),
        normalized => normalized.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{AiZynthFinderAdapter, adapt_candidates};
    use crate::route::AdaptMode;

    #[test]
    fn generic_candidate_path_preserves_failed_slots() {
        let candidates = adapt_candidates(
            json!([
                {"type": "mol", "smiles": "CCO"},
                {"type": "mol", "smiles": "not-smiles"}
            ]),
            &AiZynthFinderAdapter,
            AdaptMode::Strict,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(candidates[0].route.is_some());
        assert_eq!(
            candidates[1].failure.as_ref().unwrap().code,
            "chem.invalid_smiles"
        );
    }
}
