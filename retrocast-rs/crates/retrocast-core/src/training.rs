//! Training-release construction owned by the RetroCast engine.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use thiserror::Error;

use crate::{
    adapters::{Adapter, ConditionSlotParseStatistics, PaRoutesAdapter, analyze_condition_slots},
    curation,
    error::EngineError,
    model::{Molecule, Route, Target},
    route::AdaptMode,
    route_path::RoutePath,
    route_view::InchiKeyLevel,
    sampling,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RawRouteSource {
    pub dataset: String,
    pub raw_index: usize,
    pub raw_route_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patent_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AdaptedTrainingRoute {
    pub route: Route,
    pub source: RawRouteSource,
}

#[derive(Clone, Debug, Serialize)]
pub struct AdaptationStatistics {
    pub raw_routes: usize,
    pub adapted_routes: usize,
    pub skipped_routes: usize,
    pub skipped_without_error_code: usize,
    pub failures_by_code: BTreeMap<String, usize>,
    pub non_fatal_condition_slot_parse: ConditionSlotParseStatistics,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingRouteAdaptation {
    pub routes: Vec<AdaptedTrainingRoute>,
    pub stats: AdaptationStatistics,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TestRouteRecord {
    pub id: String,
    pub dataset: String,
    pub route: Route,
    #[serde(default)]
    pub sources: Vec<RawRouteSource>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrainingReactionSource {
    pub route_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_index: Option<usize>,
    pub reaction_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TestReactionRecord {
    pub id: String,
    pub dataset: String,
    pub reactants: Vec<String>,
    pub product: String,
    pub mapped_smiles: String,
    #[serde(default)]
    pub alternative_mapped_smiles: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_slot: Option<String>,
    #[serde(default)]
    pub condition_slot_smiles: Vec<String>,
    #[serde(default)]
    pub sources: Vec<TrainingReactionSource>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrainingRouteRecord {
    pub id: String,
    pub split: String,
    pub route: Route,
    #[serde(default)]
    pub sources: Vec<RawRouteSource>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrainingReactionRecord {
    pub id: String,
    pub split: String,
    pub reactants: Vec<String>,
    pub product: String,
    pub mapped_smiles: String,
    #[serde(default)]
    pub alternative_mapped_smiles: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition_slot: Option<String>,
    #[serde(default)]
    pub condition_slot_smiles: Vec<String>,
    #[serde(default)]
    pub sources: Vec<TrainingReactionSource>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrainingSetBuildConfig {
    pub holdout_mode: String,
    #[serde(default = "default_validation_fraction")]
    pub val_fraction: f64,
    #[serde(default = "default_training_seed")]
    pub seed: i64,
    #[serde(default = "default_route_prefix")]
    pub route_prefix: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingReactionBuildResult {
    pub release_name: String,
    pub records: Vec<TrainingReactionRecord>,
    pub summary: Value,
    #[serde(rename = "_warnings", skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingSetBuildResult {
    pub release_name: String,
    pub records: Vec<TrainingRouteRecord>,
    pub summary: Value,
}

#[derive(Clone)]
struct PreparedTrainingRoute {
    route: Route,
    sources: Vec<RawRouteSource>,
}

#[derive(Debug, Error, Serialize)]
#[error("{message}")]
pub struct TrainingError {
    pub code: String,
    pub message: String,
    pub context: Value,
}

pub fn adapt_training_routes(
    payload: Value,
    dataset: &str,
) -> Result<TrainingRouteAdaptation, TrainingError> {
    let adapter = PaRoutesAdapter;
    let entries = adapter
        .entries(payload, None)
        .map_err(training_adapter_error)?;
    let raw_routes = entries.len();
    let mut parse_stats = ConditionSlotParseStatistics::default();
    let mut routes = Vec::new();
    let mut failures_by_code: BTreeMap<String, usize> = BTreeMap::new();
    let mut reaction_hash_signature_pairs = HashSet::new();

    for entry in entries {
        let source_order = entry
            .source_order
            .expect("PaRoutes entries have source order");
        let Some(raw_smiles) = entry.payload.get("smiles").and_then(Value::as_str) else {
            return Err(training_error(
                "adapter.schema_invalid",
                "PaRoutes route root is missing a string smiles field",
                json!({"adapter": "paroutes"}),
            ));
        };
        let (smiles, inchikey) = match crate::chem::normalize(raw_smiles) {
            Ok(identity) => identity,
            Err(error) => {
                *failures_by_code
                    .entry(crate::adapters::error_code(&error).to_owned())
                    .or_default() += 1;
                continue;
            }
        };
        analyze_condition_slots(&entry.payload, &mut parse_stats);
        let target = Target {
            id: format!("{dataset}-{source_order:06}"),
            smiles,
            inchikey,
            acceptable_routes: Vec::new(),
            annotations: serde_json::Map::new(),
        };
        let raw_route = entry.payload;
        let route = match adapter.cast(raw_route.clone(), AdaptMode::Strict, Some(&target)) {
            Ok(route) => route,
            Err(error) => {
                *failures_by_code
                    .entry(crate::adapters::error_code(&error).to_owned())
                    .or_default() += 1;
                continue;
            }
        };
        collect_reaction_hash_sanity(&route, &raw_route, &mut reaction_hash_signature_pairs);
        let patent_id = route
            .annotations
            .get("patent_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let raw_route_hash = crate::provenance::content_hash(&raw_route).map_err(|error| {
            training_error(
                "workflow.training_route_hash_failed",
                error.to_string(),
                json!({"dataset": dataset, "raw_index": source_order - 1}),
            )
        })?;
        routes.push(AdaptedTrainingRoute {
            route,
            source: RawRouteSource {
                dataset: dataset.to_owned(),
                raw_index: source_order - 1,
                raw_route_hash,
                patent_id,
            },
        });
    }
    assert_reaction_hash_matches_signature(&reaction_hash_signature_pairs)?;
    let skipped_routes = raw_routes - routes.len();
    let skipped_with_error_code: usize = failures_by_code.values().sum();
    Ok(TrainingRouteAdaptation {
        stats: AdaptationStatistics {
            raw_routes,
            adapted_routes: routes.len(),
            skipped_routes,
            skipped_without_error_code: skipped_routes - skipped_with_error_code,
            failures_by_code,
            non_fatal_condition_slot_parse: parse_stats,
        },
        routes,
    })
}

fn training_adapter_error(error: EngineError) -> TrainingError {
    training_error(
        crate::adapters::error_code(&error),
        error.to_string(),
        json!({"adapter": "paroutes"}),
    )
}

fn collect_reaction_hash_sanity(
    route: &Route,
    raw_route: &Value,
    pairs: &mut HashSet<(String, String)>,
) {
    let mut reaction_hashes = BTreeMap::new();
    collect_raw_reaction_hashes(raw_route, &mut reaction_hashes);
    for reaction in route.reactions() {
        let Some(source_id) = reaction
            .value
            .annotations
            .get("source_id")
            .and_then(Value::as_str)
        else {
            continue;
        };
        if let Some(reaction_hash) = reaction_hashes.get(source_id) {
            pairs.insert((
                reaction_hash.clone(),
                reaction.signature(InchiKeyLevel::Full),
            ));
        }
    }
}

fn collect_raw_reaction_hashes(node: &Value, output: &mut BTreeMap<String, String>) {
    if let Some(metadata) = node.get("metadata").and_then(Value::as_object)
        && let (Some(source_id), Some(reaction_hash)) = (
            metadata.get("ID").and_then(Value::as_str),
            metadata.get("reaction_hash").and_then(Value::as_str),
        )
        && !reaction_hash.is_empty()
    {
        output.insert(source_id.to_owned(), reaction_hash.to_owned());
    }
    if let Some(children) = node.get("children").and_then(Value::as_array) {
        for child in children {
            collect_raw_reaction_hashes(child, output);
        }
    }
}

fn assert_reaction_hash_matches_signature(
    pairs: &HashSet<(String, String)>,
) -> Result<(), TrainingError> {
    let hashes: HashSet<_> = pairs.iter().map(|(hash, _)| hash).collect();
    let signatures: HashSet<_> = pairs.iter().map(|(_, signature)| signature).collect();
    if pairs.len() == hashes.len() && pairs.len() == signatures.len() {
        return Ok(());
    }
    Err(training_error(
        "workflow.paroutes_reaction_hash_mismatch",
        format!(
            "PaRoutes reaction_hash is not equivalent to RetroCast reaction signatures: {} unique pairs, {} unique hashes, {} unique signatures",
            pairs.len(),
            hashes.len(),
            signatures.len(),
        ),
        json!({
            "unique_pairs": pairs.len(),
            "unique_hashes": hashes.len(),
            "unique_signatures": signatures.len(),
        }),
    ))
}

pub fn build_test_route_records(
    dataset: &str,
    routes: &[AdaptedTrainingRoute],
    route_prefix: &str,
) -> Vec<TestRouteRecord> {
    let width = 5.max(routes.len().to_string().len());
    routes
        .iter()
        .enumerate()
        .map(|(index, route)| TestRouteRecord {
            id: format!(
                "{route_prefix}-{dataset}-routes-{number:0width$}",
                number = index + 1
            ),
            dataset: dataset.to_owned(),
            route: route.route.clone(),
            sources: vec![route.source.clone()],
        })
        .collect()
}

pub fn build_test_reaction_records(
    dataset: &str,
    route_records: &[TestRouteRecord],
    route_prefix: &str,
) -> Result<Vec<TestReactionRecord>, TrainingError> {
    let mut records = Vec::new();
    for route_record in route_records {
        if route_record.dataset != dataset {
            return Err(training_error(
                "workflow.test_single_step_dataset_mismatch",
                "test single-step release route dataset does not match requested dataset",
                json!({
                    "dataset": dataset,
                    "route_id": route_record.id,
                    "route_dataset": route_record.dataset,
                }),
            ));
        }
        for (step_index, reaction) in route_record.route.reactions().into_iter().enumerate() {
            let reaction_id = reaction.id().to_string();
            let Some(mapped_smiles) = reaction.value.mapped_reaction_smiles.clone() else {
                return Err(training_error(
                    "workflow.test_single_step_missing_mapped_smiles",
                    format!(
                        "test single-step release requires mapped_reaction_smiles; missing on route {} reaction {reaction_id}",
                        route_record.id
                    ),
                    json!({"route_id": route_record.id, "reaction_id": reaction_id}),
                ));
            };
            let mapped_smiles = mapped_smiles.into_string();
            let annotations = &reaction.value.annotations;
            let condition_slot_smiles = string_list_annotation(
                annotations.get("condition_slot_smiles"),
                "condition_slot_smiles",
                &route_record.id,
                &reaction_id,
            )?;
            let alternative_mapped_smiles = string_list_annotation(
                annotations.get("alternative_mapped_smiles"),
                "alternative_mapped_smiles",
                &route_record.id,
                &reaction_id,
            )?
            .into_iter()
            .filter(|value| value != &mapped_smiles)
            .collect();
            records.push(TestReactionRecord {
                id: format!(
                    "{route_prefix}-{dataset}-single-step-reactions-{number:06}",
                    number = records.len() + 1
                ),
                dataset: dataset.to_owned(),
                reactants: reaction
                    .value
                    .reactants
                    .iter()
                    .map(|reactant| reactant.smiles.to_string())
                    .collect(),
                product: reaction.product().value.smiles.to_string(),
                mapped_smiles,
                alternative_mapped_smiles,
                condition_slot: annotations
                    .get("condition_slot")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                condition_slot_smiles,
                sources: vec![TrainingReactionSource {
                    route_id: route_record.id.clone(),
                    step_index: Some(step_index + 1),
                    reaction_id,
                    source_id: annotations
                        .get("source_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                }],
            });
        }
    }
    Ok(records)
}

fn string_list_annotation(
    value: Option<&Value>,
    key: &str,
    route_id: &str,
    reaction_id: &str,
) -> Result<Vec<String>, TrainingError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        return Err(training_error(
            &format!("workflow.test_single_step_invalid_{key}"),
            format!("test single-step release annotation '{key}' must be a list"),
            json!({"route_id": route_id, "reaction_id": reaction_id}),
        ));
    };
    Ok(values
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect())
}

fn training_error(code: &str, message: impl Into<String>, context: Value) -> TrainingError {
    TrainingError {
        code: code.to_owned(),
        message: message.into(),
        context,
    }
}

pub fn build_training_reaction_release(
    route_records: &[TrainingRouteRecord],
    config: &TrainingSetBuildConfig,
) -> Result<TrainingReactionBuildResult, TrainingError> {
    let (training, training_summary) = build_reaction_split(route_records, "training")?;
    let (validation_before, validation_summary) =
        build_reaction_split(route_records, "validation")?;
    let overlap_before = summarize_cross_split_overlap(&training, &validation_before);
    let (validation, overlap_removed) = if config.holdout_mode == "reaction" {
        let training_keys: HashSet<_> = training.iter().map(reaction_identity).collect();
        let validation: Vec<_> = validation_before
            .iter()
            .filter(|record| !training_keys.contains(&reaction_identity(record)))
            .cloned()
            .collect();
        let removed = validation_before.len() - validation.len();
        (validation, removed)
    } else {
        (validation_before, 0)
    };
    let overlap_after = summarize_cross_split_overlap(&training, &validation);
    if config.holdout_mode == "reaction"
        && (overlap_after.shared_identities != 0 || overlap_after.shared_exact != 0)
    {
        return Err(training_error(
            "workflow.single_step_validation_overlap",
            "single-step release validation split still overlaps with training after cleanup",
            overlap_after.to_json(),
        ));
    }
    let warnings = if config.holdout_mode == "route" && overlap_after.shared_identities != 0 {
        vec![
            "single-step route-holdout release has cross-split reaction identity overlap"
                .to_owned(),
        ]
    } else {
        Vec::new()
    };
    let mut records = training;
    records.extend(validation);
    renumber_reactions(&mut records, &config.route_prefix);
    let training_count = records
        .iter()
        .filter(|record| record.split == "training")
        .count();
    let validation_count = records.len() - training_count;
    let mut validation_summary = validation_summary;
    validation_summary.insert(
        "overlap_removed_from_validation".to_owned(),
        overlap_removed,
    );
    Ok(TrainingReactionBuildResult {
        release_name: format!("single-step-{}-holdout-n1-n5", config.holdout_mode),
        summary: json!({
            "reaction_postprocessing": {
                "training": training_summary,
                "validation": validation_summary,
                "cross_split_overlap_before_cleanup": overlap_before.to_json(),
                "cross_split_overlap_after_cleanup": overlap_after.to_json(),
            },
            "output": {
                "all_records": {
                    "total": records.len(),
                    "training": training_count,
                    "validation": validation_count,
                }
            }
        }),
        records,
        warnings,
    })
}

fn build_reaction_split(
    route_records: &[TrainingRouteRecord],
    split: &str,
) -> Result<(Vec<TrainingReactionRecord>, BTreeMap<String, usize>), TrainingError> {
    let flattened = flatten_reactions(route_records, split)?;
    let input_routes = route_records
        .iter()
        .filter(|record| record.split == split)
        .count();
    let flattened_count = flattened.len();
    let (exact, exact_removed) = merge_exact_reactions(flattened);
    let (transforms, mapped_removed) = merge_transform_reactions(exact);
    Ok((
        transforms,
        BTreeMap::from([
            ("input_routes".to_owned(), input_routes),
            ("flattened_reactions".to_owned(), flattened_count),
            ("chemical_duplicates_removed".to_owned(), exact_removed),
            (
                "mapped_smiles_variants_collapsed".to_owned(),
                mapped_removed,
            ),
            (
                "duplicate_reactions_removed".to_owned(),
                exact_removed + mapped_removed,
            ),
        ]),
    ))
}

fn flatten_reactions(
    route_records: &[TrainingRouteRecord],
    split: &str,
) -> Result<Vec<TrainingReactionRecord>, TrainingError> {
    let mut records = Vec::new();
    for route_record in route_records.iter().filter(|record| record.split == split) {
        for (step_index, reaction) in route_record.route.reactions().into_iter().enumerate() {
            let reaction_id = reaction.id().to_string();
            let Some(mapped_smiles) = reaction.value.mapped_reaction_smiles.clone() else {
                return Err(training_error(
                    "workflow.single_step_missing_mapped_smiles",
                    format!(
                        "single-step release requires mapped_reaction_smiles; missing on route {} reaction {reaction_id}",
                        route_record.id
                    ),
                    json!({"route_id": route_record.id, "reaction_id": reaction_id}),
                ));
            };
            let mapped_smiles = mapped_smiles.into_string();
            let annotations = &reaction.value.annotations;
            records.push(TrainingReactionRecord {
                id: "pending".to_owned(),
                split: route_record.split.clone(),
                reactants: reaction
                    .value
                    .reactants
                    .iter()
                    .map(|reactant| reactant.smiles.to_string())
                    .collect(),
                product: reaction.product().value.smiles.to_string(),
                alternative_mapped_smiles: string_values(
                    annotations.get("alternative_mapped_smiles"),
                )
                .into_iter()
                .filter(|value| value != &mapped_smiles)
                .collect(),
                mapped_smiles,
                condition_slot: annotations
                    .get("condition_slot")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                condition_slot_smiles: string_values(annotations.get("condition_slot_smiles")),
                sources: vec![TrainingReactionSource {
                    route_id: route_record.id.clone(),
                    step_index: Some(step_index + 1),
                    reaction_id,
                    source_id: annotations
                        .get("source_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                }],
            });
        }
    }
    Ok(records)
}

fn string_values(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExactReactionKey(String, Vec<String>, Option<String>);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ReactionIdentity(Vec<String>, String, Vec<String>);

fn exact_reaction_key(record: &TrainingReactionRecord) -> ExactReactionKey {
    ExactReactionKey(
        record.mapped_smiles.clone(),
        record.condition_slot_smiles.clone(),
        record.condition_slot.clone(),
    )
}

fn reaction_identity(record: &TrainingReactionRecord) -> ReactionIdentity {
    let mut reactants = record.reactants.clone();
    reactants.sort();
    ReactionIdentity(
        reactants,
        record.product.clone(),
        condition_identity(record),
    )
}

fn condition_identity(record: &TrainingReactionRecord) -> Vec<String> {
    if !record.condition_slot_smiles.is_empty() {
        record.condition_slot_smiles.clone()
    } else {
        record.condition_slot.iter().cloned().collect()
    }
}

fn merge_exact_reactions(
    records: Vec<TrainingReactionRecord>,
) -> (Vec<TrainingReactionRecord>, usize) {
    let mut output: Vec<TrainingReactionRecord> = Vec::new();
    let mut removed = 0;
    for record in records {
        let key = exact_reaction_key(&record);
        if let Some(existing) = output
            .iter_mut()
            .find(|candidate| exact_reaction_key(candidate) == key)
        {
            existing.sources.extend(record.sources);
            let mut alternatives: BTreeSet<_> = existing
                .alternative_mapped_smiles
                .iter()
                .chain(&record.alternative_mapped_smiles)
                .cloned()
                .collect();
            alternatives.remove(&existing.mapped_smiles);
            existing.alternative_mapped_smiles = alternatives.into_iter().collect();
            removed += 1;
        } else {
            output.push(record);
        }
    }
    (output, removed)
}

fn merge_transform_reactions(
    records: Vec<TrainingReactionRecord>,
) -> (Vec<TrainingReactionRecord>, usize) {
    let mut groups: Vec<(ReactionIdentity, Vec<TrainingReactionRecord>)> = Vec::new();
    for record in records {
        let identity = reaction_identity(&record);
        match groups.iter_mut().find(|(key, _)| key == &identity) {
            Some((_, group)) => group.push(record),
            None => groups.push((identity, vec![record])),
        }
    }
    let mut output = Vec::new();
    let mut removed = 0;
    for (_, group) in groups {
        removed += group.len() - 1;
        if group.len() == 1 {
            output.push(group.into_iter().next().expect("one record"));
            continue;
        }
        let mut weights: BTreeMap<String, usize> = BTreeMap::new();
        let mut all_mapped = BTreeSet::new();
        for record in &group {
            *weights.entry(record.mapped_smiles.clone()).or_default() += record.sources.len();
            all_mapped.insert(record.mapped_smiles.clone());
            all_mapped.extend(record.alternative_mapped_smiles.iter().cloned());
        }
        let canonical_mapped = weights
            .iter()
            .min_by_key(|(mapped, weight)| (std::cmp::Reverse(**weight), *mapped))
            .map(|(mapped, _)| mapped.clone())
            .expect("nonempty group");
        let mut candidates: Vec<_> = group
            .iter()
            .filter(|record| record.mapped_smiles == canonical_mapped)
            .collect();
        candidates.sort_by_key(|record| {
            record
                .sources
                .iter()
                .map(|source| source.route_id.as_str())
                .collect::<Vec<_>>()
        });
        let mut canonical = (*candidates[0]).clone();
        all_mapped.remove(&canonical_mapped);
        canonical.alternative_mapped_smiles = all_mapped.into_iter().collect();
        canonical.sources = group
            .into_iter()
            .flat_map(|record| record.sources)
            .collect();
        output.push(canonical);
    }
    (output, removed)
}

struct OverlapSummary {
    shared_exact: usize,
    shared_identities: usize,
    training_records: usize,
    validation_records: usize,
}

impl OverlapSummary {
    fn to_json(&self) -> Value {
        json!({
            "shared_exact_reaction_signatures": self.shared_exact,
            "shared_reaction_identities": self.shared_identities,
            "training_records_with_shared_identity": self.training_records,
            "validation_records_with_shared_identity": self.validation_records,
        })
    }
}

fn summarize_cross_split_overlap(
    training: &[TrainingReactionRecord],
    validation: &[TrainingReactionRecord],
) -> OverlapSummary {
    let training_exact: HashSet<_> = training.iter().map(exact_reaction_key).collect();
    let validation_exact: HashSet<_> = validation.iter().map(exact_reaction_key).collect();
    let training_identities: HashSet<_> = training.iter().map(reaction_identity).collect();
    let validation_identities: HashSet<_> = validation.iter().map(reaction_identity).collect();
    let shared: HashSet<_> = training_identities
        .intersection(&validation_identities)
        .cloned()
        .collect();
    OverlapSummary {
        shared_exact: training_exact.intersection(&validation_exact).count(),
        shared_identities: shared.len(),
        training_records: training
            .iter()
            .filter(|record| shared.contains(&reaction_identity(record)))
            .count(),
        validation_records: validation
            .iter()
            .filter(|record| shared.contains(&reaction_identity(record)))
            .count(),
    }
}

fn renumber_reactions(records: &mut [TrainingReactionRecord], prefix: &str) {
    let width = 6.max(records.len().to_string().len());
    for (index, record) in records.iter_mut().enumerate() {
        record.id = format!("{prefix}-rxn-{number:0width$}", number = index + 1);
    }
}

pub fn build_training_route_release(
    all_routes: &[AdaptedTrainingRoute],
    all_adaptation: Value,
    holdout_routes: &BTreeMap<String, Vec<AdaptedTrainingRoute>>,
    holdout_adaptation: BTreeMap<String, Value>,
    config: &TrainingSetBuildConfig,
) -> Result<TrainingSetBuildResult, TrainingError> {
    let holdouts: Vec<_> = holdout_routes.values().flatten().collect();
    let holdout_route_signatures: HashSet<_> = holdouts
        .iter()
        .map(|route| route.route.signature(InchiKeyLevel::Full, None))
        .collect();
    let holdout_reaction_signatures: HashSet<_> = if config.holdout_mode == "reaction" {
        holdouts
            .iter()
            .flat_map(|route| route.route.reaction_signatures(InchiKeyLevel::Full))
            .collect()
    } else {
        HashSet::new()
    };

    let mut candidates = Vec::new();
    let mut skipped_route_holdout = 0;
    let mut routes_with_overlapping_reactions = 0;
    let mut reaction_excision_fragments = 0;
    let mut routes_fully_removed_after_excision = 0;
    for adapted in all_routes {
        let structural_signature = adapted.route.signature(InchiKeyLevel::Full, None);
        if holdout_route_signatures.contains(&structural_signature) {
            skipped_route_holdout += 1;
            continue;
        }
        let overlaps = config.holdout_mode == "reaction"
            && adapted
                .route
                .reaction_signatures(InchiKeyLevel::Full)
                .iter()
                .any(|signature| holdout_reaction_signatures.contains(signature));
        if overlaps {
            routes_with_overlapping_reactions += 1;
            let fragments =
                curation::excise_reactions(&adapted.route, &holdout_reaction_signatures);
            if fragments.is_empty() {
                routes_fully_removed_after_excision += 1;
                continue;
            }
            reaction_excision_fragments += fragments.len();
            candidates.extend(fragments.into_iter().map(|route| PreparedTrainingRoute {
                route,
                sources: vec![adapted.source.clone()],
            }));
        } else {
            candidates.push(PreparedTrainingRoute {
                route: adapted.route.clone(),
                sources: vec![adapted.source.clone()],
            });
        }
    }

    let (exact, exact_removed) = merge_exact_routes(candidates);
    let (transforms, mapped_removed) = merge_transform_routes(exact);
    let route_values: Vec<_> = transforms.iter().map(|route| route.route.clone()).collect();
    let validation: HashSet<_> =
        validation_indices(&route_values, config.val_fraction, &config.seed.to_string())
            .map_err(|error| {
                training_error(
                    "workflow.training_split_failed",
                    error.to_string(),
                    json!({}),
                )
            })?
            .into_iter()
            .collect();
    let width = 6.max(transforms.len().to_string().len());
    let release_name = format!("{}-holdout-n1-n5", config.holdout_mode);
    let records: Vec<_> = transforms
        .into_iter()
        .enumerate()
        .map(|(index, route)| TrainingRouteRecord {
            id: format!(
                "{}-{release_name}-{number:0width$}",
                config.route_prefix,
                number = index + 1
            ),
            split: if validation.contains(&index) {
                "validation".to_owned()
            } else {
                "training".to_owned()
            },
            route: route.route,
            sources: route.sources,
        })
        .collect();

    let training_count = records
        .iter()
        .filter(|record| record.split == "training")
        .count();
    let mut depths: BTreeMap<String, usize> = BTreeMap::new();
    for record in &records {
        *depths.entry(record.route.depth().to_string()).or_default() += 1;
    }
    let mut adaptation = serde_json::Map::new();
    adaptation.insert("all_routes".to_owned(), all_adaptation);
    adaptation.extend(holdout_adaptation);
    let reaction_overlap = (config.holdout_mode == "reaction").then(|| {
        json!({
            "unique_reference_reaction_signatures": holdout_reaction_signatures.len(),
            "routes_with_overlapping_reactions": routes_with_overlapping_reactions,
            "fragments_kept_after_excision": reaction_excision_fragments,
            "routes_fully_removed_after_excision": routes_fully_removed_after_excision,
        })
    });
    Ok(TrainingSetBuildResult {
        release_name,
        summary: json!({
            "input": {"all_routes": all_routes.len()},
            "adaptation": adaptation,
            "postprocessing": {
                "exact_route_matches_removed": skipped_route_holdout,
                "duplicate_routes_removed": exact_removed + mapped_removed,
                "chemical_duplicates_removed": exact_removed,
                "mapped_smiles_variants_collapsed": mapped_removed,
                "reaction_overlap": reaction_overlap,
            },
            "output": {
                "all_records": {
                    "total": records.len(),
                    "training": training_count,
                    "validation": records.len() - training_count,
                },
                "by_depth": depths,
            }
        }),
        records,
    })
}

pub fn audit_route_release(
    release_name: &str,
    all: &[TrainingRouteRecord],
    training: &[TrainingRouteRecord],
    validation: &[TrainingRouteRecord],
) -> Result<(), TrainingError> {
    let mut failures = split_file_failures(all, training, validation, true);
    let training_identities: Vec<_> = training
        .iter()
        .map(|record| transform_route_key(&record.route))
        .collect();
    let validation_identities: Vec<_> = validation
        .iter()
        .map(|record| transform_route_key(&record.route))
        .collect();
    insert_duplicate_failure(
        &mut failures,
        "duplicate_training_route_identities",
        &training_identities,
    );
    insert_duplicate_failure(
        &mut failures,
        "duplicate_validation_route_identities",
        &validation_identities,
    );
    if failures.is_empty() {
        return Ok(());
    }
    Err(training_error(
        "workflow.route_release_audit_failed",
        format!("{release_name} failed route release sanity checks"),
        json!({"release_name": release_name, "failures": failures}),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn audit_single_step_release(
    release_name: &str,
    all: &[TrainingReactionRecord],
    training: &[TrainingReactionRecord],
    validation: &[TrainingReactionRecord],
    all_rsmi_count: usize,
    training_rsmi_count: usize,
    validation_rsmi_count: usize,
    parent_route_ids: &HashSet<String>,
) -> Result<Value, TrainingError> {
    let mut failures = split_file_failures(all, training, validation, false);
    let count_deltas = json!({
        "all": all.len() as isize - all_rsmi_count as isize,
        "training": training.len() as isize - training_rsmi_count as isize,
        "validation": validation.len() as isize - validation_rsmi_count as isize,
    });
    if count_deltas
        .as_object()
        .expect("object")
        .values()
        .any(|value| value.as_i64() != Some(0))
    {
        failures.insert("rsmi_count_mismatches".to_owned(), count_deltas);
    }
    let training_exact: Vec<_> = training.iter().map(exact_reaction_key).collect();
    let validation_exact: Vec<_> = validation.iter().map(exact_reaction_key).collect();
    insert_duplicate_failure(
        &mut failures,
        "duplicate_training_exact_reaction_keys",
        &training_exact,
    );
    insert_duplicate_failure(
        &mut failures,
        "duplicate_validation_exact_reaction_keys",
        &validation_exact,
    );
    let training_identities: HashSet<_> = training.iter().map(reaction_identity).collect();
    let validation_identities: HashSet<_> = validation.iter().map(reaction_identity).collect();
    let shared = training_identities
        .intersection(&validation_identities)
        .count();
    if shared > 0 && release_name != "single-step-route-holdout-n1-n5" {
        failures.insert(
            "cross_split_reaction_identity_overlap".to_owned(),
            json!(shared),
        );
    }
    let missing_sources = all
        .iter()
        .flat_map(|record| &record.sources)
        .filter(|source| !parent_route_ids.contains(&source.route_id))
        .count();
    if missing_sources > 0 {
        failures.insert(
            "missing_parent_route_sources".to_owned(),
            json!(missing_sources),
        );
    }
    if !failures.is_empty() {
        return Err(training_error(
            "workflow.single_step_release_audit_failed",
            format!("{release_name} failed single-step sanity checks"),
            json!({"release_name": release_name, "failures": failures}),
        ));
    }
    Ok(json!({
        "release_name": release_name,
        "total": all.len(),
        "training": training.len(),
        "validation": validation.len(),
        "parent_routes": parent_route_ids.len(),
        "cross_split_reaction_identity_overlap": shared,
    }))
}

fn split_file_failures<T: AuditRecord>(
    all: &[T],
    training: &[T],
    validation: &[T],
    check_split_duplicate_ids: bool,
) -> serde_json::Map<String, Value> {
    let all_ids: HashSet<_> = all.iter().map(AuditRecord::id).collect();
    let split_ids: HashSet<_> = training
        .iter()
        .chain(validation)
        .map(AuditRecord::id)
        .collect();
    let split_count = training.len() + validation.len();
    let mut failures = serde_json::Map::new();
    if all_ids.len() != all.len() {
        failures.insert(
            "duplicate_record_ids_in_all".to_owned(),
            json!(all.len() - all_ids.len()),
        );
    }
    if check_split_duplicate_ids && split_ids.len() != split_count {
        failures.insert(
            "duplicate_record_ids_in_splits".to_owned(),
            json!(split_count - split_ids.len()),
        );
    }
    let id_delta = all_ids.symmetric_difference(&split_ids).count();
    if id_delta > 0 {
        failures.insert("all_split_id_delta".to_owned(), json!(id_delta));
    }
    let wrong_training = training
        .iter()
        .filter(|record| record.split() != "training")
        .count();
    if wrong_training > 0 {
        failures.insert(
            "training_file_wrong_split_records".to_owned(),
            json!(wrong_training),
        );
    }
    let wrong_validation = validation
        .iter()
        .filter(|record| record.split() != "validation")
        .count();
    if wrong_validation > 0 {
        failures.insert(
            "validation_file_wrong_split_records".to_owned(),
            json!(wrong_validation),
        );
    }
    failures
}

trait AuditRecord {
    fn id(&self) -> &str;
    fn split(&self) -> &str;
}

impl AuditRecord for TrainingRouteRecord {
    fn id(&self) -> &str {
        &self.id
    }

    fn split(&self) -> &str {
        &self.split
    }
}

impl AuditRecord for TrainingReactionRecord {
    fn id(&self) -> &str {
        &self.id
    }

    fn split(&self) -> &str {
        &self.split
    }
}

fn insert_duplicate_failure<T: std::hash::Hash + Eq>(
    failures: &mut serde_json::Map<String, Value>,
    name: &str,
    values: &[T],
) {
    let unique: HashSet<_> = values.iter().collect();
    if unique.len() != values.len() {
        failures.insert(name.to_owned(), json!(values.len() - unique.len()));
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReactionProfile {
    signature: String,
    mapped: Option<String>,
    condition: ConditionIdentity,
}

type ConditionIdentity = Option<(u8, Vec<String>)>;
type TransformRouteKey = (String, Vec<(String, ConditionIdentity)>);

fn reaction_profiles(route: &Route) -> Vec<ReactionProfile> {
    route
        .reactions()
        .into_iter()
        .map(|reaction| ReactionProfile {
            signature: reaction.signature(InchiKeyLevel::Full),
            mapped: reaction
                .value
                .mapped_reaction_smiles
                .as_ref()
                .map(ToString::to_string),
            condition: route_condition_identity(&reaction.value.annotations),
        })
        .collect()
}

fn route_condition_identity(annotations: &serde_json::Map<String, Value>) -> ConditionIdentity {
    let mut smiles = string_values(annotations.get("condition_slot_smiles"));
    if !smiles.is_empty() {
        smiles.sort();
        return Some((0, smiles));
    }
    annotations
        .get("condition_slot")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(|value| (1, vec![value.to_owned()]))
}

fn exact_route_key(route: &Route) -> (String, Vec<ReactionProfile>) {
    let mut profiles = reaction_profiles(route);
    profiles.sort();
    (route.signature(InchiKeyLevel::Full, None), profiles)
}

fn transform_route_key(route: &Route) -> TransformRouteKey {
    let mut profiles: Vec<_> = reaction_profiles(route)
        .into_iter()
        .map(|profile| (profile.signature, profile.condition))
        .collect();
    profiles.sort();
    (route.signature(InchiKeyLevel::Full, None), profiles)
}

fn merge_exact_routes(routes: Vec<PreparedTrainingRoute>) -> (Vec<PreparedTrainingRoute>, usize) {
    let mut output: Vec<PreparedTrainingRoute> = Vec::new();
    let mut removed = 0;
    for route in routes {
        let key = exact_route_key(&route.route);
        if let Some(existing) = output
            .iter_mut()
            .find(|candidate| exact_route_key(&candidate.route) == key)
        {
            existing.sources.extend(route.sources);
            removed += 1;
        } else {
            output.push(route);
        }
    }
    (output, removed)
}

fn merge_transform_routes(
    routes: Vec<PreparedTrainingRoute>,
) -> (Vec<PreparedTrainingRoute>, usize) {
    let mut groups: Vec<(TransformRouteKey, Vec<PreparedTrainingRoute>)> = Vec::new();
    for route in routes {
        let key = transform_route_key(&route.route);
        match groups.iter_mut().find(|(candidate, _)| candidate == &key) {
            Some((_, group)) => group.push(route),
            None => groups.push((key, vec![route])),
        }
    }
    let mut output = Vec::new();
    let mut removed = 0;
    for (_, group) in groups {
        removed += group.len() - 1;
        let canonical_index = choose_canonical_route(&group);
        let canonical = &group[canonical_index];
        output.push(PreparedTrainingRoute {
            route: route_with_merged_annotations(&canonical.route, &group),
            sources: group
                .iter()
                .flat_map(|route| route.sources.iter().cloned())
                .collect(),
        });
    }
    (output, removed)
}

fn choose_canonical_route(group: &[PreparedTrainingRoute]) -> usize {
    let mut support: BTreeMap<Vec<ReactionProfile>, usize> = BTreeMap::new();
    for route in group {
        let mut profile = reaction_profiles(&route.route);
        profile.sort();
        *support.entry(profile).or_default() += route.sources.len();
    }
    group
        .iter()
        .enumerate()
        .min_by_key(|(_, route)| {
            let mut profile = reaction_profiles(&route.route);
            profile.sort();
            let first_hash = route
                .sources
                .iter()
                .map(|source| source.raw_route_hash.as_str())
                .min()
                .unwrap_or("");
            (std::cmp::Reverse(support[&profile]), profile, first_hash)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn route_with_merged_annotations(route: &Route, group: &[PreparedTrainingRoute]) -> Route {
    let mut alternatives: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for variant in group {
        for profile in reaction_profiles(&variant.route) {
            if let Some(mapped) = profile.mapped {
                alternatives
                    .entry(profile.signature)
                    .or_default()
                    .insert(mapped);
            }
        }
    }
    let mut patents: BTreeSet<String> = BTreeSet::new();
    for source in group.iter().flat_map(|route| &route.sources) {
        patents.extend(source.patent_id.iter().cloned());
    }
    let target = annotate_route_molecule(&route.target, &RoutePath::target(), route, &alternatives);
    let mut annotations = route.annotations.clone();
    annotations.remove("patent_id");
    if !patents.is_empty() {
        annotations.insert(
            "source_patent_ids".to_owned(),
            Value::Array(patents.into_iter().map(Value::String).collect()),
        );
    }
    Route {
        target,
        annotations,
        schema_version: Default::default(),
    }
}

fn annotate_route_molecule(
    molecule: &Molecule,
    path: &RoutePath,
    route: &Route,
    alternatives: &BTreeMap<String, BTreeSet<String>>,
) -> Molecule {
    let Some(reaction) = molecule.product_of.as_deref() else {
        return molecule.clone();
    };
    let reaction_path = path.produced_by().expect("molecule path");
    let signature = route
        .reaction_at(&reaction_path)
        .expect("path belongs to route")
        .signature(InchiKeyLevel::Full);
    let mut rebuilt = molecule.clone();
    let rebuilt_reaction = rebuilt.product_of.as_deref_mut().expect("cloned reaction");
    rebuilt_reaction.reactants = reaction
        .reactants
        .iter()
        .enumerate()
        .map(|(index, reactant)| {
            annotate_route_molecule(
                reactant,
                &reaction_path.reactant(index).expect("reaction path"),
                route,
                alternatives,
            )
        })
        .collect();
    let alternatives: Vec<_> = alternatives
        .get(&signature)
        .into_iter()
        .flatten()
        .filter(|mapped| Some(mapped.as_str()) != reaction.mapped_reaction_smiles.as_deref())
        .cloned()
        .map(Value::String)
        .collect();
    if !alternatives.is_empty() {
        rebuilt_reaction.annotations.insert(
            "alternative_mapped_smiles".to_owned(),
            Value::Array(alternatives),
        );
    }
    rebuilt
}

fn default_validation_fraction() -> f64 {
    0.05
}

fn default_training_seed() -> i64 {
    20_260_502
}

fn default_route_prefix() -> String {
    "paroutes".to_owned()
}

/// Choose a validation subset stratified by route depth and convergence.
///
/// Groups retain first-observation order, matching Python's ordered dictionaries,
/// and share a CPython-compatible RNG stream across strata.
pub fn validation_indices(
    routes: &[Route],
    validation_fraction: f64,
    seed: &str,
) -> Result<Vec<usize>, sampling::SamplingError> {
    let mut strata: Vec<((usize, bool), Vec<usize>)> = Vec::new();
    for (index, route) in routes.iter().enumerate() {
        let key = (route.depth(), route.is_convergent());
        match strata.iter_mut().find(|(candidate, _)| *candidate == key) {
            Some((_, indices)) => indices.push(index),
            None => strata.push((key, vec![index])),
        }
    }

    let groups: Vec<_> = strata.into_iter().map(|(_, indices)| indices).collect();
    let sample_sizes: Vec<_> = groups
        .iter()
        .map(|indices| {
            let count = (indices.len() as f64 * validation_fraction).round_ties_even();
            if count > 0.0 { count as usize } else { 0 }
        })
        .collect();
    Ok(sampling::sample_index_groups(&groups, &sample_sizes, seed)?
        .into_iter()
        .flatten()
        .collect())
}
