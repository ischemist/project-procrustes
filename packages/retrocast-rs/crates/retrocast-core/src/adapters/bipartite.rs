use std::collections::HashSet;

use serde_json::{Map, Value, json};

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::Result,
    model::{Molecule, Reaction, Route, Target},
    route::{AdaptMode, normalize_reactants},
    schema::{CanonicalSmiles, ReactionSmiles},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct SyntheseusAdapter;

#[derive(Clone, Copy, Debug, Default)]
pub struct SynPlannerAdapter;

#[derive(Clone, Copy)]
enum Flavor {
    Syntheseus,
    SynPlanner,
}

impl Adapter for SyntheseusAdapter {
    fn name(&self) -> &'static str {
        "syntheseus"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let routes = payload
            .as_array()
            .ok_or_else(|| common::schema(self.name(), "route payload must be a list"))?;
        Ok(make_entries(routes, source_key))
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        cast_bipartite(raw_route, mode, target, self.name(), Flavor::Syntheseus)
    }
}

impl Adapter for SynPlannerAdapter {
    fn name(&self) -> &'static str {
        "synplanner"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let routes = payload
            .as_array()
            .ok_or_else(|| common::schema(self.name(), "route payload must be a list"))?;
        Ok(make_entries(routes, source_key))
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        cast_bipartite(raw_route, mode, target, self.name(), Flavor::SynPlanner)
    }
}

fn make_entries(routes: &[Value], source_key: Option<&str>) -> Vec<RawRouteEntry> {
    routes
        .iter()
        .cloned()
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
        .collect()
}

fn validate_node(
    node: &Value,
    expected_kind: &str,
    adapter: &'static str,
    path: &str,
) -> Result<()> {
    let object = node
        .as_object()
        .ok_or_else(|| common::schema(adapter, format!("{path}: node must be an object")))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| common::schema(adapter, format!("{path}.type: field is required")))?;
    if kind != expected_kind {
        return Err(common::schema(
            adapter,
            format!("{path}.type: expected {expected_kind:?}, got {kind:?}"),
        ));
    }
    if object.get("smiles").and_then(Value::as_str).is_none() {
        return Err(common::schema(
            adapter,
            format!("{path}.smiles: string field is required"),
        ));
    }
    let children = object
        .get("children")
        .map(|children| {
            children
                .as_array()
                .ok_or_else(|| common::schema(adapter, format!("{path}.children: expected a list")))
        })
        .transpose()?
        .map(Vec::as_slice)
        .unwrap_or_default();
    let child_kind = if expected_kind == "mol" {
        "reaction"
    } else {
        "mol"
    };
    for (index, child) in children.iter().enumerate() {
        validate_node(
            child,
            child_kind,
            adapter,
            &format!("{path}.children.{index}"),
        )?;
    }
    Ok(())
}

fn cast_bipartite(
    raw_route: Value,
    mode: AdaptMode,
    target: Option<&Target>,
    adapter: &'static str,
    flavor: Flavor,
) -> Result<Route> {
    validate_node(&raw_route, "mol", adapter, "route root")?;
    let target_molecule = build_molecule(&raw_route, mode, adapter, flavor, &HashSet::new())?
        .ok_or_else(|| common::target_pruned(adapter))?;
    if let Some(target) = target {
        let expected = match flavor {
            Flavor::Syntheseus => chem::normalize(&target.smiles)?.0,
            Flavor::SynPlanner => chem::normalize_unmapped(&target.smiles)?.0,
        };
        if target_molecule.smiles != expected {
            return Err(crate::error::EngineError::TargetMismatch {
                adapter,
                target_id: target.id.clone(),
                expected: expected.to_string(),
                actual: target_molecule.smiles.to_string(),
            });
        }
    }
    Ok(Route {
        target: target_molecule,
        annotations: Map::new(),
        schema_version: Default::default(),
    })
}

fn build_molecule(
    node: &Value,
    mode: AdaptMode,
    adapter: &'static str,
    flavor: Flavor,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Molecule>> {
    let object = node.as_object().expect("validated molecule object");
    let raw_smiles = object["smiles"].as_str().expect("validated smiles");
    let identity = match flavor {
        Flavor::Syntheseus => chem::normalize(raw_smiles),
        Flavor::SynPlanner => chem::normalize_unmapped(raw_smiles),
    };
    let (smiles, inchikey) = match identity {
        Ok(identity) => identity,
        Err(_) if mode == AdaptMode::Prune => return Ok(None),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(common::logic(
            adapter,
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    let children = object
        .get("children")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if object
        .get("in_stock")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || children.is_empty()
    {
        return Ok(Some(Molecule {
            smiles,
            inchikey,
            product_of: None,
            annotations: Map::new(),
        }));
    }
    if children.len() != 1 {
        return Err(common::logic(
            adapter,
            "adapter.route_not_tree",
            format!("molecule {smiles:?} has {} child reactions", children.len()),
        ));
    }
    let reaction_node = children[0].as_object().expect("validated reaction object");
    let mut next_visited = visited.clone();
    next_visited.insert(smiles.clone());
    let mut reactants = Vec::new();
    for child in reaction_node
        .get("children")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        if let Some(reactant) = build_molecule(child, mode, adapter, flavor, &next_visited)? {
            reactants.push(reactant);
        }
    }
    if reactants.is_empty() {
        return if mode == AdaptMode::Prune {
            Ok(None)
        } else {
            Err(common::logic(
                adapter,
                "adapter.reaction_empty",
                format!("reaction for {smiles:?} has no reactants"),
            ))
        };
    }
    normalize_reactants(&mut reactants);
    let metadata = reaction_node
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let source_smiles = reaction_node["smiles"]
        .as_str()
        .expect("validated reaction smiles");
    let (mapped_reaction_smiles, template, annotations) = match flavor {
        Flavor::Syntheseus => (
            metadata
                .get("mapped_reaction_smiles")
                .and_then(Value::as_str)
                .map(str::to_owned),
            metadata
                .get("template")
                .and_then(Value::as_str)
                .map(str::to_owned),
            metadata,
        ),
        Flavor::SynPlanner => (
            Some(source_smiles.to_owned()),
            None,
            Map::from_iter([("source_smiles".to_owned(), json!(source_smiles))]),
        ),
    };
    let mapped_reaction_smiles = mapped_reaction_smiles
        .map(ReactionSmiles::try_from)
        .transpose()?;
    Ok(Some(Molecule {
        smiles,
        inchikey,
        product_of: Some(Box::new(Reaction {
            reactants,
            mapped_reaction_smiles,
            template,
            reagents: None,
            solvents: None,
            annotations,
        })),
        annotations: Map::new(),
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{SynPlannerAdapter, SyntheseusAdapter};
    use crate::{
        adapters::{Adapter, adapt_candidates},
        model::Target,
        route::AdaptMode,
    };

    struct CandidateCase {
        name: &'static str,
        payload: Value,
        mode: AdaptMode,
        max_candidates: Option<usize>,
        expected_failure_codes: Vec<Option<&'static str>>,
    }

    fn candidate_cases() -> Vec<CandidateCase> {
        let valid = json!([
            {"type": "mol", "smiles": "CCO"},
            {"type": "mol", "smiles": "OCC"}
        ]);
        let mixed = json!([
            {"type": "mol", "smiles": "CCO"},
            {"type": "mol", "smiles": "CCO", "children": [null]},
            {"type": "mol", "smiles": "not-smiles"}
        ]);
        let all_invalid = json!([
            {"type": "mol", "smiles": "CCO", "children": [true]},
            {"type": "mol", "smiles": "not-smiles"}
        ]);
        let truncated = json!([
            {"type": "mol", "smiles": "CCO", "children": [null]},
            {"type": "mol", "smiles": "OCC"},
            {"type": "mol", "smiles": "not-smiles"}
        ]);

        vec![
            CandidateCase {
                name: "all-valid strict",
                payload: valid.clone(),
                mode: AdaptMode::Strict,
                max_candidates: None,
                expected_failure_codes: vec![None, None],
            },
            CandidateCase {
                name: "all-valid prune",
                payload: valid,
                mode: AdaptMode::Prune,
                max_candidates: None,
                expected_failure_codes: vec![None, None],
            },
            CandidateCase {
                name: "mixed strict",
                payload: mixed.clone(),
                mode: AdaptMode::Strict,
                max_candidates: None,
                expected_failure_codes: vec![
                    None,
                    Some("adapter.schema_invalid"),
                    Some("chem.invalid_smiles"),
                ],
            },
            CandidateCase {
                name: "mixed prune",
                payload: mixed,
                mode: AdaptMode::Prune,
                max_candidates: None,
                expected_failure_codes: vec![
                    None,
                    Some("adapter.schema_invalid"),
                    Some("adapter.target_pruned"),
                ],
            },
            CandidateCase {
                name: "all-invalid strict",
                payload: all_invalid.clone(),
                mode: AdaptMode::Strict,
                max_candidates: None,
                expected_failure_codes: vec![
                    Some("adapter.schema_invalid"),
                    Some("chem.invalid_smiles"),
                ],
            },
            CandidateCase {
                name: "all-invalid prune",
                payload: all_invalid,
                mode: AdaptMode::Prune,
                max_candidates: None,
                expected_failure_codes: vec![
                    Some("adapter.schema_invalid"),
                    Some("adapter.target_pruned"),
                ],
            },
            CandidateCase {
                name: "max-candidates strict",
                payload: truncated.clone(),
                mode: AdaptMode::Strict,
                max_candidates: Some(2),
                expected_failure_codes: vec![Some("adapter.schema_invalid"), None],
            },
            CandidateCase {
                name: "max-candidates prune",
                payload: truncated,
                mode: AdaptMode::Prune,
                max_candidates: Some(2),
                expected_failure_codes: vec![Some("adapter.schema_invalid"), None],
            },
        ]
    }

    fn fixture_target() -> Target {
        serde_json::from_value(json!({
            "id": "target-ethanol",
            "smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        }))
        .unwrap()
    }

    #[test]
    fn syntheseus_preserves_reaction_metadata() {
        let route = SyntheseusAdapter
            .cast(
                json!({
                    "type": "mol", "smiles": "CCO", "children": [{
                        "type": "reaction", "smiles": "CCO",
                        "metadata": {"template": "tmpl", "mapped_reaction_smiles": "C>>CCO"},
                        "children": [{"type": "mol", "smiles": "C"}]
                    }]
                }),
                AdaptMode::Strict,
                None,
            )
            .unwrap();
        let reaction = route.target.product_of.unwrap();
        assert_eq!(reaction.template.as_deref(), Some("tmpl"));
        assert_eq!(reaction.annotations["template"], "tmpl");
    }

    #[test]
    fn bipartite_candidate_slot_matrix() {
        let target = fixture_target();
        for adapter in [&SynPlannerAdapter as &dyn Adapter, &SyntheseusAdapter] {
            for case in candidate_cases() {
                let candidates = adapt_candidates(
                    case.payload,
                    adapter,
                    case.mode,
                    Some(&target),
                    Some("source-ethanol"),
                    case.max_candidates,
                )
                .unwrap_or_else(|error| {
                    panic!("{} / {} failed: {error}", adapter.name(), case.name)
                });

                assert_eq!(
                    candidates.len(),
                    case.expected_failure_codes.len(),
                    "{} / {} candidate count",
                    adapter.name(),
                    case.name
                );
                for (index, (candidate, expected_failure_code)) in candidates
                    .iter()
                    .zip(&case.expected_failure_codes)
                    .enumerate()
                {
                    assert_eq!(
                        candidate.rank,
                        index + 1,
                        "{} / {} rank",
                        adapter.name(),
                        case.name
                    );
                    match expected_failure_code {
                        None => {
                            let route = candidate.route.as_ref().unwrap_or_else(|| {
                                panic!(
                                    "{} / {} rank {} should be a route",
                                    adapter.name(),
                                    case.name,
                                    candidate.rank
                                )
                            });
                            assert_eq!(route.target.smiles, "CCO");
                            assert!(candidate.failure.is_none());
                        }
                        Some(expected_code) => {
                            let failure = candidate.failure.as_ref().unwrap_or_else(|| {
                                panic!(
                                    "{} / {} rank {} should be a failure",
                                    adapter.name(),
                                    case.name,
                                    candidate.rank
                                )
                            });
                            assert_eq!(failure.code, *expected_code);
                            assert_eq!(failure.target_id.as_deref(), Some("target-ethanol"));
                            assert!(candidate.route.is_none());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bipartite_adapters_reject_non_list_envelopes() {
        for adapter in [&SynPlannerAdapter as &dyn Adapter, &SyntheseusAdapter] {
            let error = adapt_candidates(
                json!({"not": "a route list"}),
                adapter,
                AdaptMode::Strict,
                None,
                None,
                None,
            )
            .unwrap_err();
            assert!(error.to_string().contains("route payload must be a list"));
        }
    }

    #[test]
    fn synplanner_removes_atom_mapping_from_molecules() {
        let route = SynPlannerAdapter
            .cast(
                json!({"type": "mol", "smiles": "[CH3:1][CH2:2]O"}),
                AdaptMode::Strict,
                None,
            )
            .unwrap();
        assert_eq!(route.target.smiles, "CCO");
    }
}
