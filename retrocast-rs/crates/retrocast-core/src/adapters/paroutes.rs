use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::Result,
    model::{Molecule, Reaction, Route, Target},
    route::{AdaptMode, normalize_reactants},
    schema::{CanonicalSmiles, ReactionSmiles},
};

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ConditionSlotParseStatistics {
    pub malformed_rsmi_count: usize,
    pub uncanonicalizable_token_count: usize,
    pub uncanonicalizable_tokens: BTreeMap<String, usize>,
}

impl ConditionSlotParseStatistics {
    pub fn distinct_uncanonicalizable_token_count(&self) -> usize {
        self.uncanonicalizable_tokens.len()
    }

    pub fn top_uncanonicalizable_tokens(&self) -> Vec<(&str, usize)> {
        let mut tokens: Vec<_> = self
            .uncanonicalizable_tokens
            .iter()
            .map(|(token, count)| (token.as_str(), *count))
            .collect();
        tokens.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
        tokens.truncate(5);
        tokens
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PaRoutesAdapter;

impl Adapter for PaRoutesAdapter {
    fn name(&self) -> &'static str {
        "paroutes"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        match payload {
            Value::Array(routes) => routes
                .into_iter()
                .enumerate()
                .map(|(index, route)| {
                    validate_molecule(&route, self.name(), "route root")?;
                    Ok(RawRouteEntry {
                        payload: route,
                        source_key: source_key.map(str::to_owned),
                        source_row_index: Some(index + 1),
                        source_record_id: None,
                        target_hint_id: None,
                        target_hint_smiles: None,
                        source_order: Some(index + 1),
                    })
                })
                .collect(),
            Value::Object(routes) if looks_like_route(&routes) => {
                let route = Value::Object(routes);
                validate_molecule(&route, self.name(), "route root")?;
                Ok(vec![RawRouteEntry {
                    payload: route,
                    source_key: source_key.map(str::to_owned),
                    source_row_index: None,
                    source_record_id: None,
                    target_hint_id: None,
                    target_hint_smiles: None,
                    source_order: Some(1),
                }])
            }
            Value::Object(routes) => routes
                .into_iter()
                .enumerate()
                .map(|(index, (target_id, route))| {
                    validate_molecule(&route, self.name(), "route root")?;
                    Ok(RawRouteEntry {
                        payload: route,
                        source_key: Some(target_id),
                        source_row_index: None,
                        source_record_id: None,
                        target_hint_id: None,
                        target_hint_smiles: None,
                        source_order: Some(index + 1),
                    })
                })
                .collect(),
            _ => Err(common::schema(
                self.name(),
                "expected route root, route list, or target route mapping",
            )),
        }
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        validate_molecule(&raw_route, self.name(), "route root")?;
        let patent_ids = collect_patent_ids(&raw_route, mode, &HashSet::new())?;
        if patent_ids.is_empty() {
            return Err(common::logic(
                self.name(),
                "adapter.patent_id_missing",
                "route does not contain a patent id",
            ));
        }
        if patent_ids.len() > 1 {
            return Err(common::logic(
                self.name(),
                "adapter.multiple_patents",
                format!("route contains reactions from multiple patents: {patent_ids:?}"),
            ));
        }
        let target_molecule = build_molecule(&raw_route, mode, &HashSet::new())?
            .ok_or_else(|| common::target_pruned(self.name()))?;
        if let Some(target) = target {
            common::require_target_match(
                &target_molecule.smiles,
                &target.smiles,
                &target.id,
                self.name(),
            )?;
        }
        Ok(Route {
            target: target_molecule,
            annotations: Map::from_iter([(
                "patent_id".to_owned(),
                json!(patent_ids.first().expect("patent set is nonempty")),
            )]),
            schema_version: Default::default(),
        })
    }
}

fn looks_like_route(object: &Map<String, Value>) -> bool {
    object.get("type").and_then(Value::as_str) == Some("mol")
        || object.contains_key("smiles")
        || object.contains_key("children")
}

fn validate_molecule(node: &Value, adapter: &'static str, path: &str) -> Result<()> {
    let object = node
        .as_object()
        .ok_or_else(|| common::schema(adapter, format!("{path}: molecule must be an object")))?;
    if object.get("type").and_then(Value::as_str) != Some("mol") {
        return Err(common::schema(
            adapter,
            format!("{path}.type: expected 'mol'"),
        ));
    }
    if object.get("smiles").and_then(Value::as_str).is_none() {
        return Err(common::schema(
            adapter,
            format!("{path}.smiles: string field is required"),
        ));
    }
    let children = children(object, adapter, path)?;
    for (index, reaction) in children.iter().enumerate() {
        validate_reaction(reaction, adapter, &format!("{path}.children.{index}"))?;
    }
    Ok(())
}

fn validate_reaction(node: &Value, adapter: &'static str, path: &str) -> Result<()> {
    let object = node
        .as_object()
        .ok_or_else(|| common::schema(adapter, format!("{path}: reaction must be an object")))?;
    if object.get("type").and_then(Value::as_str) != Some("reaction") {
        return Err(common::logic(
            adapter,
            "adapter.node_type_invalid",
            format!("{path}: expected reaction node"),
        ));
    }
    if object.get("smiles").and_then(Value::as_str).is_none() {
        return Err(common::schema(
            adapter,
            format!("{path}.smiles: string field is required"),
        ));
    }
    let metadata = object
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| common::schema(adapter, format!("{path}.metadata: object is required")))?;
    if metadata.get("ID").and_then(Value::as_str).is_none() {
        return Err(common::schema(
            adapter,
            format!("{path}.metadata.ID: string field is required"),
        ));
    }
    for (index, molecule) in children(object, adapter, path)?.iter().enumerate() {
        validate_molecule(molecule, adapter, &format!("{path}.children.{index}"))?;
    }
    Ok(())
}

fn children<'a>(
    object: &'a Map<String, Value>,
    adapter: &'static str,
    path: &str,
) -> Result<&'a [Value]> {
    object
        .get("children")
        .map(|children| {
            children
                .as_array()
                .map(Vec::as_slice)
                .ok_or_else(|| common::schema(adapter, format!("{path}.children: expected list")))
        })
        .transpose()
        .map(|children| children.unwrap_or_default())
}

fn collect_patent_ids(
    molecule: &Value,
    mode: AdaptMode,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<BTreeSet<String>> {
    let object = molecule.as_object().expect("validated molecule");
    let raw_smiles = object["smiles"].as_str().expect("validated smiles");
    let smiles = match chem::normalize(raw_smiles) {
        Ok((smiles, _)) => smiles,
        Err(_) if mode == AdaptMode::Prune => return Ok(BTreeSet::new()),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(common::logic(
            "paroutes",
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    let reactions = children(object, "paroutes", "molecule")?;
    let Some(reaction) = reactions.first() else {
        return Ok(BTreeSet::new());
    };
    let reaction = reaction.as_object().expect("validated reaction");
    let source_id = reaction["metadata"]["ID"]
        .as_str()
        .expect("validated patent ID");
    let patent_id = source_id.split(';').next().unwrap_or_default().trim();
    if patent_id.is_empty() {
        return Err(common::logic(
            "paroutes",
            "adapter.patent_id_missing",
            format!("empty patent id in source {source_id:?}"),
        ));
    }
    let mut patent_ids = BTreeSet::from([patent_id.to_owned()]);
    let mut next_visited = visited.clone();
    next_visited.insert(smiles);
    for reactant in children(reaction, "paroutes", "reaction")? {
        patent_ids.extend(collect_patent_ids(reactant, mode, &next_visited)?);
    }
    Ok(patent_ids)
}

fn build_molecule(
    molecule: &Value,
    mode: AdaptMode,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Molecule>> {
    let object = molecule.as_object().expect("validated molecule");
    let raw_smiles = object["smiles"].as_str().expect("validated smiles");
    let (smiles, inchikey) = match chem::normalize(raw_smiles) {
        Ok(identity) => identity,
        Err(_) if mode == AdaptMode::Prune => return Ok(None),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(common::logic(
            "paroutes",
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    let reactions = children(object, "paroutes", "molecule")?;
    if object
        .get("in_stock")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || reactions.is_empty()
    {
        return Ok(Some(Molecule {
            smiles,
            inchikey,
            product_of: None,
            annotations: Map::new(),
        }));
    }
    let reaction = reactions[0].as_object().expect("validated reaction");
    let mut next_visited = visited.clone();
    next_visited.insert(smiles.clone());
    let mut reactants = Vec::new();
    for reactant in children(reaction, "paroutes", "reaction")? {
        if let Some(reactant) = build_molecule(reactant, mode, &next_visited)? {
            reactants.push(reactant);
        }
    }
    if reactants.is_empty() {
        return if mode == AdaptMode::Prune {
            Ok(None)
        } else {
            Err(common::logic(
                "paroutes",
                "adapter.reaction_empty",
                format!("reaction for {smiles:?} has no reactants"),
            ))
        };
    }
    normalize_reactants(&mut reactants);
    let metadata = reaction["metadata"]
        .as_object()
        .expect("validated metadata");
    let source_id = metadata["ID"].as_str().expect("validated source id");
    let rsmi = metadata.get("rsmi").and_then(Value::as_str);
    let mapped_reaction_smiles = rsmi.map(ReactionSmiles::try_from).transpose()?;
    Ok(Some(Molecule {
        smiles,
        inchikey,
        product_of: Some(Box::new(Reaction {
            reactants,
            mapped_reaction_smiles,
            template: None,
            reagents: None,
            solvents: None,
            annotations: condition_annotations(
                source_id,
                rsmi,
                metadata.get("RingBreaker").and_then(Value::as_bool),
                None,
            ),
        })),
        annotations: Map::new(),
    }))
}

fn condition_annotations(
    source_id: &str,
    rsmi: Option<&str>,
    ring_breaker: Option<bool>,
    mut statistics: Option<&mut ConditionSlotParseStatistics>,
) -> Map<String, Value> {
    let mut annotations = Map::from_iter([("source_id".to_owned(), json!(source_id))]);
    if let Some(ring_breaker) = ring_breaker {
        annotations.insert("ring_breaker".to_owned(), json!(ring_breaker));
    }
    let Some(rsmi) = rsmi else {
        return annotations;
    };
    let parts: Vec<_> = rsmi.split('>').collect();
    if parts.len() != 3 {
        if let Some(statistics) = statistics.as_deref_mut() {
            statistics.malformed_rsmi_count += 1;
        }
        return annotations;
    }
    let slot = parts[1].trim();
    if slot.is_empty() {
        return annotations;
    }
    annotations.insert("condition_slot".to_owned(), json!(slot));
    let mut parsed = Vec::new();
    for token in slot
        .split('.')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        match chem::normalize_unmapped(token) {
            Ok((smiles, _)) => parsed.push(smiles),
            Err(_) => {
                if let Some(statistics) = statistics.as_deref_mut() {
                    statistics.uncanonicalizable_token_count += 1;
                    *statistics
                        .uncanonicalizable_tokens
                        .entry(token.to_owned())
                        .or_default() += 1;
                }
            }
        }
    }
    parsed.sort();
    if !parsed.is_empty() {
        annotations.insert("condition_slot_smiles".to_owned(), json!(parsed));
    }
    annotations
}

pub fn analyze_condition_slots(route: &Value, statistics: &mut ConditionSlotParseStatistics) {
    fn visit(node: &Value, statistics: &mut ConditionSlotParseStatistics) {
        let Some(children) = node
            .as_object()
            .and_then(|object| object.get("children"))
            .and_then(Value::as_array)
        else {
            return;
        };
        for child in children {
            if child.get("type").and_then(Value::as_str) == Some("reaction") {
                if let Some(metadata) = child.get("metadata").and_then(Value::as_object) {
                    if let Some(source_id) = metadata.get("ID").and_then(Value::as_str) {
                        condition_annotations(
                            source_id,
                            metadata.get("rsmi").and_then(Value::as_str),
                            metadata.get("RingBreaker").and_then(Value::as_bool),
                            Some(statistics),
                        );
                    }
                }
            }
            visit(child, statistics);
        }
    }
    visit(route, statistics);
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ConditionSlotParseStatistics, PaRoutesAdapter, analyze_condition_slots};
    use crate::{adapters::Adapter, route::AdaptMode};

    fn route() -> Value {
        json!({
            "type": "mol", "smiles": "CCO", "children": [{
                "type": "reaction", "smiles": "CCO",
                "metadata": {"ID": "US123;1", "rsmi": "C>O.not-smiles>CCO", "RingBreaker": false},
                "children": [{"type": "mol", "smiles": "C", "in_stock": true}]
            }]
        })
    }

    #[test]
    fn preserves_patent_and_condition_annotations() {
        let route = PaRoutesAdapter
            .cast(route(), AdaptMode::Strict, None)
            .unwrap();
        assert_eq!(route.annotations["patent_id"], "US123");
        let reaction = route.target.product_of.unwrap();
        assert_eq!(reaction.annotations["condition_slot"], "O.not-smiles");
        assert_eq!(reaction.annotations["condition_slot_smiles"][0], "O");
    }

    #[test]
    fn condition_diagnostics_are_nonfatal() {
        let mut statistics = ConditionSlotParseStatistics::default();
        analyze_condition_slots(&route(), &mut statistics);
        assert_eq!(statistics.uncanonicalizable_token_count, 1);
        assert_eq!(
            statistics.top_uncanonicalizable_tokens(),
            vec![("not-smiles", 1)]
        );
    }

    #[test]
    fn rejects_mixed_patents() {
        let mut route = route();
        route["children"][0]["children"][0]["children"] = json!([{
            "type": "reaction", "smiles": "C", "metadata": {"ID": "OTHER;1"},
            "children": [{"type": "mol", "smiles": "[H][H]"}]
        }]);
        assert!(
            PaRoutesAdapter
                .cast(route, AdaptMode::Strict, None)
                .is_err()
        );
    }
}
