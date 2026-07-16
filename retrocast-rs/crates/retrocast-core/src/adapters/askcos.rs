use std::collections::{BTreeMap, HashSet};

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

const ROOT_UUID: &str = "00000000-0000-0000-0000-000000000000";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Edge {
    source: String,
    target: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExtractedPathway {
    pathway_edges: Vec<Edge>,
    uuid2smiles: BTreeMap<String, String>,
    node_dict: BTreeMap<String, Value>,
    annotations: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AskcosAdapter {
    use_full_graph: bool,
}

impl AskcosAdapter {
    pub fn full_graph() -> Self {
        Self {
            use_full_graph: true,
        }
    }
}

impl Adapter for AskcosAdapter {
    fn name(&self) -> &'static str {
        "askcos"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        if self.use_full_graph {
            return Err(common::logic(
                self.name(),
                "adapter.unsupported_feature",
                "full-graph route extraction is not implemented",
            ));
        }
        let Value::Object(mut payload) = payload else {
            return Err(common::schema(self.name(), "results object is required"));
        };
        let results = payload
            .remove("results")
            .ok_or_else(|| common::schema(self.name(), "results object is required"))?;
        let Value::Object(mut results) = results else {
            return Err(common::schema(self.name(), "results object is required"));
        };
        let annotations = run_annotations(results.get("stats"));
        let uds = results
            .remove("uds")
            .ok_or_else(|| common::schema(self.name(), "results.uds object is required"))?;
        let Value::Object(mut uds) = uds else {
            return Err(common::schema(
                self.name(),
                "results.uds object is required",
            ));
        };
        let uuid2smiles: BTreeMap<String, String> = serde_json::from_value(
            uds.remove("uuid2smiles")
                .ok_or_else(|| common::schema(self.name(), "uds.uuid2smiles is required"))?,
        )
        .map_err(|error| common::schema(self.name(), format!("invalid uuid2smiles: {error}")))?;
        let node_dict: BTreeMap<String, Value> = serde_json::from_value(
            uds.remove("node_dict")
                .ok_or_else(|| common::schema(self.name(), "uds.node_dict is required"))?,
        )
        .map_err(|error| common::schema(self.name(), format!("invalid node_dict: {error}")))?;
        let pathways: Vec<Vec<Edge>> = serde_json::from_value(
            uds.remove("pathways")
                .ok_or_else(|| common::schema(self.name(), "uds.pathways is required"))?,
        )
        .map_err(|error| common::schema(self.name(), format!("invalid pathways: {error}")))?;
        validate_nodes(&node_dict, self.name())?;
        Ok(pathways
            .into_iter()
            .enumerate()
            .map(|(index, edges)| RawRouteEntry {
                payload: serde_json::to_value(compact_pathway(
                    edges,
                    &uuid2smiles,
                    &node_dict,
                    &annotations,
                ))
                .expect("ASKCOS pathway is serializable"),
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
        let pathway: ExtractedPathway = serde_json::from_value(raw_route)
            .map_err(|error| common::schema(self.name(), format!("invalid pathway: {error}")))?;
        let mut adjacency: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for edge in &pathway.pathway_edges {
            adjacency
                .entry(edge.source.clone())
                .or_default()
                .push(edge.target.clone());
        }
        if !pathway.uuid2smiles.contains_key(ROOT_UUID) {
            return Err(node_missing("root chemical", ROOT_UUID, "uuid2smiles"));
        }
        let target_molecule = build_molecule(
            ROOT_UUID,
            &adjacency,
            &pathway.uuid2smiles,
            &pathway.node_dict,
            mode,
            &HashSet::new(),
        )?
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
            annotations: pathway.annotations,
            schema_version: Default::default(),
        })
    }
}

fn compact_pathway(
    pathway_edges: Vec<Edge>,
    uuid2smiles: &BTreeMap<String, String>,
    node_dict: &BTreeMap<String, Value>,
    annotations: &Map<String, Value>,
) -> ExtractedPathway {
    let pathway_uuids: HashSet<&str> = pathway_edges
        .iter()
        .flat_map(|edge| [edge.source.as_str(), edge.target.as_str()])
        .chain([ROOT_UUID])
        .collect();
    let compact_uuid2smiles: BTreeMap<_, _> = pathway_uuids
        .into_iter()
        .filter_map(|uuid| {
            uuid2smiles
                .get(uuid)
                .map(|smiles| (uuid.to_owned(), smiles.clone()))
        })
        .collect();
    let compact_node_dict = compact_uuid2smiles
        .values()
        .filter_map(|smiles| {
            node_dict
                .get(smiles)
                .map(|node| (smiles.clone(), node.clone()))
        })
        .collect();
    ExtractedPathway {
        pathway_edges,
        uuid2smiles: compact_uuid2smiles,
        node_dict: compact_node_dict,
        annotations: annotations.clone(),
    }
}

fn validate_nodes(nodes: &BTreeMap<String, Value>, adapter: &'static str) -> Result<()> {
    for (key, node) in nodes {
        let object = node.as_object().ok_or_else(|| {
            common::schema(adapter, format!("node_dict[{key:?}] must be an object"))
        })?;
        let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
            common::schema(adapter, format!("node_dict[{key:?}].type is required"))
        })?;
        if !matches!(kind, "chemical" | "reaction") {
            return Err(common::schema(
                adapter,
                format!("node_dict[{key:?}] has invalid type {kind:?}"),
            ));
        }
        for field in ["smiles", "id"] {
            if object.get(field).and_then(Value::as_str).is_none() {
                return Err(common::schema(
                    adapter,
                    format!("node_dict[{key:?}].{field} must be a string"),
                ));
            }
        }
        if kind == "chemical" && object.get("terminal").and_then(Value::as_bool).is_none() {
            return Err(common::schema(
                adapter,
                format!("node_dict[{key:?}].terminal must be a bool"),
            ));
        }
    }
    Ok(())
}

fn build_molecule(
    chemical_uuid: &str,
    adjacency: &BTreeMap<String, Vec<String>>,
    uuid2smiles: &BTreeMap<String, String>,
    node_dict: &BTreeMap<String, Value>,
    mode: AdaptMode,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Molecule>> {
    let raw_smiles = uuid2smiles
        .get(chemical_uuid)
        .filter(|smiles| !smiles.is_empty())
        .ok_or_else(|| node_missing("chemical", chemical_uuid, "uuid2smiles"))?;
    let node = node_dict
        .get(raw_smiles)
        .and_then(Value::as_object)
        .filter(|node| node.get("type").and_then(Value::as_str) == Some("chemical"))
        .ok_or_else(|| node_missing("chemical", raw_smiles, "node_dict"))?;
    let (smiles, inchikey) = match chem::normalize(
        node.get("smiles")
            .and_then(Value::as_str)
            .expect("validated chemical smiles"),
    ) {
        Ok(identity) => identity,
        Err(_) if mode == AdaptMode::Prune => return Ok(None),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(common::logic(
            "askcos",
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    let child_reactions = adjacency
        .get(chemical_uuid)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if node["terminal"].as_bool().expect("validated terminal") || child_reactions.is_empty() {
        return Ok(Some(Molecule {
            smiles,
            inchikey,
            product_of: None,
            annotations: Map::new(),
        }));
    }
    if child_reactions.len() > 1 {
        return Err(common::logic(
            "askcos",
            "adapter.route_not_tree",
            format!("molecule {smiles:?} has multiple child reactions"),
        ));
    }
    let mut next_visited = visited.clone();
    next_visited.insert(smiles.clone());
    let reaction = build_reaction(
        &child_reactions[0],
        adjacency,
        uuid2smiles,
        node_dict,
        mode,
        &next_visited,
    )?;
    let Some(reaction) = reaction else {
        return if mode == AdaptMode::Prune {
            Ok(None)
        } else {
            Err(common::logic(
                "askcos",
                "adapter.reaction_empty",
                format!("reaction for {smiles:?} has no reactants"),
            ))
        };
    };
    Ok(Some(Molecule {
        smiles,
        inchikey,
        product_of: Some(Box::new(reaction)),
        annotations: Map::new(),
    }))
}

fn build_reaction(
    reaction_uuid: &str,
    adjacency: &BTreeMap<String, Vec<String>>,
    uuid2smiles: &BTreeMap<String, String>,
    node_dict: &BTreeMap<String, Value>,
    mode: AdaptMode,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Reaction>> {
    let raw_smiles = uuid2smiles
        .get(reaction_uuid)
        .filter(|smiles| !smiles.is_empty())
        .ok_or_else(|| node_missing("reaction", reaction_uuid, "uuid2smiles"))?;
    let node = node_dict
        .get(raw_smiles)
        .and_then(Value::as_object)
        .filter(|node| node.get("type").and_then(Value::as_str) == Some("reaction"))
        .ok_or_else(|| node_missing("reaction", raw_smiles, "node_dict"))?;
    let mut reactants = Vec::new();
    for reactant_uuid in adjacency
        .get(reaction_uuid)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        if let Some(reactant) = build_molecule(
            reactant_uuid,
            adjacency,
            uuid2smiles,
            node_dict,
            mode,
            visited,
        )? {
            reactants.push(reactant);
        }
    }
    if reactants.is_empty() {
        return Ok(None);
    }
    normalize_reactants(&mut reactants);
    let mapped_reaction_smiles = node
        .get("reaction_properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get("mapped_smiles"))
        .and_then(Value::as_str)
        .map(ReactionSmiles::try_from)
        .transpose()?;
    Ok(Some(Reaction {
        reactants,
        mapped_reaction_smiles,
        template: template(node),
        reagents: None,
        solvents: None,
        annotations: Map::from_iter([(
            "source_id".to_owned(),
            json!(node["id"].as_str().expect("validated reaction id")),
        )]),
    }))
}

fn template(node: &Map<String, Value>) -> Option<String> {
    node.get("model_metadata")?
        .as_array()?
        .first()?
        .get("source")?
        .get("template")?
        .get("reaction_smarts")?
        .as_str()
        .map(str::to_owned)
}

fn node_missing(role: &str, node_id: &str, lookup: &str) -> crate::error::EngineError {
    common::logic(
        "askcos",
        "adapter.node_missing",
        format!("missing {role} {node_id:?} in {lookup}"),
    )
}

fn run_annotations(stats: Option<&Value>) -> Map<String, Value> {
    let Some(stats) = stats.and_then(Value::as_object) else {
        return Map::new();
    };
    [
        "total_iterations",
        "total_chemicals",
        "total_reactions",
        "total_templates",
        "total_paths",
    ]
    .into_iter()
    .filter_map(|key| stats.get(key).map(|value| (key.to_owned(), value.clone())))
    .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{AskcosAdapter, ExtractedPathway};
    use crate::{adapters::Adapter, route::AdaptMode};

    fn output() -> Value {
        json!({"results": {
            "stats": {"total_paths": 1, "ignored": 2},
            "uds": {
                "node_dict": {
                    "CCOC(C)=O": {"smiles": "CCOC(C)=O", "id": "chem-root", "type": "chemical", "terminal": false},
                    "CCO": {"smiles": "CCO", "id": "chem-leaf", "type": "chemical", "terminal": true},
                    "CO": {"smiles": "CO", "id": "chem-unused", "type": "chemical", "terminal": true},
                    "CCO>>CCOC(C)=O": {
                        "smiles": "CCO>>CCOC(C)=O", "id": "rxn-1", "type": "reaction",
                        "reaction_properties": {"mapped_smiles": "CCO>>CCOC(C)=O"},
                        "model_metadata": [{"source": {"template": {"reaction_smarts": "esterification"}}}]
                    }
                },
                "uuid2smiles": {
                    "00000000-0000-0000-0000-000000000000": "CCOC(C)=O",
                    "uuid-rxn": "CCO>>CCOC(C)=O",
                    "uuid-leaf": "CCO",
                    "uuid-unused": "CO"
                },
                "pathways": [[
                    {"source": "00000000-0000-0000-0000-000000000000", "target": "uuid-rxn"},
                    {"source": "uuid-rxn", "target": "uuid-leaf"}
                ]]
            }
        }})
    }

    #[test]
    fn extracts_path_and_preserves_reaction_fields() {
        let entry = AskcosAdapter::default()
            .entries(output(), None)
            .unwrap()
            .pop()
            .unwrap();
        let pathway: ExtractedPathway = serde_json::from_value(entry.payload).unwrap();
        assert_eq!(pathway.node_dict.len(), 3);
        assert!(!pathway.node_dict.contains_key("CO"));
        let route = AskcosAdapter::default()
            .cast(
                serde_json::to_value(pathway).unwrap(),
                AdaptMode::Strict,
                None,
            )
            .unwrap();
        assert_eq!(route.annotations["total_paths"], 1);
        let reaction = route.target.product_of.unwrap();
        assert_eq!(reaction.template.as_deref(), Some("esterification"));
        assert_eq!(reaction.annotations["source_id"], "rxn-1");
    }

    #[test]
    fn full_graph_mode_is_explicitly_unsupported() {
        assert!(AskcosAdapter::full_graph().entries(output(), None).is_err());
    }
}
