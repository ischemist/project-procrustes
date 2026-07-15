use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::Result,
    model::{Molecule, Reaction, Route, Target},
    route::{AdaptMode, normalize_reactants},
    schema::CanonicalSmiles,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Precursor {
    smiles: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    cost_per_kg: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Disconnection {
    reaction_name: String,
    #[serde(default)]
    named_reaction: Option<String>,
    #[serde(default)]
    category: String,
    #[serde(default)]
    score: f64,
    #[serde(default)]
    precursors: Vec<Precursor>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Node {
    smiles: String,
    #[serde(default)]
    is_purchasable: bool,
    #[serde(default)]
    functional_groups: Vec<String>,
    #[serde(default)]
    best_disconnection: Option<Disconnection>,
    #[serde(default)]
    children: Vec<Node>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MolBuilderAdapter;

impl Adapter for MolBuilderAdapter {
    fn name(&self) -> &'static str {
        "molbuilder"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let routes: Vec<Node> = serde_json::from_value(payload)
            .map_err(|error| common::schema(self.name(), format!("invalid route list: {error}")))?;
        Ok(routes
            .into_iter()
            .enumerate()
            .map(|(index, route)| RawRouteEntry {
                payload: serde_json::to_value(route).expect("MolBuilder route is serializable"),
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
        let node: Node = serde_json::from_value(raw_route)
            .map_err(|error| common::schema(self.name(), format!("invalid route: {error}")))?;
        let target_molecule = build_node(&node, mode, &HashSet::new())?
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
            annotations: Map::new(),
            schema_version: Default::default(),
        })
    }
}

fn build_node(
    node: &Node,
    mode: AdaptMode,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Molecule>> {
    let (smiles, inchikey) = match chem::normalize(&node.smiles) {
        Ok(identity) => identity,
        Err(_) if mode == AdaptMode::Prune => return Ok(None),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(common::logic(
            "molbuilder",
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    let molecule_annotations = if node.functional_groups.is_empty() {
        Map::new()
    } else {
        Map::from_iter([(
            "functional_groups".to_owned(),
            json!(node.functional_groups),
        )])
    };
    if node.is_purchasable || node.children.is_empty() {
        return Ok(Some(Molecule {
            smiles,
            inchikey,
            product_of: None,
            annotations: molecule_annotations,
        }));
    }
    let mut next_visited = visited.clone();
    next_visited.insert(smiles.clone());
    let mut reactants = Vec::new();
    for child in &node.children {
        if let Some(child) = build_node(child, mode, &next_visited)? {
            reactants.push(child);
        }
    }
    if reactants.is_empty() {
        return if mode == AdaptMode::Prune {
            Ok(None)
        } else {
            Err(common::logic(
                "molbuilder",
                "adapter.reaction_empty",
                format!("reaction for {smiles:?} has no reactants"),
            ))
        };
    }
    normalize_reactants(&mut reactants);
    let (template, annotations) = reaction_fields(node);
    Ok(Some(Molecule {
        smiles,
        inchikey,
        product_of: Some(Box::new(Reaction {
            reactants,
            mapped_reaction_smiles: None,
            template,
            reagents: None,
            solvents: None,
            annotations,
        })),
        annotations: molecule_annotations,
    }))
}

fn reaction_fields(node: &Node) -> (Option<String>, Map<String, Value>) {
    let Some(disconnection) = &node.best_disconnection else {
        return (None, Map::new());
    };
    let reaction_name = disconnection.reaction_name.trim();
    let mut annotations = Map::from_iter([("score".to_owned(), json!(disconnection.score))]);
    if !reaction_name.is_empty() {
        annotations.insert("reaction_name".to_owned(), json!(reaction_name));
    }
    if let Some(named_reaction) = &disconnection.named_reaction {
        annotations.insert("named_reaction".to_owned(), json!(named_reaction));
    }
    if !disconnection.category.is_empty() {
        annotations.insert("category".to_owned(), json!(disconnection.category));
    }
    if !disconnection.precursors.is_empty() {
        annotations.insert(
            "precursors".to_owned(),
            serde_json::to_value(&disconnection.precursors).expect("precursors are serializable"),
        );
    }
    (
        (!reaction_name.is_empty()).then(|| reaction_name.to_owned()),
        annotations,
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::MolBuilderAdapter;
    use crate::{adapters::Adapter, route::AdaptMode};

    #[test]
    fn preserves_molecule_and_disconnection_metadata() {
        let route = MolBuilderAdapter
            .cast(
                json!({
                    "smiles": "CCO",
                    "functional_groups": ["alcohol"],
                    "best_disconnection": {
                        "reaction_name": "Reduction",
                        "named_reaction": "NaBH4 Reduction",
                        "score": 0.85,
                        "precursors": [{"smiles": "CC=O", "name": "acetaldehyde", "cost_per_kg": 15.0}]
                    },
                    "children": [{"smiles": "CC=O", "is_purchasable": true}]
                }),
                AdaptMode::Strict,
                None,
            )
            .unwrap();
        assert_eq!(route.target.annotations["functional_groups"][0], "alcohol");
        let reaction = route.target.product_of.unwrap();
        assert_eq!(reaction.template.as_deref(), Some("Reduction"));
        assert_eq!(reaction.annotations["precursors"][0]["cost_per_kg"], 15.0);
    }
}
