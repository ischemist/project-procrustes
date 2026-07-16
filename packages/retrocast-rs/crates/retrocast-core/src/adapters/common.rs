use std::collections::{BTreeMap, HashSet};

use serde_json::{Map, Value};

use crate::{
    chem,
    error::{EngineError, Result},
    model::{Molecule, Reaction},
    route::{AdaptMode, normalize_reactants},
    schema::CanonicalSmiles,
};

pub type PrecursorMap = BTreeMap<CanonicalSmiles, Vec<String>>;
pub type ReactionAnnotations = BTreeMap<CanonicalSmiles, Map<String, Value>>;

pub fn build_from_precursor_map(
    root_smiles: &str,
    precursor_map: &PrecursorMap,
    adapter: &'static str,
    mode: AdaptMode,
    annotations: Option<&ReactionAnnotations>,
) -> Result<Option<Molecule>> {
    build_precursor_node(
        root_smiles,
        precursor_map,
        adapter,
        mode,
        annotations,
        &HashSet::new(),
    )
}

fn build_precursor_node(
    smiles: &str,
    precursor_map: &PrecursorMap,
    adapter: &'static str,
    mode: AdaptMode,
    annotations: Option<&ReactionAnnotations>,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Molecule>> {
    let (smiles, inchikey) = match chem::normalize(smiles) {
        Ok(identity) => identity,
        Err(_) if mode == AdaptMode::Prune => return Ok(None),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(logic(
            adapter,
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    let Some(reactant_smiles) = precursor_map.get(&smiles) else {
        return Ok(Some(Molecule {
            smiles,
            inchikey,
            product_of: None,
            annotations: Map::new(),
        }));
    };

    let mut next_visited = visited.clone();
    next_visited.insert(smiles.clone());
    let mut reactants = Vec::with_capacity(reactant_smiles.len());
    for reactant in reactant_smiles {
        if let Some(reactant) = build_precursor_node(
            reactant,
            precursor_map,
            adapter,
            mode,
            annotations,
            &next_visited,
        )? {
            reactants.push(reactant);
        }
    }
    if reactants.is_empty() {
        return if mode == AdaptMode::Prune {
            Ok(None)
        } else {
            Err(logic(
                adapter,
                "adapter.reaction_empty",
                format!("reaction for {smiles:?} has no reactants"),
            ))
        };
    }
    normalize_reactants(&mut reactants);
    Ok(Some(Molecule {
        annotations: Map::new(),
        product_of: Some(Box::new(Reaction {
            reactants,
            mapped_reaction_smiles: None,
            template: None,
            reagents: None,
            solvents: None,
            annotations: annotations
                .and_then(|annotations| annotations.get(&smiles))
                .cloned()
                .unwrap_or_default(),
        })),
        smiles,
        inchikey,
    }))
}

pub fn build_plain_tree(
    node: &Value,
    adapter: &'static str,
    mode: AdaptMode,
) -> Result<Option<Molecule>> {
    build_plain_node(node, adapter, mode, &HashSet::new())
}

fn build_plain_node(
    node: &Value,
    adapter: &'static str,
    mode: AdaptMode,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Molecule>> {
    let object = node.as_object().ok_or_else(|| {
        schema(
            adapter,
            "route node must be an object with smiles and children",
        )
    })?;
    let raw_smiles = object
        .get("smiles")
        .and_then(Value::as_str)
        .ok_or_else(|| schema(adapter, "route node is missing string smiles"))?;
    let children = object
        .get("children")
        .map(|children| {
            children
                .as_array()
                .ok_or_else(|| schema(adapter, "route node children must be a list"))
        })
        .transpose()?
        .map(Vec::as_slice)
        .unwrap_or_default();
    let (smiles, inchikey) = match chem::normalize(raw_smiles) {
        Ok(identity) => identity,
        Err(_) if mode == AdaptMode::Prune => return Ok(None),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(logic(
            adapter,
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    if children.is_empty() {
        return Ok(Some(Molecule {
            smiles,
            inchikey,
            product_of: None,
            annotations: Map::new(),
        }));
    }
    let mut next_visited = visited.clone();
    next_visited.insert(smiles.clone());
    let mut reactants = Vec::with_capacity(children.len());
    for child in children {
        if let Some(child) = build_plain_node(child, adapter, mode, &next_visited)? {
            reactants.push(child);
        }
    }
    if reactants.is_empty() {
        return if mode == AdaptMode::Prune {
            Ok(None)
        } else {
            Err(logic(
                adapter,
                "adapter.reaction_empty",
                format!("reaction for {smiles:?} has no reactants"),
            ))
        };
    }
    normalize_reactants(&mut reactants);
    Ok(Some(Molecule {
        smiles,
        inchikey,
        product_of: Some(Box::new(Reaction {
            reactants,
            mapped_reaction_smiles: None,
            template: None,
            reagents: None,
            solvents: None,
            annotations: Map::new(),
        })),
        annotations: Map::new(),
    }))
}

pub fn require_target_match(
    route_smiles: &str,
    target_smiles: &str,
    target_id: &str,
    adapter: &'static str,
) -> Result<()> {
    let (expected, _) = chem::normalize(target_smiles)?;
    if expected == route_smiles {
        Ok(())
    } else {
        Err(EngineError::TargetMismatch {
            adapter,
            target_id: target_id.to_owned(),
            expected: expected.to_string(),
            actual: route_smiles.to_owned(),
        })
    }
}

pub fn target_pruned(adapter: &'static str) -> EngineError {
    logic(
        adapter,
        "adapter.target_pruned",
        "target molecule was pruned",
    )
}

pub fn schema(adapter: &'static str, message: impl Into<String>) -> EngineError {
    EngineError::AdapterLogic {
        adapter,
        code: "adapter.schema_invalid",
        message: message.into(),
    }
}

pub fn logic(adapter: &'static str, code: &'static str, message: impl Into<String>) -> EngineError {
    EngineError::AdapterLogic {
        adapter,
        code,
        message: message.into(),
    }
}
