use std::collections::HashSet;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    chem,
    error::{EngineError, Result},
    model::{Molecule, RawNode, Reaction, Route},
    schema::{CanonicalSmiles, ReactionSmiles},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptMode {
    Strict,
    Prune,
}

impl AdaptMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "strict" => Ok(Self::Strict),
            "prune" => Ok(Self::Prune),
            _ => Err(EngineError::AdapterSchema(format!(
                "unsupported adapt mode {value:?}"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Prune => "prune",
        }
    }
}

pub fn adapt_route(raw: &RawNode, mode: AdaptMode) -> Result<Route> {
    let target = build_molecule(raw, mode, &HashSet::new())?
        .ok_or_else(|| adapter_logic("adapter.target_pruned", "target molecule was pruned"))?;
    let mut annotations = serde_json::Map::new();
    if !raw.scores.is_empty() {
        annotations.insert("scores".to_owned(), Value::Object(raw.scores.clone()));
        if let Some(state_score) = raw.scores.get("state score") {
            annotations.insert("state_score".to_owned(), state_score.clone());
        }
    }
    Ok(Route {
        target,
        annotations,
        schema_version: Default::default(),
    })
}

fn build_molecule(
    raw: &RawNode,
    mode: AdaptMode,
    visited: &HashSet<CanonicalSmiles>,
) -> Result<Option<Molecule>> {
    if raw.kind != "mol" {
        return Err(adapter_logic(
            "adapter.node_type_invalid",
            format!("expected molecule node, found {:?}", raw.kind),
        ));
    }
    let (smiles, inchikey) = match chem::normalize(&raw.smiles) {
        Ok(identity) => identity,
        Err(_error) if mode == AdaptMode::Prune => return Ok(None),
        Err(error) => return Err(error),
    };
    if visited.contains(&smiles) {
        return Err(adapter_logic(
            "adapter.cycle_detected",
            format!("cycle at {smiles:?}"),
        ));
    }
    if raw.in_stock || raw.children.is_empty() {
        return Ok(Some(Molecule {
            smiles,
            inchikey,
            product_of: None,
            annotations: serde_json::Map::new(),
        }));
    }
    if raw.children.len() != 1 {
        return Err(adapter_logic(
            "adapter.route_not_tree",
            format!(
                "molecule {smiles:?} has {} child reactions",
                raw.children.len()
            ),
        ));
    }
    let reaction_node = &raw.children[0];
    if reaction_node.kind != "reaction" {
        return Err(adapter_logic(
            "adapter.node_type_invalid",
            format!(
                "expected reaction child under molecule {smiles:?}, found {:?}",
                reaction_node.kind
            ),
        ));
    }
    let mut next_visited = visited.clone();
    next_visited.insert(smiles.clone());
    let mut reactants = Vec::with_capacity(reaction_node.children.len());
    for child in &reaction_node.children {
        if child.kind != "mol" {
            return Err(adapter_logic(
                "adapter.node_type_invalid",
                format!(
                    "expected molecule child under reaction, found {:?}",
                    child.kind
                ),
            ));
        }
        if let Some(molecule) = build_molecule(child, mode, &next_visited)? {
            reactants.push(molecule);
        }
    }
    if reactants.is_empty() {
        return if mode == AdaptMode::Prune {
            Ok(None)
        } else {
            Err(adapter_logic(
                "adapter.reaction_empty",
                format!("reaction for {smiles:?} has no reactants"),
            ))
        };
    }
    normalize_reactants(&mut reactants);
    let mapped_reaction_smiles = reaction_node
        .metadata
        .get("mapped_reaction_smiles")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let template = reaction_node
        .metadata
        .get("template")
        .and_then(Value::as_str)
        .map(str::to_owned);
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
            annotations: reaction_node.metadata.clone(),
        })),
        annotations: serde_json::Map::new(),
    }))
}

fn adapter_logic(code: &'static str, message: impl Into<String>) -> EngineError {
    EngineError::AdapterLogic {
        adapter: "aizynth",
        code,
        message: message.into(),
    }
}

pub(crate) fn normalize_reactants(reactants: &mut [Molecule]) {
    reactants.sort_by_cached_key(reactant_order_key);
}

pub fn reactant_order(reactants: &[Molecule]) -> Vec<usize> {
    let mut order: Vec<_> = (0..reactants.len()).collect();
    order.sort_by_cached_key(|index| reactant_order_key(&reactants[*index]));
    order
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct StructuralOrderKey {
    identity: String,
    reaction: Option<(ReactionOrderKey, Vec<String>)>,
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct ReactionOrderKey {
    product: String,
    reactants: Vec<String>,
}

fn reactant_order_key(molecule: &Molecule) -> (StructuralOrderKey, String) {
    let structural = structural_order_key(molecule, "full", None);
    let complete = serde_json::to_string(&molecule_order_value(molecule)).unwrap();
    (structural, complete)
}

fn structural_order_key(
    molecule: &Molecule,
    level: &str,
    depth: Option<usize>,
) -> StructuralOrderKey {
    let identity = reduce_inchikey(&molecule.inchikey, level);
    let reaction = molecule.product_of.as_deref().and_then(|reaction| {
        if depth == Some(0) {
            return None;
        }
        let child_depth = depth.map(|value| value - 1);
        let mut reactants = reaction
            .reactants
            .iter()
            .map(|reactant| reduce_inchikey(&reactant.inchikey, level))
            .collect::<Vec<_>>();
        reactants.sort();
        let mut child_signatures = reaction
            .reactants
            .iter()
            .map(|reactant| stable_hash(&molecule_subtree_key(reactant, level, child_depth)))
            .collect::<Vec<_>>();
        child_signatures.sort();
        Some((
            ReactionOrderKey {
                product: identity.clone(),
                reactants,
            },
            child_signatures,
        ))
    });
    StructuralOrderKey { identity, reaction }
}

fn molecule_order_value(molecule: &Molecule) -> Value {
    let reaction = molecule.product_of.as_deref().map(|reaction| {
        json!([
            reaction.mapped_reaction_smiles,
            reaction.template,
            ordered_smiles(&reaction.reagents),
            ordered_smiles(&reaction.solvents),
            reaction
                .reactants
                .iter()
                .map(molecule_order_value)
                .collect::<Vec<_>>()
        ])
    });
    json!([molecule.smiles, molecule.inchikey, reaction])
}

fn ordered_smiles<T: Clone + Ord + serde::Serialize>(values: &Option<Vec<T>>) -> Value {
    let Some(values) = values.as_ref().filter(|values| !values.is_empty()) else {
        return Value::Null;
    };
    let mut values = values.clone();
    values.sort();
    json!(values)
}

pub fn reduce_inchikey(inchikey: &str, level: &str) -> String {
    match level {
        "connectivity" => inchikey.split('-').next().unwrap_or(inchikey).to_owned(),
        "no_stereo" => {
            let parts: Vec<_> = inchikey.split('-').collect();
            if parts.len() == 3 {
                format!("{}-UHFFFAOYSA-{}", parts[0], parts[2])
            } else {
                inchikey.to_owned()
            }
        }
        _ => inchikey.to_owned(),
    }
}

pub fn route_depth(route: &Route) -> usize {
    molecule_depth(&route.target)
}

fn molecule_depth(molecule: &Molecule) -> usize {
    molecule.product_of.as_ref().map_or(0, |reaction| {
        1 + reaction
            .reactants
            .iter()
            .map(molecule_depth)
            .max()
            .unwrap_or(0)
    })
}

pub fn leaves(route: &Route) -> Vec<&Molecule> {
    let mut leaves = Vec::new();
    collect_leaves(&route.target, &mut leaves);
    leaves
}

fn collect_leaves<'a>(molecule: &'a Molecule, leaves: &mut Vec<&'a Molecule>) {
    match &molecule.product_of {
        None => leaves.push(molecule),
        Some(reaction) => {
            for reactant in &reaction.reactants {
                collect_leaves(reactant, leaves);
            }
        }
    }
}

pub fn route_signature(route: &Route, level: &str, depth: Option<usize>) -> String {
    stable_hash(&molecule_subtree_key(&route.target, level, depth))
}

pub fn root_reaction_signature(route: &Route, level: &str) -> Option<String> {
    let reaction = route.target.product_of.as_ref()?;
    Some(stable_hash(&reaction_key(&route.target, reaction, level)))
}

fn molecule_subtree_key(molecule: &Molecule, level: &str, depth: Option<usize>) -> Value {
    let molecule_key = reduce_inchikey(&molecule.inchikey, level);
    let Some(reaction) = &molecule.product_of else {
        return json!(["mol", molecule_key]);
    };
    if depth == Some(0) {
        return json!(["mol", molecule_key]);
    }
    let next_depth = depth.map(|value| value - 1);
    let mut child_signatures: Vec<String> = reaction
        .reactants
        .iter()
        .map(|reactant| stable_hash(&molecule_subtree_key(reactant, level, next_depth)))
        .collect();
    child_signatures.sort();
    json!([
        "mol",
        molecule_key,
        reaction_key(molecule, reaction, level),
        child_signatures
    ])
}

fn reaction_key(product: &Molecule, reaction: &Reaction, level: &str) -> Value {
    let mut reactants: Vec<_> = reaction
        .reactants
        .iter()
        .map(|molecule| reduce_inchikey(&molecule.inchikey, level))
        .collect();
    reactants.sort();
    json!(["rxn", reduce_inchikey(&product.inchikey, level), reactants])
}

fn stable_hash(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).expect("route identity is serializable");
    format!("{:x}", Sha256::digest(bytes))
}
