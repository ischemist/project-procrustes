use std::{collections::BTreeSet, fmt};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    model::{Molecule, Reaction, Route},
    route_path::{MoleculeId, ReactionId, RoutePath, RoutePathError},
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InchiKeyLevel {
    #[default]
    Full,
    NoStereo,
    Connectivity,
}

impl InchiKeyLevel {
    pub fn reduce(self, inchikey: &str) -> String {
        match self {
            Self::Full => inchikey.to_owned(),
            Self::Connectivity => inchikey.split('-').next().unwrap_or(inchikey).to_owned(),
            Self::NoStereo => {
                let parts: Vec<_> = inchikey.split('-').collect();
                if parts.len() == 3 {
                    format!("{}-UHFFFAOYSA-{}", parts[0], parts[2])
                } else {
                    inchikey.to_owned()
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ReactionContentField {
    MappedReactionSmiles,
    Template,
    Reagents,
    Solvents,
}

impl fmt::Display for ReactionContentField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MappedReactionSmiles => "mapped_reaction_smiles",
            Self::Template => "template",
            Self::Reagents => "reagents",
            Self::Solvents => "solvents",
        })
    }
}

impl ReactionContentField {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "mapped_reaction_smiles" => Some(Self::MappedReactionSmiles),
            "template" => Some(Self::Template),
            "reagents" => Some(Self::Reagents),
            "solvents" => Some(Self::Solvents),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RouteLookupError {
    #[error(transparent)]
    InvalidPath(#[from] RoutePathError),
    #[error("expected a molecule path, got {0}")]
    ExpectedMolecule(RoutePath),
    #[error("expected a reaction path, got {0}")]
    ExpectedReaction(RoutePath),
    #[error("route has no node at {0}")]
    MissingNode(RoutePath),
}

#[derive(Clone)]
pub struct MoleculeView<'route> {
    pub route: &'route Route,
    pub path: RoutePath,
    pub value: &'route Molecule,
}

#[derive(Clone)]
pub struct ReactionView<'route> {
    pub route: &'route Route,
    pub path: RoutePath,
    pub value: &'route Reaction,
}

impl Route {
    pub fn key(&self, level: InchiKeyLevel, depth: Option<usize>) -> Value {
        molecule_key(&self.target, level, depth, &[])
    }

    pub fn content_key(
        &self,
        level: InchiKeyLevel,
        fields: &[ReactionContentField],
        depth: Option<usize>,
    ) -> Value {
        molecule_key(&self.target, level, depth, &normalized_fields(fields))
    }

    pub fn molecule_at(&self, path: &RoutePath) -> Result<MoleculeView<'_>, RouteLookupError> {
        if !path.is_molecule() {
            return Err(RouteLookupError::ExpectedMolecule(path.clone()));
        }
        let mut molecule = &self.target;
        for index in path.indices() {
            molecule = molecule
                .product_of
                .as_deref()
                .and_then(|reaction| reaction.reactants.get(*index))
                .ok_or_else(|| RouteLookupError::MissingNode(path.clone()))?;
        }
        Ok(MoleculeView {
            route: self,
            path: path.clone(),
            value: molecule,
        })
    }

    pub fn reaction_at(&self, path: &RoutePath) -> Result<ReactionView<'_>, RouteLookupError> {
        if !path.is_reaction() {
            return Err(RouteLookupError::ExpectedReaction(path.clone()));
        }
        let product_path = path.product()?;
        let product = self.molecule_at(&product_path)?;
        let reaction = product
            .value
            .product_of
            .as_deref()
            .ok_or_else(|| RouteLookupError::MissingNode(path.clone()))?;
        Ok(ReactionView {
            route: self,
            path: path.clone(),
            value: reaction,
        })
    }

    pub fn signature(&self, level: InchiKeyLevel, depth: Option<usize>) -> String {
        molecule_signature(&self.target, level, depth, &[])
    }

    pub fn content_signature(
        &self,
        level: InchiKeyLevel,
        fields: &[ReactionContentField],
        depth: Option<usize>,
    ) -> String {
        let fields = normalized_fields(fields);
        molecule_signature(&self.target, level, depth, &fields)
    }

    pub fn leaves(&self) -> Vec<MoleculeView<'_>> {
        let mut leaves = Vec::new();
        collect_leaves(self, RoutePath::target(), &self.target, &mut leaves);
        leaves
    }

    pub fn molecules(&self) -> Vec<MoleculeView<'_>> {
        let mut molecules = Vec::new();
        collect_molecules(self, RoutePath::target(), &self.target, &mut molecules);
        molecules
    }

    pub fn reactions(&self) -> Vec<ReactionView<'_>> {
        let mut reactions = Vec::new();
        collect_reactions(self, RoutePath::target(), &self.target, &mut reactions);
        reactions
    }

    pub fn depth(&self) -> usize {
        molecule_depth(&self.target)
    }

    pub fn reaction_signatures(&self, level: InchiKeyLevel) -> BTreeSet<String> {
        self.reactions()
            .into_iter()
            .map(|reaction| reaction.signature(level))
            .collect()
    }

    pub fn is_convergent(&self) -> bool {
        self.reactions().into_iter().any(|reaction| {
            reaction
                .value
                .reactants
                .iter()
                .filter(|reactant| reactant.product_of.is_some())
                .count()
                > 1
        })
    }
}

impl<'route> MoleculeView<'route> {
    pub fn id(&self) -> MoleculeId {
        MoleculeId::new(self.path.clone()).expect("molecule view always has a molecule path")
    }

    pub fn key(&self, level: InchiKeyLevel) -> String {
        level.reduce(&self.value.inchikey)
    }

    pub fn produced_by(&self) -> Option<ReactionView<'route>> {
        let reaction = self.value.product_of.as_deref()?;
        Some(ReactionView {
            route: self.route,
            path: self
                .path
                .produced_by()
                .expect("molecule view always has a molecule path"),
            value: reaction,
        })
    }

    pub fn leaves(&self) -> Vec<MoleculeView<'route>> {
        let mut leaves = Vec::new();
        collect_leaves(self.route, self.path.clone(), self.value, &mut leaves);
        leaves
    }

    pub fn molecules(&self) -> Vec<MoleculeView<'route>> {
        let mut molecules = Vec::new();
        collect_molecules(self.route, self.path.clone(), self.value, &mut molecules);
        molecules
    }

    pub fn depth(&self) -> usize {
        molecule_depth(self.value)
    }

    pub fn subtree_signature(&self, level: InchiKeyLevel, depth: Option<usize>) -> String {
        molecule_signature(self.value, level, depth, &[])
    }

    pub fn subtree_key(&self, level: InchiKeyLevel, depth: Option<usize>) -> Value {
        molecule_key(self.value, level, depth, &[])
    }

    pub fn content_subtree_key(
        &self,
        level: InchiKeyLevel,
        fields: &[ReactionContentField],
        depth: Option<usize>,
    ) -> Value {
        molecule_key(self.value, level, depth, &normalized_fields(fields))
    }

    pub fn content_subtree_signature(
        &self,
        level: InchiKeyLevel,
        fields: &[ReactionContentField],
        depth: Option<usize>,
    ) -> String {
        molecule_signature(self.value, level, depth, &normalized_fields(fields))
    }
}

impl<'route> ReactionView<'route> {
    pub fn id(&self) -> ReactionId {
        ReactionId::new(self.path.clone()).expect("reaction view always has a reaction path")
    }

    pub fn product(&self) -> MoleculeView<'route> {
        self.route
            .molecule_at(
                &self
                    .path
                    .product()
                    .expect("reaction view always has a reaction path"),
            )
            .expect("reaction view is created from an existing product")
    }

    pub fn reactants(&self) -> Vec<MoleculeView<'route>> {
        self.value
            .reactants
            .iter()
            .enumerate()
            .map(|(index, reactant)| MoleculeView {
                route: self.route,
                path: self
                    .path
                    .reactant(index)
                    .expect("reaction view always has a reaction path"),
                value: reactant,
            })
            .collect()
    }

    pub fn signature(&self, level: InchiKeyLevel) -> String {
        stable_hash(&reaction_key(self.product().value, self.value, level, &[]))
    }

    pub fn key(&self, level: InchiKeyLevel) -> Value {
        reaction_key(self.product().value, self.value, level, &[])
    }

    pub fn content_key(&self, level: InchiKeyLevel, fields: &[ReactionContentField]) -> Value {
        reaction_key(
            self.product().value,
            self.value,
            level,
            &normalized_fields(fields),
        )
    }

    pub fn content_signature(
        &self,
        level: InchiKeyLevel,
        fields: &[ReactionContentField],
    ) -> String {
        stable_hash(&reaction_key(
            self.product().value,
            self.value,
            level,
            &normalized_fields(fields),
        ))
    }
}

fn collect_leaves<'route>(
    route: &'route Route,
    path: RoutePath,
    molecule: &'route Molecule,
    output: &mut Vec<MoleculeView<'route>>,
) {
    let Some(reaction) = molecule.product_of.as_deref() else {
        output.push(MoleculeView {
            route,
            path,
            value: molecule,
        });
        return;
    };
    let reaction_path = path.produced_by().expect("recursive path is a molecule");
    for (index, reactant) in reaction.reactants.iter().enumerate() {
        collect_leaves(
            route,
            reaction_path
                .reactant(index)
                .expect("recursive path is a reaction"),
            reactant,
            output,
        );
    }
}

fn collect_molecules<'route>(
    route: &'route Route,
    path: RoutePath,
    molecule: &'route Molecule,
    output: &mut Vec<MoleculeView<'route>>,
) {
    output.push(MoleculeView {
        route,
        path: path.clone(),
        value: molecule,
    });
    let Some(reaction) = molecule.product_of.as_deref() else {
        return;
    };
    let reaction_path = path.produced_by().expect("recursive path is a molecule");
    for (index, reactant) in reaction.reactants.iter().enumerate() {
        collect_molecules(
            route,
            reaction_path
                .reactant(index)
                .expect("recursive path is a reaction"),
            reactant,
            output,
        );
    }
}

fn collect_reactions<'route>(
    route: &'route Route,
    product_path: RoutePath,
    molecule: &'route Molecule,
    output: &mut Vec<ReactionView<'route>>,
) {
    let Some(reaction) = molecule.product_of.as_deref() else {
        return;
    };
    let reaction_path = product_path
        .produced_by()
        .expect("recursive path is a molecule");
    output.push(ReactionView {
        route,
        path: reaction_path.clone(),
        value: reaction,
    });
    for (index, reactant) in reaction.reactants.iter().enumerate() {
        collect_reactions(
            route,
            reaction_path
                .reactant(index)
                .expect("recursive path is a reaction"),
            reactant,
            output,
        );
    }
}

fn molecule_depth(molecule: &Molecule) -> usize {
    molecule.product_of.as_deref().map_or(0, |reaction| {
        1 + reaction
            .reactants
            .iter()
            .map(molecule_depth)
            .max()
            .unwrap_or(0)
    })
}

fn normalized_fields(fields: &[ReactionContentField]) -> Vec<ReactionContentField> {
    fields
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn molecule_signature(
    molecule: &Molecule,
    level: InchiKeyLevel,
    depth: Option<usize>,
    fields: &[ReactionContentField],
) -> String {
    stable_hash(&molecule_key(molecule, level, depth, fields))
}

fn molecule_key(
    molecule: &Molecule,
    level: InchiKeyLevel,
    depth: Option<usize>,
    fields: &[ReactionContentField],
) -> Value {
    let label = if fields.is_empty() {
        "mol"
    } else {
        "mol-content"
    };
    let identity = level.reduce(&molecule.inchikey);
    let Some(reaction) = molecule.product_of.as_deref() else {
        return json!([label, identity]);
    };
    if depth == Some(0) {
        return json!([label, identity]);
    }
    let child_depth = depth.map(|value| value - 1);
    let mut child_signatures: Vec<_> = reaction
        .reactants
        .iter()
        .map(|reactant| molecule_signature(reactant, level, child_depth, fields))
        .collect();
    child_signatures.sort();
    json!([
        label,
        identity,
        reaction_key(molecule, reaction, level, fields),
        child_signatures
    ])
}

fn reaction_key(
    product: &Molecule,
    reaction: &Reaction,
    level: InchiKeyLevel,
    fields: &[ReactionContentField],
) -> Value {
    let mut reactants: Vec<_> = reaction
        .reactants
        .iter()
        .map(|reactant| level.reduce(&reactant.inchikey))
        .collect();
    reactants.sort();
    let structural = json!(["rxn", level.reduce(&product.inchikey), reactants]);
    if fields.is_empty() {
        return structural;
    }
    let content: Vec<_> = fields
        .iter()
        .map(|field| json!([field.to_string(), content_value(reaction, *field)]))
        .collect();
    json!(["rxn-content", structural, content])
}

fn content_value(reaction: &Reaction, field: ReactionContentField) -> Value {
    match field {
        ReactionContentField::MappedReactionSmiles => json!(reaction.mapped_reaction_smiles),
        ReactionContentField::Template => json!(reaction.template),
        ReactionContentField::Reagents => ordered_optional(&reaction.reagents),
        ReactionContentField::Solvents => ordered_optional(&reaction.solvents),
    }
}

fn ordered_optional<T>(values: &Option<Vec<T>>) -> Value
where
    T: Clone + Ord + serde::Serialize,
{
    let Some(values) = values.as_ref().filter(|values| !values.is_empty()) else {
        return Value::Null;
    };
    let mut values = values.clone();
    values.sort();
    json!(values)
}

fn stable_hash(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).expect("route identity is serializable");
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{InchiKeyLevel, ReactionContentField};
    use crate::{model::Route, route_path::RoutePath};

    fn route() -> Route {
        serde_json::from_value(json!({
            "target": {
                "smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "product_of": {
                    "reactants": [
                        {"smiles": "O", "inchikey": "XLYOFNOQVPJJNP-UHFFFAOYSA-N"},
                        {"smiles": "CC", "inchikey": "OTMSDBZUPAUEDD-UHFFFAOYSA-N"}
                    ],
                    "template": "join"
                }
            },
            "schema_version": "2"
        }))
        .unwrap()
    }

    #[test]
    fn traverses_route_with_typed_paths() {
        let route = route();
        assert_eq!(route.molecules().len(), 3);
        assert_eq!(route.reactions().len(), 1);
        assert_eq!(route.leaves().len(), 2);
        assert_eq!(
            route
                .molecule_at(&RoutePath::parse("rc:m:/1").unwrap())
                .unwrap()
                .id()
                .to_string(),
            "rc:m:/1"
        );
    }

    #[test]
    fn content_is_opt_in_to_route_identity() {
        let route = route();
        let structural = route.signature(InchiKeyLevel::Full, None);
        let content =
            route.content_signature(InchiKeyLevel::Full, &[ReactionContentField::Template], None);
        assert_ne!(structural, content);
        assert_eq!(structural.len(), 64);
    }

    #[test]
    fn detects_convergence_from_synthesized_siblings() {
        let mut route = route();
        let reaction = route.target.product_of.as_deref_mut().unwrap();
        let leaf = reaction.reactants[0].clone();
        reaction.reactants[0].product_of = Some(Box::new(crate::model::Reaction {
            reactants: vec![leaf.clone()],
            mapped_reaction_smiles: None,
            template: None,
            reagents: None,
            solvents: None,
            annotations: Default::default(),
        }));
        reaction.reactants[1].product_of = Some(Box::new(crate::model::Reaction {
            reactants: vec![leaf],
            mapped_reaction_smiles: None,
            template: None,
            reagents: None,
            solvents: None,
            annotations: Default::default(),
        }));
        assert!(route.is_convergent());
    }
}
