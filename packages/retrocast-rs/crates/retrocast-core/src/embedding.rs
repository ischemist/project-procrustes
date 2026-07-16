use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    model::Route,
    route_path::RoutePath,
    route_view::{InchiKeyLevel, MoleculeView},
};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LeafExtension {
    pub query_leaf_path: RoutePath,
    pub container_path: RoutePath,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EmbeddingMatch {
    pub query_path: RoutePath,
    pub container_path: RoutePath,
    pub matched_reactions: usize,
    pub leaf_extensions: Vec<LeafExtension>,
}

impl EmbeddingMatch {
    pub fn leaf_extended(&self) -> bool {
        !self.leaf_extensions.is_empty()
    }

    pub fn root_shifted(&self) -> bool {
        self.container_path != RoutePath::target()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EmbeddingOptions {
    pub match_level: InchiKeyLevel,
    pub allow_leaf_extension: bool,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            match_level: InchiKeyLevel::Full,
            allow_leaf_extension: true,
        }
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error(
    "route embedding query must contain at least one reaction; use molecule membership for molecule-only queries"
)]
pub struct InvalidEmbeddingQuery;

#[derive(Clone, Debug, Default)]
struct Trace {
    matched_reactions: usize,
    leaf_extensions: Vec<LeafExtension>,
}

pub fn find_route_embeddings(
    query: &Route,
    container: &Route,
    options: EmbeddingOptions,
) -> Result<Vec<EmbeddingMatch>, InvalidEmbeddingQuery> {
    if query.target.product_of.is_none() {
        return Err(InvalidEmbeddingQuery);
    }
    let query_root = query
        .molecule_at(&RoutePath::target())
        .expect("every route has a target");
    Ok(container
        .molecules()
        .into_iter()
        .filter_map(|container_root| route_embeds_at(query_root.clone(), container_root, options))
        .collect())
}

pub fn route_embeds_at(
    query: MoleculeView<'_>,
    container: MoleculeView<'_>,
    options: EmbeddingOptions,
) -> Option<EmbeddingMatch> {
    if !can_match_root(&query, &container, options) {
        return None;
    }
    if query.subtree_signature(options.match_level, None)
        == container.subtree_signature(options.match_level, None)
    {
        let matched_reactions = subtree_reaction_count(&query);
        return Some(EmbeddingMatch {
            query_path: query.path,
            container_path: container.path,
            matched_reactions,
            leaf_extensions: Vec::new(),
        });
    }

    let mut memo = HashMap::new();
    let trace = match_molecule(&query, &container, options, &mut memo)?;
    Some(EmbeddingMatch {
        query_path: query.path,
        container_path: container.path,
        matched_reactions: trace.matched_reactions,
        leaf_extensions: trace.leaf_extensions,
    })
}

pub fn subtree_reaction_count(molecule: &MoleculeView<'_>) -> usize {
    molecule
        .molecules()
        .into_iter()
        .filter(|candidate| candidate.produced_by().is_some())
        .count()
}

fn can_match_root(
    query: &MoleculeView<'_>,
    container: &MoleculeView<'_>,
    options: EmbeddingOptions,
) -> bool {
    if query.key(options.match_level) != container.key(options.match_level) {
        return false;
    }
    match (query.produced_by(), container.produced_by()) {
        (None, None) => true,
        (None, Some(_)) => options.allow_leaf_extension,
        (Some(_), None) => false,
        (Some(query), Some(container)) => {
            query.signature(options.match_level) == container.signature(options.match_level)
        }
    }
}

fn match_molecule(
    query: &MoleculeView<'_>,
    container: &MoleculeView<'_>,
    options: EmbeddingOptions,
    memo: &mut HashMap<(RoutePath, RoutePath), Option<Trace>>,
) -> Option<Trace> {
    let key = (query.path.clone(), container.path.clone());
    if let Some(trace) = memo.get(&key) {
        return trace.clone();
    }
    if query.key(options.match_level) != container.key(options.match_level) {
        memo.insert(key, None);
        return None;
    }

    let trace = match (query.produced_by(), container.produced_by()) {
        (None, None) => Some(Trace::default()),
        (None, Some(_)) if options.allow_leaf_extension => Some(Trace {
            matched_reactions: 0,
            leaf_extensions: vec![LeafExtension {
                query_leaf_path: query.path.clone(),
                container_path: container.path.clone(),
            }],
        }),
        (None, Some(_)) | (Some(_), None) => None,
        (Some(query_reaction), Some(container_reaction)) => {
            if query_reaction.signature(options.match_level)
                != container_reaction.signature(options.match_level)
            {
                None
            } else {
                match_reactants(
                    &query_reaction.reactants(),
                    &container_reaction.reactants(),
                    options,
                    memo,
                )
                .map(|mut trace| {
                    trace.matched_reactions += 1;
                    trace
                })
            }
        }
    };
    memo.insert(key, trace.clone());
    trace
}

fn match_reactants(
    query: &[MoleculeView<'_>],
    container: &[MoleculeView<'_>],
    options: EmbeddingOptions,
    memo: &mut HashMap<(RoutePath, RoutePath), Option<Trace>>,
) -> Option<Trace> {
    if query.len() != container.len() {
        return None;
    }
    let query_groups = group_by_key(query, options.match_level);
    let container_groups = group_by_key(container, options.match_level);
    if query_groups.keys().ne(container_groups.keys()) {
        return None;
    }

    let mut result = Trace::default();
    for (key, query_group) in query_groups {
        let container_group = &container_groups[&key];
        if query_group.len() != container_group.len() {
            return None;
        }
        let trace = match_same_key(&query_group, container_group, options, memo)?;
        merge(&mut result, trace);
    }
    Some(result)
}

fn group_by_key<'route>(
    molecules: &[MoleculeView<'route>],
    level: InchiKeyLevel,
) -> BTreeMap<String, Vec<MoleculeView<'route>>> {
    let mut groups: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for molecule in molecules {
        groups
            .entry(molecule.key(level))
            .or_default()
            .push(molecule.clone());
    }
    groups
}

fn match_same_key(
    query: &[MoleculeView<'_>],
    container: &[MoleculeView<'_>],
    options: EmbeddingOptions,
    memo: &mut HashMap<(RoutePath, RoutePath), Option<Trace>>,
) -> Option<Trace> {
    fn assign(
        index: usize,
        query: &[MoleculeView<'_>],
        container: &[MoleculeView<'_>],
        used: &mut [bool],
        options: EmbeddingOptions,
        memo: &mut HashMap<(RoutePath, RoutePath), Option<Trace>>,
    ) -> Option<Trace> {
        if index == query.len() {
            return Some(Trace::default());
        }
        let mut best: Option<Trace> = None;
        for container_index in 0..container.len() {
            if used[container_index] {
                continue;
            }
            let Some(trace) =
                match_molecule(&query[index], &container[container_index], options, memo)
            else {
                continue;
            };
            used[container_index] = true;
            let rest = assign(index + 1, query, container, used, options, memo);
            used[container_index] = false;
            let Some(mut rest) = rest else {
                continue;
            };
            let mut candidate = trace;
            merge(&mut candidate, std::mem::take(&mut rest));
            if best
                .as_ref()
                .is_none_or(|current| trace_rank(&candidate) < trace_rank(current))
            {
                best = Some(candidate);
            }
        }
        best
    }

    assign(
        0,
        query,
        container,
        &mut vec![false; container.len()],
        options,
        memo,
    )
}

fn merge(target: &mut Trace, source: Trace) {
    target.matched_reactions += source.matched_reactions;
    target.leaf_extensions.extend(source.leaf_extensions);
}

fn trace_rank(trace: &Trace) -> (usize, Vec<String>, Vec<String>) {
    (
        trace.leaf_extensions.len(),
        trace
            .leaf_extensions
            .iter()
            .map(|extension| extension.query_leaf_path.to_string())
            .collect(),
        trace
            .leaf_extensions
            .iter()
            .map(|extension| extension.container_path.to_string())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{EmbeddingOptions, find_route_embeddings};
    use crate::model::Route;

    fn route(value: serde_json::Value) -> Route {
        serde_json::from_value(value).unwrap()
    }

    fn key(label: char) -> String {
        format!(
            "{}-{}-{label}",
            label.to_string().repeat(14),
            label.to_string().repeat(10)
        )
    }

    #[test]
    fn finds_shifted_root_with_leaf_extension() {
        let query = route(json!({
            "target": {"smiles": "B", "inchikey": key('B'), "product_of": {
                "reactants": [{"smiles": "C", "inchikey": key('C')}]
            }}
        }));
        let container = route(json!({
            "target": {"smiles": "A", "inchikey": key('A'), "product_of": {
                "reactants": [{"smiles": "B", "inchikey": key('B'), "product_of": {
                    "reactants": [{"smiles": "C", "inchikey": key('C'), "product_of": {
                        "reactants": [{"smiles": "D", "inchikey": key('D')}]
                    }}]
                }}]
            }}
        }));

        let matches =
            find_route_embeddings(&query, &container, EmbeddingOptions::default()).unwrap();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].root_shifted());
        assert!(matches[0].leaf_extended());
        assert_eq!(matches[0].matched_reactions, 1);
    }

    #[test]
    fn rejects_molecule_only_query() {
        let query = route(json!({"target": {"smiles": "C", "inchikey": key('C')}}));
        assert!(find_route_embeddings(&query, &query, EmbeddingOptions::default()).is_err());
    }
}
