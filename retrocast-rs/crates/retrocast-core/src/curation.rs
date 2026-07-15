use std::collections::HashSet;

use crate::{
    model::{Molecule, Route, Target, Task},
    route_path::RoutePath,
    route_view::InchiKeyLevel,
};

pub fn excise_reactions(route: &Route, excluded: &HashSet<String>) -> Vec<Route> {
    if route.target.product_of.is_none() {
        return Vec::new();
    }
    let mut subroutes = Vec::new();
    let target = rebuild_without_reactions(
        route,
        &route.target,
        &RoutePath::target(),
        excluded,
        &mut subroutes,
    );
    let mut routes = Vec::new();
    if target.product_of.is_some() {
        routes.push(Route {
            target,
            annotations: route.annotations.clone(),
            schema_version: Default::default(),
        });
    }
    routes.extend(subroutes);
    routes
}

fn rebuild_without_reactions(
    route: &Route,
    molecule: &Molecule,
    path: &RoutePath,
    excluded: &HashSet<String>,
    subroutes: &mut Vec<Route>,
) -> Molecule {
    let Some(reaction) = molecule.product_of.as_deref() else {
        return molecule.clone();
    };
    let reaction_path = path.produced_by().expect("molecule path");
    let signature = route
        .reaction_at(&reaction_path)
        .expect("path came from route")
        .signature(InchiKeyLevel::Full);
    if excluded.contains(&signature) {
        for (index, reactant) in reaction.reactants.iter().enumerate() {
            if reactant.product_of.is_none() {
                continue;
            }
            let child_path = reaction_path.reactant(index).expect("reaction path");
            let rebuilt =
                rebuild_without_reactions(route, reactant, &child_path, excluded, subroutes);
            if rebuilt.product_of.is_some() {
                subroutes.push(Route {
                    target: rebuilt,
                    annotations: route.annotations.clone(),
                    schema_version: Default::default(),
                });
            }
        }
        let mut leaf = molecule.clone();
        leaf.product_of = None;
        return leaf;
    }

    let mut rebuilt = molecule.clone();
    let rebuilt_reaction = rebuilt.product_of.as_deref_mut().expect("cloned reaction");
    rebuilt_reaction.reactants = reaction
        .reactants
        .iter()
        .enumerate()
        .map(|(index, reactant)| {
            let child_path = reaction_path.reactant(index).expect("reaction path");
            rebuild_without_reactions(route, reactant, &child_path, excluded, subroutes)
        })
        .collect();
    rebuilt
}

pub fn deduplicate_routes(routes: Vec<Route>) -> Vec<Route> {
    let mut seen = HashSet::new();
    routes
        .into_iter()
        .filter(|route| seen.insert(route.signature(InchiKeyLevel::Full, None)))
        .collect()
}

pub fn filter_by_route_type(task: &Task, convergent: bool) -> Vec<Target> {
    task.targets
        .values()
        .filter(|target| {
            target
                .acceptable_routes
                .first()
                .is_some_and(|route| route.is_convergent() == convergent)
        })
        .cloned()
        .collect()
}

pub fn clean_and_prioritize_pools(
    primary: Vec<Target>,
    secondary: Vec<Target>,
) -> (Vec<Target>, Vec<Target>) {
    let primary_signatures: HashSet<_> = primary
        .iter()
        .filter_map(|target| target.acceptable_routes.first())
        .map(|route| route.signature(InchiKeyLevel::Full, None))
        .collect();
    let secondary: Vec<_> = secondary
        .into_iter()
        .filter(|target| {
            target.acceptable_routes.first().is_none_or(|route| {
                !primary_signatures.contains(&route.signature(InchiKeyLevel::Full, None))
            })
        })
        .collect();
    let secondary_smiles: HashSet<_> = secondary
        .iter()
        .map(|target| target.smiles.clone())
        .collect();
    let primary_smiles: HashSet<_> = primary.iter().map(|target| target.smiles.clone()).collect();
    let ambiguous: HashSet<_> = primary_smiles
        .intersection(&secondary_smiles)
        .cloned()
        .collect();
    (
        primary
            .into_iter()
            .filter(|target| !ambiguous.contains(&target.smiles))
            .collect(),
        secondary
            .into_iter()
            .filter(|target| !ambiguous.contains(&target.smiles))
            .collect(),
    )
}

pub fn generate_pruned_routes(route: &Route, stock: &HashSet<String>) -> Vec<Route> {
    if route.target.product_of.is_none() {
        return Vec::new();
    }
    let mut intermediate_paths = Vec::new();
    collect_stock_intermediates(
        &route.target,
        &RoutePath::target(),
        stock,
        &mut intermediate_paths,
    );
    let mut routes = Vec::new();
    for prune_paths in antichains(&intermediate_paths) {
        let target = rebuild_pruned(&route.target, &RoutePath::target(), &prune_paths);
        let candidate = Route {
            target,
            annotations: route.annotations.clone(),
            schema_version: Default::default(),
        };
        if candidate
            .leaves()
            .iter()
            .all(|leaf| stock.contains(leaf.value.inchikey.as_str()))
        {
            routes.push(candidate);
        }
    }
    deduplicate_routes(routes)
}

fn collect_stock_intermediates(
    molecule: &Molecule,
    path: &RoutePath,
    stock: &HashSet<String>,
    output: &mut Vec<RoutePath>,
) {
    let Some(reaction) = molecule.product_of.as_deref() else {
        return;
    };
    if path.depth() > 0 && stock.contains(molecule.inchikey.as_str()) {
        output.push(path.clone());
    }
    let reaction_path = path.produced_by().expect("molecule path");
    for (index, reactant) in reaction.reactants.iter().enumerate() {
        collect_stock_intermediates(
            reactant,
            &reaction_path.reactant(index).expect("reaction path"),
            stock,
            output,
        );
    }
}

fn antichains(paths: &[RoutePath]) -> Vec<HashSet<RoutePath>> {
    fn combinations(
        paths: &[RoutePath],
        size: usize,
        start: usize,
        selected: &mut Vec<RoutePath>,
        output: &mut Vec<HashSet<RoutePath>>,
    ) {
        if selected.len() == size {
            if selected.iter().enumerate().all(|(left_index, left)| {
                selected
                    .iter()
                    .skip(left_index + 1)
                    .all(|right| !is_ancestor(left, right))
            }) {
                output.push(selected.iter().cloned().collect());
            }
            return;
        }
        for index in start..paths.len() {
            selected.push(paths[index].clone());
            combinations(paths, size, index + 1, selected, output);
            selected.pop();
        }
    }

    let mut output = vec![HashSet::new()];
    for size in 1..=paths.len() {
        combinations(paths, size, 0, &mut Vec::new(), &mut output);
    }
    output
}

fn is_ancestor(left: &RoutePath, right: &RoutePath) -> bool {
    let (shorter, longer) = if left.depth() < right.depth() {
        (left.indices(), right.indices())
    } else {
        (right.indices(), left.indices())
    };
    shorter == &longer[..shorter.len()]
}

fn rebuild_pruned(
    molecule: &Molecule,
    path: &RoutePath,
    prune_paths: &HashSet<RoutePath>,
) -> Molecule {
    if prune_paths.contains(path) || molecule.product_of.is_none() {
        let mut leaf = molecule.clone();
        leaf.product_of = None;
        return leaf;
    }
    let mut rebuilt = molecule.clone();
    let reaction = rebuilt
        .product_of
        .as_deref_mut()
        .expect("non-leaf molecule");
    let reaction_path = path.produced_by().expect("molecule path");
    let reactants = reaction.reactants.clone();
    reaction.reactants = reactants
        .iter()
        .enumerate()
        .map(|(index, reactant)| {
            rebuild_pruned(
                reactant,
                &reaction_path.reactant(index).expect("reaction path"),
                prune_paths,
            )
        })
        .collect();
    rebuilt
}
