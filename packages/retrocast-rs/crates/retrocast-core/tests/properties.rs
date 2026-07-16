use std::collections::{BTreeMap, BTreeSet};

use proptest::{collection, prelude::*};
use retrocast_core::{
    io::validate_path_component,
    model::{Candidate, Constraint, Molecule, Reaction, Route},
    route_path::RoutePath,
    route_view::InchiKeyLevel,
    schema::{CanonicalSmiles, InchiKey, ReactionSmiles, SchemaVersion},
    score::{Stocks, check_task_constraints},
    stats::{bootstrap_distribution, paired_difference, probabilistic_ranking, summarize_values},
};
use serde_json::{Map, Value, json};

fn bounded_string() -> impl Strategy<Value = String> {
    collection::vec(any::<char>(), 0..64).prop_map(|characters| characters.into_iter().collect())
}

fn valid_route_wire() -> Value {
    json!({
        "target": {
            "smiles": "CCO",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        },
        "schema_version": "2"
    })
}

#[derive(Clone, Debug)]
enum Tree {
    Leaf(u8),
    Branch(u8, Vec<Tree>),
}

fn trees() -> impl Strategy<Value = Tree> {
    any::<u8>()
        .prop_map(Tree::Leaf)
        .prop_recursive(4, 32, 4, |child| {
            (any::<u8>(), collection::vec(child, 1..4))
                .prop_map(|(tag, children)| Tree::Branch(tag, children))
        })
}

fn molecule(tag: u8) -> Molecule {
    let letter = char::from(b'A' + tag % 26);
    Molecule {
        smiles: CanonicalSmiles::try_from(format!("node-{tag}")).unwrap(),
        inchikey: InchiKey::try_from(format!("AAAAAAAAAAAAA{letter}-BBBBBBBBBB-C")).unwrap(),
        product_of: None,
        annotations: Map::new(),
    }
}

fn molecule_from_tree(tree: &Tree) -> Molecule {
    let tag = match tree {
        Tree::Leaf(tag) | Tree::Branch(tag, _) => *tag,
    };
    let mut molecule = molecule(tag);
    if let Tree::Branch(_, children) = tree {
        molecule.product_of = Some(Box::new(Reaction {
            reactants: children.iter().map(molecule_from_tree).collect(),
            mapped_reaction_smiles: None,
            template: None,
            reagents: None,
            solvents: None,
            annotations: Map::new(),
        }));
    }
    molecule
}

fn route_from_tree(tree: &Tree) -> Route {
    Route {
        target: molecule_from_tree(tree),
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

fn tree_counts(tree: &Tree) -> (usize, usize, usize, usize) {
    match tree {
        Tree::Leaf(_) => (1, 0, 1, 0),
        Tree::Branch(_, children) => {
            let child_counts = children.iter().map(tree_counts).collect::<Vec<_>>();
            (
                1 + child_counts.iter().map(|counts| counts.0).sum::<usize>(),
                1 + child_counts.iter().map(|counts| counts.1).sum::<usize>(),
                child_counts.iter().map(|counts| counts.2).sum(),
                1 + child_counts
                    .iter()
                    .map(|counts| counts.3)
                    .max()
                    .unwrap_or(0),
            )
        }
    }
}

fn mutate_nonstructural_content(molecule: &mut Molecule) {
    molecule
        .annotations
        .insert("ignored".to_owned(), json!(true));
    let Some(reaction) = molecule.product_of.as_deref_mut() else {
        return;
    };
    reaction.reactants.reverse();
    reaction.template = Some("changed-without-changing-topology".to_owned());
    reaction
        .annotations
        .insert("ignored".to_owned(), json!({"nested": [1, 2, 3]}));
    for reactant in &mut reaction.reactants {
        mutate_nonstructural_content(reactant);
    }
}

fn chain_route(depth: usize) -> Route {
    let mut child = molecule(0);
    for index in 0..depth {
        let mut parent = molecule((index + 1) as u8);
        parent.product_of = Some(Box::new(Reaction {
            reactants: vec![child],
            mapped_reaction_smiles: None,
            template: None,
            reagents: None,
            solvents: None,
            annotations: Map::new(),
        }));
        child = parent;
    }
    Route {
        target: child,
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

fn three_leaf_route() -> Route {
    let mut target = molecule(3);
    target.product_of = Some(Box::new(Reaction {
        reactants: vec![molecule(0), molecule(1), molecule(2)],
        mapped_reaction_smiles: None,
        template: None,
        reagents: None,
        solvents: None,
        annotations: Map::new(),
    }));
    Route {
        target,
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

proptest! {
    #[test]
    fn route_paths_round_trip_canonically(
        indices in collection::vec(any::<usize>(), 0..32),
        molecule in any::<bool>(),
    ) {
        let path = if molecule {
            RoutePath::Molecule(indices.into_boxed_slice())
        } else {
            RoutePath::Reaction(indices.into_boxed_slice())
        };
        let encoded = path.to_string();
        let decoded = RoutePath::parse(&encoded).unwrap();

        prop_assert_eq!(&decoded, &path);
        prop_assert_eq!(
            serde_json::from_str::<RoutePath>(&serde_json::to_string(&path).unwrap()).unwrap(),
            path,
        );
    }

    #[test]
    fn path_component_validation_matches_its_security_boundary(value in bounded_string()) {
        let unsafe_component = value.is_empty()
            || value == "."
            || value == ".."
            || value.contains('/')
            || value.contains('\\')
            || value.contains('\0');

        prop_assert_eq!(validate_path_component(&value, "component").is_err(), unsafe_component);
    }

    #[test]
    fn scalar_deserializers_accept_exactly_their_wire_invariants(value in bounded_string()) {
        let wire = serde_json::to_string(&value).unwrap();
        let key_is_valid = {
            let bytes = value.as_bytes();
            bytes.len() == 27
                && bytes[14] == b'-'
                && bytes[25] == b'-'
                && bytes.iter().enumerate().all(|(index, byte)| {
                    matches!(index, 14 | 25) || byte.is_ascii_uppercase()
                })
        };

        prop_assert_eq!(serde_json::from_str::<CanonicalSmiles>(&wire).is_ok(), !value.is_empty());
        prop_assert_eq!(serde_json::from_str::<ReactionSmiles>(&wire).is_ok(), !value.is_empty());
        prop_assert_eq!(serde_json::from_str::<InchiKey>(&wire).is_ok(), key_is_valid);
        prop_assert_eq!(serde_json::from_str::<SchemaVersion>(&wire).is_ok(), value == "2");
    }

    #[test]
    fn candidate_deserialization_enforces_exactly_one_ranked_outcome(
        rank in any::<u16>(),
        has_route in any::<bool>(),
        has_failure in any::<bool>(),
    ) {
        let mut candidate = Map::from_iter([("rank".to_owned(), Value::from(rank))]);
        if has_route {
            candidate.insert("route".to_owned(), valid_route_wire());
        }
        if has_failure {
            candidate.insert("failure".to_owned(), json!({"code": "provider.failure"}));
        }

        let parsed = serde_json::from_value::<Candidate>(Value::Object(candidate));
        prop_assert_eq!(parsed.is_ok(), rank > 0 && (has_route ^ has_failure));
    }

    #[test]
    fn generated_route_topologies_preserve_traversal_and_identity_invariants(tree in trees()) {
        let route = route_from_tree(&tree);
        let (molecule_count, reaction_count, leaf_count, depth) = tree_counts(&tree);

        let molecules = route.molecules();
        let reactions = route.reactions();
        let leaves = route.leaves();
        prop_assert_eq!(molecules.len(), molecule_count);
        prop_assert_eq!(reactions.len(), reaction_count);
        prop_assert_eq!(leaves.len(), leaf_count);
        prop_assert_eq!(route.depth(), depth);

        let molecule_ids = molecules.iter().map(|view| view.id().to_string()).collect::<BTreeSet<_>>();
        let reaction_ids = reactions.iter().map(|view| view.id().to_string()).collect::<BTreeSet<_>>();
        prop_assert_eq!(molecule_ids.len(), molecule_count);
        prop_assert_eq!(reaction_ids.len(), reaction_count);
        for molecule in molecules {
            let looked_up = route.molecule_at(&molecule.path).unwrap();
            prop_assert_eq!(&looked_up.value.inchikey, &molecule.value.inchikey);
        }
        for reaction in reactions {
            let looked_up = route.reaction_at(&reaction.path).unwrap();
            prop_assert_eq!(looked_up.reactants().len(), reaction.value.reactants.len());
            prop_assert_eq!(looked_up.product().path, reaction.path.product().unwrap());
        }

        let wire = serde_json::to_value(&route).unwrap();
        let round_trip: Route = serde_json::from_value(wire.clone()).unwrap();
        prop_assert_eq!(serde_json::to_value(round_trip).unwrap(), wire);

        let signature = route.signature(InchiKeyLevel::Full, None);
        let mut reordered_and_annotated = route.clone();
        mutate_nonstructural_content(&mut reordered_and_annotated.target);
        prop_assert_eq!(
            reordered_and_annotated.signature(InchiKeyLevel::Full, None),
            signature,
            "structural identity changed after annotation and sibling-order mutations",
        );
    }

    #[test]
    fn route_depth_constraint_matches_generated_chain_depth(
        depth in 0_usize..13,
        maximum in 0_usize..13,
    ) {
        let constraint = Constraint {
            kind: "retrocast.route_depth".to_owned(),
            fields: Map::from_iter([("max_depth".to_owned(), json!(maximum))]),
        };

        let result = check_task_constraints(
            &chain_route(depth),
            &[constraint],
            &Stocks::new(),
            "full",
        )
        .unwrap();

        prop_assert_eq!(result.status == "pass", depth <= maximum);
        prop_assert_eq!(result.checks.is_empty(), depth <= maximum);
    }

    #[test]
    fn stock_termination_passes_exactly_when_every_generated_leaf_is_present(mask in 0_u8..8) {
        let route = three_leaf_route();
        let leaf_keys = route
            .target
            .product_of
            .as_ref()
            .unwrap()
            .reactants
            .iter()
            .map(|leaf| leaf.inchikey.to_string())
            .collect::<Vec<_>>();
        let selected = leaf_keys
            .iter()
            .enumerate()
            .filter(|(index, _)| mask & (1 << index) != 0)
            .map(|(_, key)| key.clone())
            .collect::<BTreeSet<_>>();
        let stocks = BTreeMap::from([("generated".to_owned(), selected)]);
        let constraint = Constraint {
            kind: "retrocast.stock_termination".to_owned(),
            fields: Map::from_iter([("stock".to_owned(), json!("generated"))]),
        };

        let result = check_task_constraints(&route, &[constraint], &stocks, "full").unwrap();

        prop_assert_eq!(result.status == "pass", mask == 0b111);
        prop_assert_eq!(result.checks.is_empty(), mask == 0b111);
    }

    #[test]
    fn bootstrap_distribution_is_seed_stable_and_stays_inside_the_sample_range(
        raw_values in collection::vec(-1_000_i16..1_000, 1..64),
        n_boot in 1_usize..128,
        seed in any::<u64>(),
    ) {
        let values = raw_values.into_iter().map(|value| f64::from(value) / 10.0).collect::<Vec<_>>();
        let first = bootstrap_distribution(&values, n_boot, seed);
        let second = bootstrap_distribution(&values, n_boot, seed);
        let minimum = values.iter().copied().reduce(f64::min).unwrap();
        let maximum = values.iter().copied().reduce(f64::max).unwrap();

        prop_assert_eq!(&first, &second);
        prop_assert_eq!(first.len(), n_boot);
        prop_assert!(first.iter().all(|value| minimum - 1e-10 <= *value && *value <= maximum + 1e-10));
    }

    #[test]
    fn constant_samples_have_zero_width_bootstrap_intervals(
        raw_value in -1_000_i16..1_000,
        count in 1_usize..64,
        n_boot in 1_usize..128,
        seed in any::<u64>(),
    ) {
        let value = f64::from(raw_value) / 10.0;
        let summary = summarize_values(&vec![value; count], n_boot, seed, 0.05, false);

        prop_assert!((summary.value - value).abs() < 1e-10);
        prop_assert_eq!(summary.count, count);
        prop_assert!((summary.ci_low.unwrap() - value).abs() < 1e-10);
        prop_assert!((summary.ci_high.unwrap() - value).abs() < 1e-10);
    }

    #[test]
    fn pairwise_difference_is_antisymmetric(
        pairs in collection::vec((-1_000_i16..1_000, -1_000_i16..1_000), 1..32),
        n_boot in 1_usize..128,
        seed in any::<u64>(),
    ) {
        let left = pairs.iter().map(|(value, _)| f64::from(*value) / 10.0).collect::<Vec<_>>();
        let right = pairs.iter().map(|(_, value)| f64::from(*value) / 10.0).collect::<Vec<_>>();
        let forward = paired_difference(&left, &right, "left", "right", "generated", n_boot, seed);
        let reverse = paired_difference(&right, &left, "right", "left", "generated", n_boot, seed);

        prop_assert!((forward.diff_mean + reverse.diff_mean).abs() < 1e-10);
        prop_assert!((forward.diff_ci_low + reverse.diff_ci_high).abs() < 1e-10);
        prop_assert!((forward.diff_ci_high + reverse.diff_ci_low).abs() < 1e-10);
        prop_assert_eq!(forward.is_significant, reverse.is_significant);
        prop_assert_eq!(forward.count, reverse.count);
    }

    #[test]
    fn ranking_probabilities_form_distributions(
        model_count in 1_usize..8,
        n_boot in 1_usize..128,
        seed in any::<u64>(),
    ) {
        let values = (0..model_count)
            .map(|index| (format!("model-{index}"), vec![index as f64, (index + 1) as f64]))
            .collect::<BTreeMap<_, _>>();
        let ranking = probabilistic_ranking(&values, n_boot, seed);

        prop_assert_eq!(ranking.len(), model_count);
        for model in ranking {
            let probability_sum = model.rank_probs.values().sum::<f64>();
            prop_assert!((probability_sum - 1.0).abs() < 1e-12);
            prop_assert!((1.0..=model_count as f64).contains(&model.expected_rank));
        }
    }
}

#[test]
fn route_paths_reject_noncanonical_and_overflowing_indices() {
    for value in [
        "",
        "rc:",
        "rc:x:/0",
        "rc:m:",
        "rc:m:0",
        "rc:m://0",
        "rc:m:/00",
        "rc:m:/+1",
        "rc:m:/-1",
        "rc:m:/ 1",
        "rc:m:/1 ",
        "rc:m:/184467440737095516160000000000000000000",
        "rc:m:/0/",
    ] {
        assert!(RoutePath::parse(value).is_err(), "accepted {value:?}");
    }
}
