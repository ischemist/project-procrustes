use std::collections::{BTreeMap, BTreeSet};

use retrocast_core::{
    analyze::analyze,
    chem,
    model::{
        Candidate, Constraint, ConstraintResult, Molecule, Reaction, Route, RouteValidity,
        ScoredCandidate, Target, TargetResult, Task, TierResult,
    },
    schema::ReactionSmiles,
    score::{Stocks, score},
};
use serde_json::{Map, Value, json};

#[test]
fn analysis_keeps_failed_targets_in_headline_denominators_and_filters_reconstruction_by_task() {
    let acceptable = two_step_route();
    let passing = scored(2, acceptable.clone(), true, true, true);
    let rejected_but_valid = scored(1, acceptable.clone(), true, false, true);
    let solved_target = target("solved", "CCC", vec![acceptable]);
    let failed = target("failed", "CO", Vec::new());
    let task = task(BTreeMap::from([
        ("solved".to_owned(), solved_target.clone()),
        ("failed".to_owned(), failed.clone()),
    ]));
    let evaluation = retrocast_core::model::Evaluation {
        task,
        tiers: vec![0],
        metric_label: "contract".to_owned(),
        acceptable_match_level: "full".to_owned(),
        acceptable_route_match: "prefix".to_owned(),
        targets: BTreeMap::from([
            (
                "solved".to_owned(),
                TargetResult {
                    target: solved_target,
                    effective_constraints: Vec::new(),
                    candidates: vec![rejected_but_valid, passing],
                    wall_time: Some(4.0),
                    cpu_time: None,
                },
            ),
            (
                "failed".to_owned(),
                TargetResult {
                    target: failed,
                    effective_constraints: Vec::new(),
                    candidates: Vec::new(),
                    wall_time: None,
                    cpu_time: Some(3.0),
                },
            ),
        ]),
        schema_version: Default::default(),
    };

    let report = analyze(&evaluation, &[1], &[1], 64, 11, 2).unwrap();
    assert_eq!(report.metrics["tier_0_validity_rate"].value, 0.5);
    assert_eq!(report.metrics["tier_0_validity_mrr"].value, 0.5);
    assert_eq!(report.metrics["solv_0[contract]_rate"].value, 0.5);
    assert_eq!(report.metrics["solv_0[contract]_mrr"].value, 0.25);
    assert_eq!(
        report.metrics["acceptable_reconstruction_top_1[contract]"].value, 1.0,
        "the rejected rank-1 route must not consume the top-k task-satisfying slot"
    );
    assert_eq!(
        report.metrics["acceptable_reconstruction_top_1[contract]"].count,
        1
    );
    assert_eq!(report.runtime.total_wall_time, Some(4.0));
    assert_eq!(report.runtime.total_cpu_time, Some(3.0));
    assert_eq!(report.runtime.timed_target_count, 1);
}

#[test]
fn scoring_combines_constraints_uses_target_overrides_and_prefers_deepest_prefix_match() {
    let shallow = one_step_route();
    let deep = two_step_route();
    let mut target = target("propane", "CCC", vec![shallow, deep.clone()]);
    target
        .annotations
        .insert("purpose".to_owned(), json!("contract"));
    let default_constraints = vec![
        constraint(
            "retrocast.stock_termination",
            json!({"stock": "building-blocks"}),
        ),
        constraint("retrocast.required_leaves", json!({"smiles": ["C"]})),
        constraint("retrocast.route_depth", json!({"max_depth": 1})),
    ];
    let task = Task {
        targets: BTreeMap::from([("propane".to_owned(), target)]),
        default_constraints,
        constraints: BTreeMap::from([(
            "propane".to_owned(),
            vec![constraint("retrocast.route_depth", json!({"max_depth": 2}))],
        )]),
        ..task(BTreeMap::new())
    };
    let predictions = BTreeMap::from([(
        "propane".to_owned(),
        vec![Candidate {
            rank: 1,
            route: Some(deep),
            failure: None,
        }],
    )]);
    let stocks: Stocks = BTreeMap::from([(
        "building-blocks".to_owned(),
        BTreeSet::from([inchikey("C")]),
    )]);

    let prefix = score(&predictions, &task, &stocks, "full", "prefix", None, 2).unwrap();
    let result = &prefix.targets["propane"];
    assert_eq!(result.effective_constraints.len(), 3);
    assert_eq!(
        result
            .effective_constraints
            .iter()
            .find(|item| item.kind == "retrocast.route_depth")
            .unwrap()
            .fields["max_depth"],
        2
    );
    assert!(result.candidates[0].satisfies_task());
    assert_eq!(result.candidates[0].matched_acceptable_index, Some(1));

    let exact = score(&predictions, &task, &stocks, "full", "exact", None, 1).unwrap();
    assert_eq!(
        exact.targets["propane"].candidates[0].matched_acceptable_index,
        Some(1)
    );
}

fn scored(
    rank: usize,
    route: Route,
    valid: bool,
    satisfies_task: bool,
    matched: bool,
) -> ScoredCandidate {
    ScoredCandidate {
        rank,
        route: Some(route),
        failure: None,
        validity: RouteValidity {
            tiers: BTreeMap::from([(
                0,
                TierResult {
                    status: if valid { "pass" } else { "fail" }.to_owned(),
                    checks: Vec::new(),
                },
            )]),
            reactions: Vec::new(),
        },
        constraints: ConstraintResult {
            status: if satisfies_task { "pass" } else { "fail" }.to_owned(),
            checks: Vec::new(),
        },
        matches_acceptable: matched,
        matched_acceptable_index: matched.then_some(0),
    }
}

fn task(targets: BTreeMap<String, Target>) -> Task {
    Task {
        name: "evaluation-contract".to_owned(),
        description: String::new(),
        targets,
        default_constraints: Vec::new(),
        constraints: BTreeMap::new(),
        metric_label: None,
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

fn target(id: &str, smiles: &str, acceptable_routes: Vec<Route>) -> Target {
    let (smiles, inchikey) = chem::normalize(smiles).unwrap();
    Target {
        id: id.to_owned(),
        smiles,
        inchikey,
        acceptable_routes,
        annotations: Map::new(),
    }
}

fn constraint(kind: &str, fields: Value) -> Constraint {
    Constraint {
        kind: kind.to_owned(),
        fields: fields.as_object().unwrap().clone(),
    }
}

fn one_step_route() -> Route {
    Route {
        target: molecule(
            "CCC",
            Some(reaction(
                vec![molecule("CC", None), molecule("C", None)],
                "CC.C>>CCC",
            )),
        ),
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

fn two_step_route() -> Route {
    Route {
        target: molecule(
            "CCC",
            Some(reaction(
                vec![
                    molecule(
                        "CC",
                        Some(reaction(
                            vec![molecule("C", None), molecule("C", None)],
                            "C.C>>CC",
                        )),
                    ),
                    molecule("C", None),
                ],
                "CC.C>>CCC",
            )),
        ),
        annotations: Map::new(),
        schema_version: Default::default(),
    }
}

fn molecule(smiles: &str, product_of: Option<Reaction>) -> Molecule {
    let (smiles, inchikey) = chem::normalize(smiles).unwrap();
    Molecule {
        smiles,
        inchikey,
        product_of: product_of.map(Box::new),
        annotations: Map::new(),
    }
}

fn reaction(reactants: Vec<Molecule>, smiles: &str) -> Reaction {
    Reaction {
        reactants,
        mapped_reaction_smiles: Some(ReactionSmiles::try_from(smiles).unwrap()),
        template: None,
        reagents: None,
        solvents: None,
        annotations: Map::new(),
    }
}

fn inchikey(smiles: &str) -> String {
    chem::normalize(smiles).unwrap().1.into_string()
}
