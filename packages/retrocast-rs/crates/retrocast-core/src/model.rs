use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    error::{EngineError, Result},
    schema::{CanonicalSmiles, InchiKey, ReactionSmiles, SchemaVersion},
};

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, Deserialize)]
pub struct RawNode {
    #[serde(rename = "type")]
    pub kind: String,
    pub smiles: String,
    #[serde(default)]
    pub in_stock: bool,
    #[serde(default)]
    pub children: Vec<RawNode>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
    #[serde(default)]
    pub scores: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Reaction {
    pub reactants: Vec<Molecule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapped_reaction_smiles: Option<ReactionSmiles>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reagents: Option<Vec<CanonicalSmiles>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solvents: Option<Vec<CanonicalSmiles>>,
    #[serde(default)]
    pub annotations: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Molecule {
    pub smiles: CanonicalSmiles,
    pub inchikey: InchiKey,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub product_of: Option<Box<Reaction>>,
    #[serde(default)]
    pub annotations: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Route {
    pub target: Molecule,
    #[serde(default)]
    pub annotations: serde_json::Map<String, Value>,
    #[serde(default = "schema_version")]
    pub schema_version: SchemaVersion,
}

fn schema_version() -> SchemaVersion {
    SchemaVersion::V2
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FailureRecord {
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_smiles: Option<CanonicalSmiles>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_inchikey: Option<InchiKey>,
    #[serde(default)]
    pub context: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub rank: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<Route>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailureRecord>,
}

#[derive(Deserialize)]
struct CandidateWire {
    rank: usize,
    #[serde(default)]
    route: Option<Route>,
    #[serde(default)]
    failure: Option<FailureRecord>,
}

impl<'de> Deserialize<'de> for Candidate {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let candidate = CandidateWire::deserialize(deserializer)?;
        if candidate.rank == 0 {
            return Err(serde::de::Error::custom(
                "candidate rank must be at least 1",
            ));
        }
        if candidate.route.is_some() == candidate.failure.is_some() {
            return Err(serde::de::Error::custom(
                "candidate must contain exactly one of route or failure",
            ));
        }
        Ok(Self {
            rank: candidate.rank,
            route: candidate.route,
            failure: candidate.failure,
        })
    }
}

pub type Predictions = BTreeMap<String, Vec<Candidate>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Target {
    pub id: String,
    pub smiles: CanonicalSmiles,
    pub inchikey: InchiKey,
    #[serde(default)]
    pub acceptable_routes: Vec<Route>,
    #[serde(default)]
    pub annotations: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Constraint {
    pub kind: String,
    #[serde(flatten)]
    pub fields: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Task {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub targets: BTreeMap<String, Target>,
    #[serde(default)]
    pub default_constraints: Vec<Constraint>,
    #[serde(default)]
    pub constraints: BTreeMap<String, Vec<Constraint>>,
    #[serde(default)]
    pub metric_label: Option<String>,
    #[serde(default)]
    pub annotations: serde_json::Map<String, Value>,
    #[serde(default = "schema_version")]
    pub schema_version: SchemaVersion,
}

#[derive(Deserialize)]
struct TaskWire {
    name: String,
    #[serde(default)]
    description: String,
    targets: BTreeMap<String, Target>,
    #[serde(default)]
    default_constraints: Vec<Constraint>,
    #[serde(default)]
    constraints: BTreeMap<String, Vec<Constraint>>,
    #[serde(default)]
    metric_label: Option<String>,
    #[serde(default)]
    annotations: serde_json::Map<String, Value>,
    #[serde(default = "schema_version")]
    schema_version: SchemaVersion,
}

impl<'de> Deserialize<'de> for Task {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = TaskWire::deserialize(deserializer)?;
        let task = Self {
            name: value.name,
            description: value.description,
            targets: value.targets,
            default_constraints: value.default_constraints,
            constraints: value.constraints,
            metric_label: value.metric_label,
            annotations: value.annotations,
            schema_version: value.schema_version,
        };
        task.validate().map_err(serde::de::Error::custom)?;
        Ok(task)
    }
}

impl Task {
    /// Check invariants that require comparing fields across the complete task.
    pub fn validate(&self) -> Result<()> {
        for (target_key, target) in &self.targets {
            if target_key != &target.id {
                return Err(EngineError::InvalidTask(format!(
                    "target key {target_key:?} does not match Target.id {:?}",
                    target.id
                )));
            }
        }
        validate_constraint_set("default_constraints", &self.default_constraints)?;
        for (target_id, constraints) in &self.constraints {
            if !self.targets.contains_key(target_id) {
                return Err(EngineError::InvalidTask(format!(
                    "constraints reference unknown target {target_id:?}"
                )));
            }
            validate_constraint_set(&format!("constraints[{target_id:?}]"), constraints)?;
        }
        Ok(())
    }

    pub fn effective_constraints(&self, target_id: &str) -> Vec<Constraint> {
        let mut by_kind: BTreeMap<&str, &Constraint> = self
            .default_constraints
            .iter()
            .map(|constraint| (constraint.kind.as_str(), constraint))
            .collect();
        if let Some(overrides) = self.constraints.get(target_id) {
            for constraint in overrides {
                by_kind.insert(constraint.kind.as_str(), constraint);
            }
        }
        by_kind.into_values().cloned().collect()
    }

    pub fn derived_metric_label(&self) -> String {
        if let Some(label) = &self.metric_label {
            return label.clone();
        }
        let stocks: std::collections::BTreeSet<&str> = self
            .default_constraints
            .iter()
            .filter(|c| c.kind == "retrocast.stock_termination")
            .filter_map(|c| c.fields.get("stock")?.as_str())
            .collect();
        let mut parts = Vec::new();
        match stocks.len() {
            1 => parts.push(stocks.first().copied().unwrap().to_owned()),
            n if n > 1 => parts.push("stocks".to_owned()),
            _ => {}
        }
        let all = self
            .default_constraints
            .iter()
            .chain(self.constraints.values().flatten());
        let kinds: std::collections::BTreeSet<&str> = all.map(|c| c.kind.as_str()).collect();
        if kinds.contains("retrocast.required_leaves") {
            parts.push("leaf".to_owned());
        }
        if kinds.contains("retrocast.route_depth") {
            parts.push("depth".to_owned());
        }
        if parts.is_empty() {
            "task".to_owned()
        } else {
            parts.join("+")
        }
    }
}

fn validate_constraint_set(label: &str, constraints: &[Constraint]) -> Result<()> {
    let mut kinds = BTreeSet::new();
    for constraint in constraints {
        if !kinds.insert(constraint.kind.as_str()) {
            return Err(EngineError::InvalidTask(format!(
                "{label} contains duplicate constraint kind {:?}",
                constraint.kind
            )));
        }
        validate_constraint(constraint)?;
    }
    Ok(())
}

fn validate_constraint(constraint: &Constraint) -> Result<()> {
    match constraint.kind.as_str() {
        "retrocast.stock_termination" => {
            let stock = constraint
                .fields
                .get("stock")
                .and_then(Value::as_str)
                .filter(|stock| !stock.is_empty())
                .ok_or_else(|| {
                    EngineError::InvalidTask(
                        "stock-termination constraint requires a non-empty stock".to_owned(),
                    )
                })?;
            if stock.trim() != stock {
                return Err(EngineError::InvalidTask(
                    "stock-termination stock cannot start or end with whitespace".to_owned(),
                ));
            }
        }
        "retrocast.required_leaves" => {
            let smiles = constraint
                .fields
                .get("smiles")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    EngineError::InvalidTask(
                        "required-leaves constraint requires a smiles array".to_owned(),
                    )
                })?;
            if smiles
                .iter()
                .any(|value| value.as_str().is_none_or(|smiles| smiles.trim().is_empty()))
            {
                return Err(EngineError::InvalidTask(
                    "required-leaves smiles must be non-empty strings".to_owned(),
                ));
            }
        }
        "retrocast.route_depth" => {
            let maximum = constraint.fields.get("max_depth").ok_or_else(|| {
                EngineError::InvalidTask("route-depth constraint requires max_depth".to_owned())
            })?;
            let valid = maximum.as_u64().is_some_and(|value| value > 0)
                || maximum
                    .as_str()
                    .is_some_and(|value| matches!(value, "short" | "medium" | "long"));
            if !valid {
                return Err(EngineError::InvalidTask(
                    "route-depth max_depth must be a positive integer or short, medium, or long"
                        .to_owned(),
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CheckResult {
    pub code: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default)]
    pub details: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TierResult {
    pub status: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<CheckResult>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RouteValidity {
    #[serde(default)]
    pub tiers: BTreeMap<u8, TierResult>,
    #[serde(default)]
    pub reactions: Vec<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConstraintResult {
    pub status: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checks: Vec<CheckResult>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScoredCandidate {
    pub rank: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<Route>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<FailureRecord>,
    pub validity: RouteValidity,
    pub constraints: ConstraintResult,
    #[serde(default, skip_serializing_if = "is_false")]
    pub matches_acceptable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_acceptable_index: Option<usize>,
}

#[derive(Deserialize)]
struct ScoredCandidateWire {
    rank: usize,
    #[serde(default)]
    route: Option<Route>,
    #[serde(default)]
    failure: Option<FailureRecord>,
    validity: RouteValidity,
    constraints: ConstraintResult,
    #[serde(default)]
    matches_acceptable: bool,
    #[serde(default)]
    matched_acceptable_index: Option<usize>,
}

impl<'de> Deserialize<'de> for ScoredCandidate {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let candidate = ScoredCandidateWire::deserialize(deserializer)?;
        if candidate.rank == 0 {
            return Err(serde::de::Error::custom(
                "candidate rank must be at least 1",
            ));
        }
        if candidate.route.is_some() == candidate.failure.is_some() {
            return Err(serde::de::Error::custom(
                "candidate must contain exactly one of route or failure",
            ));
        }
        Ok(Self {
            rank: candidate.rank,
            route: candidate.route,
            failure: candidate.failure,
            validity: candidate.validity,
            constraints: candidate.constraints,
            matches_acceptable: candidate.matches_acceptable,
            matched_acceptable_index: candidate.matched_acceptable_index,
        })
    }
}

impl ScoredCandidate {
    pub fn satisfies_validity(&self, tier: u8) -> bool {
        self.validity
            .tiers
            .get(&tier)
            .is_some_and(|result| result.status == "pass")
    }

    pub fn satisfies_task(&self) -> bool {
        self.constraints.status == "pass"
    }

    pub fn satisfies_solv(&self, tier: u8) -> bool {
        self.satisfies_validity(tier) && self.satisfies_task()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TargetResult {
    pub target: Target,
    pub effective_constraints: Vec<Constraint>,
    #[serde(default)]
    pub candidates: Vec<ScoredCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_time: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Evaluation {
    pub task: Task,
    pub tiers: Vec<u8>,
    pub metric_label: String,
    pub acceptable_match_level: String,
    pub acceptable_route_match: String,
    pub targets: BTreeMap<String, TargetResult>,
    #[serde(default = "schema_version")]
    pub schema_version: SchemaVersion,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReliabilityFlag {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetricSummary {
    pub value: f64,
    pub count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_low: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_high: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reliability: Option<ReliabilityFlag>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RuntimeSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_wall_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_wall_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cpu_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_cpu_time: Option<f64>,
    pub timed_target_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnalysisReport {
    #[serde(default = "schema_version")]
    pub schema_version: SchemaVersion,
    pub metrics: BTreeMap<String, MetricSummary>,
    pub by_stratum: BTreeMap<String, BTreeMap<String, MetricSummary>>,
    pub bootstrap_resamples: usize,
    pub runtime: RuntimeSummary,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ExecutionStats {
    #[serde(default)]
    pub wall_time: BTreeMap<String, f64>,
    #[serde(default)]
    pub cpu_time: BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct ExecutionStatsWire {
    #[serde(default)]
    wall_time: BTreeMap<String, f64>,
    #[serde(default)]
    cpu_time: BTreeMap<String, f64>,
}

impl<'de> Deserialize<'de> for ExecutionStats {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = ExecutionStatsWire::deserialize(deserializer)?;
        let stats = Self {
            wall_time: value.wall_time,
            cpu_time: value.cpu_time,
        };
        stats.validate().map_err(serde::de::Error::custom)?;
        Ok(stats)
    }
}

impl ExecutionStats {
    /// Reject measurements that cannot represent elapsed time.
    pub fn validate(&self) -> Result<()> {
        for (kind, values) in [("wall_time", &self.wall_time), ("cpu_time", &self.cpu_time)] {
            for (target_id, value) in values {
                if target_id.is_empty() {
                    return Err(EngineError::InvalidExecutionStats(format!(
                        "{kind} contains an empty target id"
                    )));
                }
                if !value.is_finite() || *value < 0.0 {
                    return Err(EngineError::InvalidExecutionStats(format!(
                        "{kind}[{target_id:?}] must be a finite, non-negative number"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Candidate, Constraint, ExecutionStats, ScoredCandidate, Task};

    #[test]
    fn candidate_requires_one_nonzero_ranked_outcome() {
        assert!(
            serde_json::from_str::<Candidate>(r#"{"rank": 0, "failure": {"code": "x"}}"#).is_err()
        );
        assert!(serde_json::from_str::<Candidate>(r#"{"rank": 1}"#).is_err());
        assert!(
            serde_json::from_str::<Candidate>(
                r#"{"rank": 1, "route": {"target": {"smiles": "C", "inchikey": "C"}}, "failure": {"code": "x"}}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<Candidate>(r#"{"rank": 1, "failure": {"code": "x"}}"#).is_ok()
        );
    }

    #[test]
    fn scored_candidate_has_the_same_outcome_invariant() {
        assert!(
            serde_json::from_str::<ScoredCandidate>(
                r#"{"rank": 1, "validity": {}, "constraints": {"status": "pass"}}"#
            )
            .is_err()
        );
    }

    #[test]
    fn task_validation_checks_cross_field_invariants() {
        let mut task: Task = serde_json::from_value(json!({
            "name": "invalid",
            "targets": {
                "map-key": {
                    "id": "map-key",
                    "smiles": "CCO",
                    "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
                }
            }
        }))
        .unwrap();
        task.targets.get_mut("map-key").unwrap().id = "different-id".to_owned();
        assert!(
            task.validate()
                .unwrap_err()
                .to_string()
                .contains("does not match")
        );

        let mut task: Task = serde_json::from_value(json!({
            "name": "duplicate-constraint",
            "targets": {}
        }))
        .unwrap();
        task.default_constraints = vec![
            Constraint {
                kind: "retrocast.route_depth".to_owned(),
                fields: serde_json::Map::from_iter([("max_depth".to_owned(), json!(3))]),
            },
            Constraint {
                kind: "retrocast.route_depth".to_owned(),
                fields: serde_json::Map::from_iter([("max_depth".to_owned(), json!(5))]),
            },
        ];
        assert!(
            task.validate()
                .unwrap_err()
                .to_string()
                .contains("duplicate constraint")
        );

        let unknown_target = serde_json::from_value::<Task>(json!({
            "name": "unknown-target",
            "targets": {},
            "constraints": {
                "missing-target": []
            }
        }))
        .unwrap_err();
        assert!(unknown_target.to_string().contains("unknown target"));

        let extensible_constraint = serde_json::from_value::<Task>(json!({
            "name": "extension",
            "targets": {},
            "default_constraints": [{
                "kind": "example.custom_constraint",
                "parameter": true
            }]
        }))
        .unwrap();
        assert_eq!(
            extensible_constraint.default_constraints[0].kind,
            "example.custom_constraint"
        );
    }

    #[test]
    fn execution_stats_reject_negative_measurements() {
        let error = serde_json::from_value::<ExecutionStats>(json!({
            "wall_time": {"target-1": -0.5}
        }))
        .unwrap_err();
        assert!(error.to_string().contains("non-negative"));
    }
}
