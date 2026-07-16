use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::schema::{CanonicalSmiles, InchiKey, ReactionSmiles, SchemaVersion};

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

#[derive(Clone, Debug, Deserialize, Serialize)]
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

impl Task {
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

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ExecutionStats {
    #[serde(default)]
    pub wall_time: BTreeMap<String, f64>,
    #[serde(default)]
    pub cpu_time: BTreeMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::{Map, Value, json};

    use super::{Candidate, ScoredCandidate};

    fn valid_route() -> Value {
        json!({
            "target": {
                "smiles": "CCO",
                "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
            },
            "schema_version": "2"
        })
    }

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

    proptest! {
        #[test]
        fn candidate_deserialization_enforces_exactly_one_ranked_outcome(
            rank in any::<u16>(),
            has_route in any::<bool>(),
            has_failure in any::<bool>(),
        ) {
            let mut candidate = Map::from_iter([("rank".to_owned(), Value::from(rank))]);
            if has_route {
                candidate.insert("route".to_owned(), valid_route());
            }
            if has_failure {
                candidate.insert("failure".to_owned(), json!({"code": "provider.failure"}));
            }

            let parsed = serde_json::from_value::<Candidate>(Value::Object(candidate));
            prop_assert_eq!(parsed.is_ok(), rank > 0 && (has_route ^ has_failure));
        }
    }
}
