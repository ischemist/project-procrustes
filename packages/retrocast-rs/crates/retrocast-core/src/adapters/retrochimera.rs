use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::Result,
    model::{Route, Target},
    route::AdaptMode,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReactionInput {
    reactants: Vec<String>,
    product: String,
    probability: f64,
    #[serde(default)]
    metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RouteInput {
    reactions: Vec<ReactionInput>,
    num_steps: usize,
    step_probability_min: f64,
    step_probability_product: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Output {
    routes: Vec<RouteInput>,
    num_routes: usize,
    #[serde(default)]
    num_routes_initial_extraction: usize,
    #[serde(default)]
    target_is_purchasable: bool,
    #[serde(default)]
    num_model_calls_total: usize,
    #[serde(default)]
    num_model_calls_new: usize,
    #[serde(default)]
    num_model_calls_cached: usize,
    #[serde(default)]
    num_nodes_explored: usize,
    #[serde(default)]
    time_taken_s_search: f64,
    #[serde(default)]
    time_taken_s_extraction: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct ResultPayload {
    #[serde(default)]
    outputs: Option<Vec<Output>>,
    #[serde(default)]
    error: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Deserialize)]
struct Data {
    smiles: String,
    result: ResultPayload,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExtractedRoute {
    target_smiles: String,
    route: RouteInput,
    annotations: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RetroChimeraAdapter;

impl Adapter for RetroChimeraAdapter {
    fn name(&self) -> &'static str {
        "retrochimera"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let data: Data = serde_json::from_value(payload)
            .map_err(|error| common::schema(self.name(), format!("invalid output: {error}")))?;
        if let Some(error) = data.result.error {
            let kind = error
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(common::logic(
                self.name(),
                "adapter.route_transform_failed",
                format!("model reported {kind}: {message}"),
            ));
        }
        let outputs = data.result.outputs.ok_or_else(|| {
            common::logic(
                self.name(),
                "adapter.route_transform_failed",
                "validated payload is missing result outputs",
            )
        })?;
        let mut entries = Vec::new();
        for output in outputs {
            let annotations = output_annotations(&output);
            for route in output.routes {
                entries.push(RawRouteEntry {
                    payload: serde_json::to_value(ExtractedRoute {
                        target_smiles: data.smiles.clone(),
                        route,
                        annotations: annotations.clone(),
                    })
                    .expect("RetroChimera route is serializable"),
                    source_key: source_key.map(str::to_owned),
                    source_row_index: None,
                    source_record_id: None,
                    target_hint_id: None,
                    target_hint_smiles: None,
                    source_order: Some(entries.len() + 1),
                });
            }
        }
        Ok(entries)
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        let extracted: ExtractedRoute = serde_json::from_value(raw_route)
            .map_err(|error| common::schema(self.name(), format!("invalid route: {error}")))?;
        let root_smiles = match chem::normalize(&extracted.target_smiles) {
            Ok((smiles, _)) => smiles,
            Err(_) if mode == AdaptMode::Prune => {
                return Err(common::target_pruned(self.name()));
            }
            Err(error) => return Err(error),
        };
        if let Some(target) = target {
            common::require_target_match(&root_smiles, &target.smiles, &target.id, self.name())?;
        }
        let mut precursor_map = BTreeMap::new();
        let mut reaction_annotations = BTreeMap::new();
        for reaction in extracted.route.reactions {
            let product = match chem::normalize(&reaction.product) {
                Ok((smiles, _)) => smiles,
                Err(_) if mode == AdaptMode::Prune => continue,
                Err(error) => return Err(error),
            };
            precursor_map.insert(product.clone(), reaction.reactants);
            let mut annotations = reaction.metadata;
            annotations.insert("probability".to_owned(), json!(reaction.probability));
            reaction_annotations.insert(product, annotations);
        }
        let target_molecule = common::build_from_precursor_map(
            &root_smiles,
            &precursor_map,
            self.name(),
            mode,
            Some(&reaction_annotations),
        )?
        .ok_or_else(|| common::target_pruned(self.name()))?;
        Ok(Route {
            target: target_molecule,
            annotations: extracted.annotations,
            schema_version: Default::default(),
        })
    }
}

fn output_annotations(output: &Output) -> Map<String, Value> {
    Map::from_iter([
        ("num_routes".to_owned(), json!(output.num_routes)),
        (
            "num_routes_initial_extraction".to_owned(),
            json!(output.num_routes_initial_extraction),
        ),
        (
            "target_is_purchasable".to_owned(),
            json!(output.target_is_purchasable),
        ),
        (
            "num_model_calls_total".to_owned(),
            json!(output.num_model_calls_total),
        ),
        (
            "num_model_calls_new".to_owned(),
            json!(output.num_model_calls_new),
        ),
        (
            "num_model_calls_cached".to_owned(),
            json!(output.num_model_calls_cached),
        ),
        (
            "num_nodes_explored".to_owned(),
            json!(output.num_nodes_explored),
        ),
        (
            "time_taken_s_search".to_owned(),
            json!(output.time_taken_s_search),
        ),
        (
            "time_taken_s_extraction".to_owned(),
            json!(output.time_taken_s_extraction),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::RetroChimeraAdapter;
    use crate::{adapters::Adapter, route::AdaptMode};

    #[test]
    fn flattens_outputs_and_preserves_route_and_reaction_annotations() {
        let entry = RetroChimeraAdapter
            .entries(
                json!({
                    "smiles": "CCO",
                    "result": {"outputs": [{
                        "routes": [{
                            "reactions": [{"product": "CCO", "reactants": ["C"], "probability": 0.8, "metadata": {"source": "model"}}],
                            "num_steps": 1,
                            "step_probability_min": 0.8,
                            "step_probability_product": 0.8
                        }],
                        "num_routes": 1,
                        "num_nodes_explored": 4
                    }]}
                }),
                None,
            )
            .unwrap()
            .pop()
            .unwrap();
        let route = RetroChimeraAdapter
            .cast(entry.payload, AdaptMode::Strict, None)
            .unwrap();
        assert_eq!(route.annotations["num_nodes_explored"], 4);
        let annotations = &route.target.product_of.unwrap().annotations;
        assert_eq!(annotations["probability"], 0.8);
        assert_eq!(annotations["source"], "model");
    }

    #[test]
    fn model_reported_error_is_not_an_empty_result() {
        assert!(
            RetroChimeraAdapter
                .entries(
                    json!({"smiles": "CCO", "result": {"error": {"type": "search", "message": "failed"}}}),
                    None
                )
                .is_err()
        );
    }
}
