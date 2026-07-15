use std::collections::BTreeMap;

use serde_json::{Map, Number, Value, json};

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::Result,
    model::{Route, Target},
    route::AdaptMode,
    schema::CanonicalSmiles,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct DreamRetroErAdapter;

#[derive(Clone, Copy, Debug, Default)]
pub struct RetroStarAdapter;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ParsedReactionString {
    target_smiles: CanonicalSmiles,
    precursor_map: common::PrecursorMap,
    step_scores: BTreeMap<CanonicalSmiles, f64>,
}

impl Adapter for DreamRetroErAdapter {
    fn name(&self) -> &'static str {
        "dreamretro"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        single_route_entry(payload, source_key, self.name(), RouteFlavor::DreamRetro)
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        cast_route(
            raw_route,
            mode,
            target,
            self.name(),
            RouteFlavor::DreamRetro,
        )
    }
}

impl Adapter for RetroStarAdapter {
    fn name(&self) -> &'static str {
        "retrostar"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        single_route_entry(payload, source_key, self.name(), RouteFlavor::RetroStar)
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        cast_route(raw_route, mode, target, self.name(), RouteFlavor::RetroStar)
    }
}

#[derive(Clone, Copy)]
enum RouteFlavor {
    DreamRetro,
    RetroStar,
}

pub fn parse_reaction_string(
    route: &str,
    mode: AdaptMode,
    adapter: &str,
) -> Result<ParsedReactionString> {
    let (adapter, flavor) = match adapter {
        "dreamretro" | "dreamretroer" => ("dreamretro", RouteFlavor::DreamRetro),
        "retrostar" => ("retrostar", RouteFlavor::RetroStar),
        _ => return Err(common::schema("reaction-string", "unsupported adapter")),
    };
    parse_route_string(route, mode, adapter, flavor)
}

fn single_route_entry(
    payload: Value,
    source_key: Option<&str>,
    adapter: &'static str,
    flavor: RouteFlavor,
) -> Result<Vec<RawRouteEntry>> {
    let object = payload
        .as_object()
        .ok_or_else(|| common::schema(adapter, "expected an object"))?;
    if !object.get("succ").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(Vec::new());
    }
    let route = object
        .get("routes")
        .and_then(Value::as_str)
        .filter(|route| !route.is_empty())
        .ok_or_else(|| common::schema(adapter, "no valid routes string found"))?;
    let mut internal = Map::new();
    internal.insert("route".to_owned(), json!(route));
    match flavor {
        RouteFlavor::DreamRetro => {
            for key in [
                "expand_model_call",
                "value_model_call",
                "reaction_nodes_lens",
                "mol_nodes_lens",
            ] {
                if let Some(value) = object.get(key) {
                    internal.insert(key.to_owned(), value.clone());
                }
            }
        }
        RouteFlavor::RetroStar => {
            if let Some(cost) = object.get("route_cost").and_then(Value::as_f64) {
                internal.insert(
                    "route_cost".to_owned(),
                    Value::Number(Number::from_f64(cost).expect("finite JSON number")),
                );
            }
        }
    }
    Ok(vec![RawRouteEntry {
        payload: Value::Object(internal),
        source_key: source_key.map(str::to_owned),
        source_row_index: None,
        source_record_id: None,
        target_hint_id: None,
        target_hint_smiles: None,
        source_order: Some(1),
    }])
}

fn cast_route(
    raw_route: Value,
    mode: AdaptMode,
    target: Option<&Target>,
    adapter: &'static str,
    flavor: RouteFlavor,
) -> Result<Route> {
    let object = raw_route
        .as_object()
        .ok_or_else(|| common::schema(adapter, "expected an extracted route object"))?;
    let route_string = object
        .get("route")
        .or_else(|| object.get("routes"))
        .and_then(Value::as_str)
        .ok_or_else(|| common::schema(adapter, "route string is missing"))?;
    let parsed = parse_route_string(route_string, mode, adapter, flavor)?;
    if let Some(target) = target {
        common::require_target_match(&parsed.target_smiles, &target.smiles, &target.id, adapter)?;
    }
    let annotations = match flavor {
        RouteFlavor::DreamRetro => object
            .iter()
            .filter(|(key, _)| key.as_str() != "route" && key.as_str() != "routes")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        RouteFlavor::RetroStar => object
            .get("route_cost")
            .map(|value| Map::from_iter([("route_cost".to_owned(), value.clone())]))
            .unwrap_or_default(),
    };
    let reaction_annotations = if parsed.step_scores.is_empty() {
        None
    } else {
        Some(
            parsed
                .step_scores
                .into_iter()
                .map(|(product, score)| {
                    (
                        product,
                        Map::from_iter([("step_score".to_owned(), json!(score))]),
                    )
                })
                .collect(),
        )
    };
    let route_target = common::build_from_precursor_map(
        &parsed.target_smiles,
        &parsed.precursor_map,
        adapter,
        mode,
        reaction_annotations.as_ref(),
    )?
    .ok_or_else(|| common::target_pruned(adapter))?;
    Ok(Route {
        target: route_target,
        annotations,
        schema_version: Default::default(),
    })
}

fn parse_route_string(
    route: &str,
    mode: AdaptMode,
    adapter: &'static str,
    flavor: RouteFlavor,
) -> Result<ParsedReactionString> {
    let steps: Vec<_> = route.split('|').collect();
    if steps.first().is_none_or(|step| step.is_empty()) {
        return Err(common::logic(
            adapter,
            "adapter.route_string_empty",
            "empty route string",
        ));
    }
    if steps.len() == 1 && !steps[0].contains('>') {
        let (target_smiles, _) = chem::normalize(steps[0])?;
        return Ok(ParsedReactionString {
            target_smiles,
            precursor_map: BTreeMap::new(),
            step_scores: BTreeMap::new(),
        });
    }

    let first_product = steps[0].split('>').next().unwrap_or_default();
    let (target_smiles, _) = chem::normalize(first_product)?;
    let mut precursor_map = BTreeMap::new();
    let mut step_scores = BTreeMap::new();
    for step in steps {
        let parts: Vec<_> = step.split('>').collect();
        if parts.len() != 3 {
            return Err(common::logic(
                adapter,
                "adapter.route_string_invalid",
                format!("invalid reaction step {step:?}"),
            ));
        }
        let product = match chem::normalize(parts[0]) {
            Ok((product, _)) => product,
            Err(_) if mode == AdaptMode::Prune && parts[0] != first_product => continue,
            Err(error) => return Err(error),
        };
        let reactants = parts[2]
            .split('.')
            .map(str::trim)
            .filter(|reactant| !reactant.is_empty())
            .map(str::to_owned)
            .collect();
        precursor_map.insert(product.clone(), reactants);
        if matches!(flavor, RouteFlavor::RetroStar) {
            if let Ok(score) = parts[1].parse() {
                step_scores.insert(product, score);
            }
        }
    }
    Ok(ParsedReactionString {
        target_smiles,
        precursor_map,
        step_scores,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{DreamRetroErAdapter, RetroStarAdapter};
    use crate::{adapters::Adapter, route::AdaptMode};

    #[test]
    fn retrostar_preserves_route_cost_and_step_score() {
        let entry = RetroStarAdapter
            .entries(
                json!({"succ": true, "routes": "CCO>0.8>C.CC", "route_cost": 1.2}),
                None,
            )
            .unwrap()
            .pop()
            .unwrap();
        let route = RetroStarAdapter
            .cast(entry.payload, AdaptMode::Strict, None)
            .unwrap();
        assert_eq!(route.annotations["route_cost"], 1.2);
        assert_eq!(
            route.target.product_of.unwrap().annotations["step_score"],
            0.8
        );
    }

    #[test]
    fn dreamretro_preserves_run_annotations() {
        let entry = DreamRetroErAdapter
            .entries(
                json!({"succ": true, "routes": "CCO>>C", "expand_model_call": 3}),
                None,
            )
            .unwrap()
            .pop()
            .unwrap();
        let route = DreamRetroErAdapter
            .cast(entry.payload, AdaptMode::Strict, None)
            .unwrap();
        assert_eq!(route.annotations["expand_model_call"], 3);
    }

    #[test]
    fn unsuccessful_search_yields_no_entries() {
        assert!(
            RetroStarAdapter
                .entries(json!({"succ": false}), None)
                .unwrap()
                .is_empty()
        );
    }
}
