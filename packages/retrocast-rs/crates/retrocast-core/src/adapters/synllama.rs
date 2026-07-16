use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::Result,
    model::{Route, Target},
    route::AdaptMode,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SynLlamaRoute {
    synthesis_string: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SynLlamaAdapter;

impl Adapter for SynLlamaAdapter {
    fn name(&self) -> &'static str {
        "synllama"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let routes: Vec<SynLlamaRoute> = serde_json::from_value(payload)
            .map_err(|error| common::schema(self.name(), format!("invalid route list: {error}")))?;
        Ok(routes
            .into_iter()
            .enumerate()
            .map(|(index, route)| RawRouteEntry {
                payload: serde_json::to_value(route).expect("SynLlama route is serializable"),
                source_key: source_key.map(str::to_owned),
                source_row_index: None,
                source_record_id: None,
                target_hint_id: None,
                target_hint_smiles: None,
                source_order: Some(index + 1),
            })
            .collect())
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        let route: SynLlamaRoute = serde_json::from_value(raw_route)
            .map_err(|error| common::schema(self.name(), format!("invalid route: {error}")))?;
        let parts: Vec<_> = route
            .synthesis_string
            .split(';')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(common::logic(
                self.name(),
                "adapter.route_string_empty",
                "empty synthesis string",
            ));
        }
        let parsed_target = match chem::normalize(parts.last().expect("parts is nonempty")) {
            Ok((smiles, _)) => smiles,
            Err(_) if mode == AdaptMode::Prune => {
                return Err(common::target_pruned(self.name()));
            }
            Err(error) => return Err(error),
        };
        if let Some(target) = target {
            common::require_target_match(&parsed_target, &target.smiles, &target.id, self.name())?;
        }
        let precursor_map = parse_synthesis(&parts, mode, self.name())?;
        let route_target = common::build_from_precursor_map(
            &parsed_target,
            &precursor_map,
            self.name(),
            mode,
            None,
        )?
        .ok_or_else(|| common::target_pruned(self.name()))?;
        Ok(Route {
            target: route_target,
            annotations: Default::default(),
            schema_version: Default::default(),
        })
    }
}

pub fn parse_synthesis_string(
    synthesis_string: &str,
    mode: AdaptMode,
) -> Result<common::PrecursorMap> {
    let parts: Vec<_> = synthesis_string
        .split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        return Err(common::logic(
            "synllama",
            "adapter.route_string_empty",
            "empty synthesis string",
        ));
    }
    parse_synthesis(&parts, mode, "synllama")
}

fn parse_synthesis(
    parts: &[&str],
    mode: AdaptMode,
    adapter: &'static str,
) -> Result<common::PrecursorMap> {
    let template_indices: Vec<_> = parts
        .iter()
        .enumerate()
        .filter_map(|(index, part)| {
            part.strip_prefix('R')
                .filter(|suffix| {
                    !suffix.is_empty() && suffix.chars().all(|char| char.is_ascii_digit())
                })
                .map(|_| index)
        })
        .collect();
    let mut precursor_map = BTreeMap::new();
    let mut last_product: Option<String> = None;
    let mut reactant_start = 0;
    for template_index in template_indices {
        let product_index = template_index + 1;
        let Some(raw_product) = parts.get(product_index) else {
            return Err(common::logic(
                adapter,
                "adapter.route_string_invalid",
                format!("template {} has no product", parts[template_index]),
            ));
        };
        let product = match chem::normalize(raw_product) {
            Ok((smiles, _)) => smiles,
            Err(_) if mode == AdaptMode::Prune => {
                last_product = None;
                reactant_start = product_index + 1;
                continue;
            }
            Err(error) => return Err(error),
        };
        let mut reactants: Vec<_> = parts[reactant_start..template_index]
            .iter()
            .map(|reactant| (*reactant).to_owned())
            .collect();
        if let Some(previous_product) = last_product {
            reactants.push(previous_product);
        }
        if reactants.is_empty() {
            return Err(common::logic(
                adapter,
                "adapter.route_string_invalid",
                format!("no reactants found for product {raw_product}"),
            ));
        }
        precursor_map.insert(product.clone(), reactants);
        last_product = Some(product.to_string());
        reactant_start = product_index + 1;
    }
    Ok(precursor_map)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::SynLlamaAdapter;
    use crate::{adapters::Adapter, route::AdaptMode};

    #[test]
    fn parses_ranked_template_syntax() {
        let route = SynLlamaAdapter
            .cast(
                json!({"synthesis_string": "C;CC;R1;CCO"}),
                AdaptMode::Strict,
                None,
            )
            .unwrap();
        assert_eq!(route.leaves().len(), 2);
    }

    #[test]
    fn pruning_does_not_carry_product_across_failed_step() {
        let route = SynLlamaAdapter
            .cast(
                json!({"synthesis_string": "C;R1;CC;C;R2;not-smiles;CCC;R3;CCO"}),
                AdaptMode::Prune,
                None,
            )
            .unwrap();
        assert_eq!(route.leaves()[0].value.smiles, "CCC");
    }
}
