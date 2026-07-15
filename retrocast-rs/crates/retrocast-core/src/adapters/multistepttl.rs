use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::Result,
    model::{Molecule, Route, Target},
    route::AdaptMode,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TtlReaction {
    product: String,
    reactants: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TtlRoute {
    reactions: Vec<TtlReaction>,
    #[serde(default)]
    metadata: Map<String, Value>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MultiStepTtlAdapter;

impl Adapter for MultiStepTtlAdapter {
    fn name(&self) -> &'static str {
        "multistepttl"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let routes: Vec<TtlRoute> = serde_json::from_value(payload)
            .map_err(|error| common::schema(self.name(), format!("invalid route list: {error}")))?;
        Ok(routes
            .into_iter()
            .enumerate()
            .map(|(index, route)| RawRouteEntry {
                payload: serde_json::to_value(route).expect("TTL route is serializable"),
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
        let route: TtlRoute = serde_json::from_value(raw_route)
            .map_err(|error| common::schema(self.name(), format!("invalid route: {error}")))?;
        if route.reactions.is_empty() {
            let target = target.ok_or_else(|| {
                common::logic(
                    self.name(),
                    "adapter.route_transform_failed",
                    "zero-reaction route needs a target",
                )
            })?;
            let (smiles, inchikey) = chem::normalize(&target.smiles)?;
            return Ok(Route {
                target: Molecule {
                    smiles,
                    inchikey,
                    product_of: None,
                    annotations: Map::new(),
                },
                annotations: route.metadata,
                schema_version: Default::default(),
            });
        }

        let root_raw = &route.reactions[0].product;
        let root_smiles = match chem::normalize(root_raw) {
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
        for reaction in route.reactions {
            match chem::normalize(&reaction.product) {
                Ok((product, _)) => {
                    precursor_map.insert(product, reaction.reactants);
                }
                Err(_) if mode == AdaptMode::Prune => {}
                Err(error) => return Err(error),
            }
        }
        let target_molecule = common::build_from_precursor_map(
            &root_smiles,
            &precursor_map,
            self.name(),
            mode,
            None,
        )?
        .ok_or_else(|| common::target_pruned(self.name()))?;
        Ok(Route {
            target: target_molecule,
            annotations: route.metadata,
            schema_version: Default::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::MultiStepTtlAdapter;
    use crate::{adapters::Adapter, model::Target, route::AdaptMode};

    fn target() -> Target {
        let (smiles, inchikey) = crate::chem::normalize("CCO").unwrap();
        Target {
            id: "target".to_owned(),
            smiles,
            inchikey,
            acceptable_routes: Vec::new(),
            annotations: Default::default(),
        }
    }

    #[test]
    fn preserves_metadata_and_duplicate_reactants() {
        let route = MultiStepTtlAdapter
            .cast(
                json!({
                    "reactions": [{"product": "CCO", "reactants": ["C", "C"]}],
                    "metadata": {"score": 0.7}
                }),
                AdaptMode::Strict,
                Some(&target()),
            )
            .unwrap();
        assert_eq!(route.leaves().len(), 2);
        assert_eq!(route.annotations["score"], 0.7);
    }

    #[test]
    fn accepts_zero_reaction_route_only_with_target() {
        assert!(
            MultiStepTtlAdapter
                .cast(json!({"reactions": []}), AdaptMode::Strict, Some(&target()))
                .is_ok()
        );
        assert!(
            MultiStepTtlAdapter
                .cast(json!({"reactions": []}), AdaptMode::Strict, None)
                .is_err()
        );
    }
}
