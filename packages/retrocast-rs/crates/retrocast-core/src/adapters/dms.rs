use serde_json::Value;

use super::{Adapter, RawRouteEntry, common};
use crate::{
    error::Result,
    model::{Route, Target},
    route::AdaptMode,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct DirectMultiStepAdapter;

pub fn route_length(node: &Value) -> Result<usize> {
    let object = node
        .as_object()
        .ok_or_else(|| common::schema("dms", "route node must be an object"))?;
    let children = object
        .get("children")
        .map(|value| {
            value
                .as_array()
                .ok_or_else(|| common::schema("dms", "route node children must be a list"))
        })
        .transpose()?
        .map(Vec::as_slice)
        .unwrap_or_default();
    if children.is_empty() {
        return Ok(0);
    }
    Ok(1 + children
        .iter()
        .map(route_length)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0))
}

impl Adapter for DirectMultiStepAdapter {
    fn name(&self) -> &'static str {
        "dms"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let routes = payload
            .as_array()
            .ok_or_else(|| common::schema(self.name(), "route payload must be a list"))?;
        Ok(routes
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, payload)| RawRouteEntry {
                payload,
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
        let target_molecule = common::build_plain_tree(&raw_route, self.name(), mode)?
            .ok_or_else(|| common::target_pruned(self.name()))?;
        if let Some(target) = target {
            common::require_target_match(
                &target_molecule.smiles,
                &target.smiles,
                &target.id,
                self.name(),
            )?;
        }
        Ok(Route {
            target: target_molecule,
            annotations: Default::default(),
            schema_version: Default::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::DirectMultiStepAdapter;
    use crate::{adapters::Adapter, model::Target, route::AdaptMode, route_view::InchiKeyLevel};

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
    fn casts_plain_molecule_tree_and_prunes_invalid_leaf() {
        let adapter = DirectMultiStepAdapter;
        let route = adapter
            .cast(
                json!({
                    "smiles": "CCO",
                    "children": [
                        {"smiles": "C"},
                        {"smiles": "not-smiles"}
                    ]
                }),
                AdaptMode::Prune,
                Some(&target()),
            )
            .unwrap();
        assert_eq!(route.leaves().len(), 1);
        assert_eq!(route.signature(InchiKeyLevel::Full, None).len(), 64);
    }

    #[test]
    fn rejects_cycle_after_canonicalization() {
        let error = DirectMultiStepAdapter
            .cast(
                json!({"smiles": "CCO", "children": [{"smiles": "OCC"}]}),
                AdaptMode::Strict,
                Some(&target()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("adapter.cycle_detected"));
    }
}
