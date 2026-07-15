use std::{collections::BTreeMap, sync::LazyLock};

use regex::Regex;
use serde_json::Value;

use super::{Adapter, RawRouteEntry, common};
use crate::{
    chem,
    error::{EngineError, Result},
    model::{Route, Target},
    route::AdaptMode,
};

static THINK_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>.*?</think>").expect("valid think regex"));
static SYNTHESIS_STEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<synthesis_step>(.*?)</synthesis_step>").expect("valid step regex")
});
static PRODUCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<product>(.*?)</product>").expect("valid product regex"));
static REACTANT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<reactant>(.*?)</reactant>").expect("valid reactant regex"));
static SMILES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<smiles>(.*?)</smiles>").expect("valid smiles regex"));
static SM_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<sm_([^>]+)>").expect("valid token regex"));

#[derive(Clone, Copy, Debug, Default)]
pub struct UrsaAdapter;

impl Adapter for UrsaAdapter {
    fn name(&self) -> &'static str {
        "ursa"
    }

    fn entries(&self, payload: Value, source_key: Option<&str>) -> Result<Vec<RawRouteEntry>> {
        let records = payload
            .as_array()
            .ok_or_else(|| common::schema(self.name(), "expected a list of completion records"))?;
        records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                let object = record.as_object().ok_or_else(|| {
                    common::schema(
                        self.name(),
                        format!("completion record {index} must be an object"),
                    )
                })?;
                let completion = object
                    .get("completion")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        common::schema(
                            self.name(),
                            format!("completion record {index} needs string completion"),
                        )
                    })?;
                let target_hint_smiles =
                    extract_target_hint(object.get("meta"), self.name(), index)?;
                Ok(RawRouteEntry {
                    payload: Value::String(completion.to_owned()),
                    source_key: source_key.map(str::to_owned),
                    source_row_index: Some(index + 1),
                    source_record_id: None,
                    target_hint_id: None,
                    target_hint_smiles,
                    source_order: Some(index + 1),
                })
            })
            .collect()
    }

    fn cast(&self, raw_route: Value, mode: AdaptMode, target: Option<&Target>) -> Result<Route> {
        let completion = raw_route
            .as_str()
            .ok_or_else(|| common::schema(self.name(), "expected completion text"))?;
        let target = target.ok_or_else(|| {
            common::logic(
                self.name(),
                "adapter.route_transform_failed",
                "URSA adaptation requires a target",
            )
        })?;
        let (expected, _) = chem::normalize(&target.smiles)?;
        let precursor_map = parse_completion(completion, mode)?;
        if !precursor_map.contains_key(&expected) {
            return Err(EngineError::TargetMismatch {
                adapter: self.name(),
                target_id: target.id.clone(),
                expected: expected.to_string(),
                actual: format!("missing:{expected}"),
            });
        }
        let route_target =
            common::build_from_precursor_map(&expected, &precursor_map, self.name(), mode, None)?
                .ok_or_else(|| common::target_pruned(self.name()))?;
        Ok(Route {
            target: route_target,
            annotations: Default::default(),
            schema_version: Default::default(),
        })
    }
}

fn parse_completion(completion: &str, mode: AdaptMode) -> Result<common::PrecursorMap> {
    let cleaned = THINK_BLOCK.replace_all(completion, "");
    let mut precursor_map = BTreeMap::new();
    for step in SYNTHESIS_STEP.captures_iter(&cleaned) {
        let step = &step[1];
        let Some(product_block) = PRODUCT.captures(step) else {
            continue;
        };
        let Some(raw_product) = extract_smiles(&product_block[1]) else {
            continue;
        };
        let product = match chem::normalize(&raw_product) {
            Ok((smiles, _)) => smiles,
            Err(_) if mode == AdaptMode::Prune => continue,
            Err(error) => return Err(error),
        };
        let reactants: Vec<_> = REACTANT
            .captures_iter(step)
            .filter_map(|capture| extract_smiles(&capture[1]))
            .collect();
        if !reactants.is_empty() {
            precursor_map.insert(product, reactants);
        }
    }
    Ok(precursor_map)
}

fn extract_smiles(block: &str) -> Option<String> {
    let inner = SMILES.captures(block)?.get(1)?.as_str().trim();
    if inner.contains("<sm_") {
        let tokens: String = SM_TOKEN
            .captures_iter(inner)
            .filter_map(|capture| capture.get(1).map(|token| token.as_str()))
            .collect();
        (!tokens.is_empty()).then_some(tokens)
    } else {
        (!inner.is_empty()).then(|| inner.to_owned())
    }
}

fn extract_target_hint(
    metadata: Option<&Value>,
    adapter: &'static str,
    record_index: usize,
) -> Result<Option<String>> {
    let Some(metadata) = metadata.and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(product) = metadata.get("product_smiles") else {
        return Ok(None);
    };
    let product = product.as_str().ok_or_else(|| {
        common::schema(
            adapter,
            format!("completion record {record_index} has non-string meta.product_smiles"),
        )
    })?;
    chem::normalize(product)
        .map(|(smiles, _)| Some(smiles.to_string()))
        .map_err(|_| {
            common::schema(
                adapter,
                format!("completion record {record_index} has invalid meta.product_smiles"),
            )
        })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::UrsaAdapter;
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
    fn extracts_target_hint_and_tokenized_smiles() {
        let completion = "<synthesis_step><product><smiles><sm_C><sm_C><sm_O></smiles></product><reactant><smiles><sm_C></smiles></reactant></synthesis_step>";
        let entry = UrsaAdapter
            .entries(
                json!([{"completion": completion, "meta": {"product_smiles": "OCC"}}]),
                None,
            )
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(entry.target_hint_smiles.as_deref(), Some("CCO"));
        let route = UrsaAdapter
            .cast(entry.payload, AdaptMode::Strict, Some(&target()))
            .unwrap();
        assert_eq!(route.leaves()[0].value.smiles, "C");
    }

    #[test]
    fn requires_target_step() {
        let completion = "<synthesis_step><product><smiles>CCC</smiles></product><reactant><smiles>C</smiles></reactant></synthesis_step>";
        assert!(
            UrsaAdapter
                .cast(json!(completion), AdaptMode::Strict, Some(&target()))
                .is_err()
        );
    }
}
