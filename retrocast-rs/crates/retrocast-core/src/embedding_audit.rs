//! Release-level route embedding audits built on the core tree matcher.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    embedding::{EmbeddingOptions, route_embeds_at, subtree_reaction_count},
    model::Route,
    route_path::RoutePath,
    route_view::InchiKeyLevel,
    training::TrainingRouteRecord,
};

#[derive(Clone, Debug, Deserialize)]
pub struct QuerySource {
    pub source: String,
    pub queries: Vec<QueryRoute>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QueryRoute {
    pub id: String,
    pub route: Route,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct RouteEmbeddingAuditOptions {
    #[serde(default)]
    pub match_level: InchiKeyLevel,
    #[serde(default = "enabled")]
    pub allow_leaf_extension: bool,
    #[serde(default)]
    pub include_partial: bool,
    #[serde(default = "default_partial_min_reactions")]
    pub partial_min_reactions: usize,
    #[serde(default)]
    pub exclude_query_containers: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouteEmbeddingLedgerRow {
    pub query_source: String,
    pub query_id: String,
    pub query_path: String,
    pub query_route_reactions: usize,
    pub query_subroute_reactions: usize,
    pub container_route_id: String,
    pub container_split: String,
    pub container_path: String,
    pub container_route_reactions: usize,
    pub container_subtree_reactions: usize,
    pub match_kind: String,
    pub matched_reactions: usize,
    pub leaf_extension_query_paths: Vec<String>,
    pub leaf_extension_container_paths: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FullEmbeddingSummary {
    pub query_routes_with_embedding: usize,
    pub embedding_occurrences: usize,
    pub query_routes_with_root_shifted_embedding: usize,
    pub query_routes_with_leaf_extended_embedding: usize,
    pub root_distance_counts: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct InternalSubrouteEmbeddingSummary {
    pub min_reactions: usize,
    pub checked_internal_subroutes: usize,
    pub embedded_internal_subroutes: usize,
    pub query_routes_with_embedding: usize,
    pub embedding_occurrences: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct PrefixDepthSummary {
    pub depth: usize,
    pub query_routes: usize,
    pub root_prefix_signature_overlap: usize,
    pub subtree_prefix_signature_overlap: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CoverageSummary {
    pub basis: String,
    pub all_query_routes: usize,
    pub embedded_query_routes: usize,
    pub all_query_fraction_stats: (f64, f64, f64),
    pub embedded_query_fraction_stats: (f64, f64, f64),
    pub all_container_fraction_stats: (f64, f64, f64),
    pub embedded_container_fraction_stats: (f64, f64, f64),
    pub embedded_mean_container_routes_per_query: f64,
    pub embedded_mean_occurrences_per_query: f64,
    pub matched_fraction_histogram: Vec<(String, usize)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuerySetAudit {
    pub source: String,
    pub query_routes: usize,
    pub query_reaction_signatures: usize,
    pub exact_route_signature_overlap: usize,
    pub reaction_signature_overlap: usize,
    pub prefix_depths: Vec<PrefixDepthSummary>,
    pub full_embeddings: FullEmbeddingSummary,
    pub internal_subroute_embeddings: Option<InternalSubrouteEmbeddingSummary>,
    pub coverage: CoverageSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouteEmbeddingAudit {
    pub release_name: String,
    pub match_level: String,
    pub allow_leaf_extension: bool,
    pub partial_min_reactions: Option<usize>,
    pub container_routes: usize,
    pub container_route_signatures: usize,
    pub container_reaction_signatures: usize,
    pub query_sets: Vec<QuerySetAudit>,
    pub ledger_rows: Vec<RouteEmbeddingLedgerRow>,
}

#[derive(Clone, Debug, Error)]
#[error("{0}")]
pub struct EmbeddingAuditError(pub String);

#[derive(Clone)]
struct Occurrence {
    record_index: usize,
    path: RoutePath,
}

#[derive(Default)]
struct ContainerIndex {
    route_signatures: HashSet<String>,
    reaction_signatures: HashSet<String>,
    route_signature_counts: HashMap<String, usize>,
    reaction_signature_route_counts: HashMap<String, usize>,
    root_prefixes: BTreeMap<usize, HashSet<String>>,
    subtree_prefixes: BTreeMap<usize, HashSet<String>>,
    root_prefix_counts: BTreeMap<usize, HashMap<String, usize>>,
    subtree_prefix_counts: BTreeMap<usize, HashMap<String, usize>>,
    by_reaction_root: HashMap<String, Vec<Occurrence>>,
}

pub fn build_route_embedding_audit(
    release_name: &str,
    containers: &[TrainingRouteRecord],
    query_sources: &[QuerySource],
    options: RouteEmbeddingAuditOptions,
) -> Result<RouteEmbeddingAudit, EmbeddingAuditError> {
    validate_exclusions(containers, query_sources, options.exclude_query_containers)?;
    let index = build_index(containers, options.match_level);
    let mut query_sets = Vec::new();
    let mut ledger_rows = Vec::new();
    for source in query_sources {
        let (audit, mut rows) = audit_query_source(source, containers, &index, options);
        query_sets.push(audit);
        ledger_rows.append(&mut rows);
    }
    Ok(RouteEmbeddingAudit {
        release_name: release_name.to_owned(),
        match_level: match_level_name(options.match_level).to_owned(),
        allow_leaf_extension: options.allow_leaf_extension,
        partial_min_reactions: options
            .include_partial
            .then_some(options.partial_min_reactions),
        container_routes: containers.len(),
        container_route_signatures: index.route_signatures.len(),
        container_reaction_signatures: index.reaction_signatures.len(),
        query_sets,
        ledger_rows,
    })
}

fn validate_exclusions(
    containers: &[TrainingRouteRecord],
    sources: &[QuerySource],
    enabled: bool,
) -> Result<(), EmbeddingAuditError> {
    if !enabled {
        return Ok(());
    }
    let mut counts = BTreeMap::new();
    for record in containers {
        *counts.entry(record.id.as_str()).or_insert(0_usize) += 1;
    }
    let duplicates: Vec<_> = counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(id, _)| (*id).to_owned())
        .collect();
    if !duplicates.is_empty() {
        return Err(EmbeddingAuditError(format!(
            "exclude_query_containers requires unique container record ids; duplicate ids: {duplicates:?}"
        )));
    }
    let ids: HashSet<_> = containers.iter().map(|record| record.id.as_str()).collect();
    let missing: Vec<_> = sources
        .iter()
        .flat_map(|source| &source.queries)
        .map(|query| query.id.as_str())
        .filter(|id| !ids.contains(id))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if !missing.is_empty() {
        return Err(EmbeddingAuditError(format!(
            "exclude_query_containers requires query ids to match container record ids; missing ids: {missing:?}"
        )));
    }
    Ok(())
}

fn build_index(records: &[TrainingRouteRecord], level: InchiKeyLevel) -> ContainerIndex {
    let mut index = ContainerIndex::default();
    for (record_index, record) in records.iter().enumerate() {
        let signature = record.route.signature(level, None);
        index.route_signatures.insert(signature.clone());
        *index.route_signature_counts.entry(signature).or_default() += 1;
        let reaction_signatures = record.route.reaction_signatures(level);
        index
            .reaction_signatures
            .extend(reaction_signatures.iter().cloned());
        for signature in reaction_signatures {
            *index
                .reaction_signature_route_counts
                .entry(signature)
                .or_default() += 1;
        }
        for depth in 1..=record.route.depth() {
            let signature = record.route.signature(level, Some(depth));
            index
                .root_prefixes
                .entry(depth)
                .or_default()
                .insert(signature.clone());
            *index
                .root_prefix_counts
                .entry(depth)
                .or_default()
                .entry(signature)
                .or_default() += 1;
        }
        let mut record_subtree_prefixes: BTreeMap<usize, HashSet<String>> = BTreeMap::new();
        for molecule in record.route.molecules() {
            for depth in 1..=molecule.depth() {
                let signature = molecule.subtree_signature(level, Some(depth));
                index
                    .subtree_prefixes
                    .entry(depth)
                    .or_default()
                    .insert(signature.clone());
                record_subtree_prefixes
                    .entry(depth)
                    .or_default()
                    .insert(signature);
            }
            if let Some(reaction) = molecule.produced_by() {
                index
                    .by_reaction_root
                    .entry(reaction.signature(level))
                    .or_default()
                    .push(Occurrence {
                        record_index,
                        path: molecule.path,
                    });
            }
        }
        for (depth, signatures) in record_subtree_prefixes {
            let counts = index.subtree_prefix_counts.entry(depth).or_default();
            for signature in signatures {
                *counts.entry(signature).or_default() += 1;
            }
        }
    }
    index
}

fn audit_query_source(
    source: &QuerySource,
    containers: &[TrainingRouteRecord],
    index: &ContainerIndex,
    options: RouteEmbeddingAuditOptions,
) -> (QuerySetAudit, Vec<RouteEmbeddingLedgerRow>) {
    let mut full_rows = Vec::new();
    for query in &source.queries {
        full_rows.extend(embedding_rows(
            &source.source,
            query,
            &RoutePath::target(),
            "full_route",
            containers,
            index,
            options,
        ));
    }
    let (internal_summary, internal_rows) = if options.include_partial {
        let (summary, rows) = internal_subroute_audit(source, containers, index, options);
        (Some(summary), rows)
    } else {
        (None, Vec::new())
    };
    let mut rows = full_rows.clone();
    rows.extend(internal_rows);
    let reaction_signatures: HashSet<_> = source
        .queries
        .iter()
        .flat_map(|query| query.route.reaction_signatures(options.match_level))
        .collect();
    let exact_overlap = source
        .queries
        .iter()
        .filter(|query| {
            let count = index
                .route_signature_counts
                .get(&query.route.signature(options.match_level, None))
                .copied()
                .unwrap_or(0);
            if options.exclude_query_containers {
                count > 1
            } else {
                count > 0
            }
        })
        .count();
    let reaction_overlap = reaction_signatures
        .iter()
        .filter(|signature| {
            let count = index
                .reaction_signature_route_counts
                .get(*signature)
                .copied()
                .unwrap_or(0);
            if options.exclude_query_containers {
                count > 1
            } else {
                count > 0
            }
        })
        .count();
    let query_ids: Vec<_> = source
        .queries
        .iter()
        .map(|query| query.id.as_str())
        .collect();
    (
        QuerySetAudit {
            source: source.source.clone(),
            query_routes: source.queries.len(),
            query_reaction_signatures: reaction_signatures.len(),
            exact_route_signature_overlap: exact_overlap,
            reaction_signature_overlap: reaction_overlap,
            prefix_depths: summarize_prefixes(source, index, options),
            full_embeddings: summarize_full(&full_rows),
            internal_subroute_embeddings: internal_summary,
            coverage: summarize_coverage(
                &query_ids,
                &rows,
                if options.include_partial {
                    format!(
                        "full routes + internal subroutes with {}+ reactions",
                        options.partial_min_reactions
                    )
                } else {
                    "full routes only".to_owned()
                },
            ),
        },
        rows,
    )
}

fn summarize_prefixes(
    source: &QuerySource,
    index: &ContainerIndex,
    options: RouteEmbeddingAuditOptions,
) -> Vec<PrefixDepthSummary> {
    let max_depth = source
        .queries
        .iter()
        .map(|query| query.route.depth())
        .max()
        .unwrap_or(0);
    (1..=max_depth)
        .map(|depth| {
            let eligible: Vec<_> = source
                .queries
                .iter()
                .filter(|query| query.route.depth() >= depth)
                .collect();
            let signatures: Vec<_> = eligible
                .iter()
                .map(|query| query.route.signature(options.match_level, Some(depth)))
                .collect();
            let overlap = |counts: Option<&HashMap<String, usize>>,
                           values: Option<&HashSet<String>>| {
                signatures
                    .iter()
                    .filter(|signature| {
                        if options.exclude_query_containers {
                            counts
                                .and_then(|counts| counts.get(*signature))
                                .copied()
                                .unwrap_or(0)
                                > 1
                        } else {
                            values.is_some_and(|values| values.contains(*signature))
                        }
                    })
                    .count()
            };
            PrefixDepthSummary {
                depth,
                query_routes: eligible.len(),
                root_prefix_signature_overlap: overlap(
                    index.root_prefix_counts.get(&depth),
                    index.root_prefixes.get(&depth),
                ),
                subtree_prefix_signature_overlap: overlap(
                    index.subtree_prefix_counts.get(&depth),
                    index.subtree_prefixes.get(&depth),
                ),
            }
        })
        .collect()
}

fn internal_subroute_audit(
    source: &QuerySource,
    containers: &[TrainingRouteRecord],
    index: &ContainerIndex,
    options: RouteEmbeddingAuditOptions,
) -> (
    InternalSubrouteEmbeddingSummary,
    Vec<RouteEmbeddingLedgerRow>,
) {
    let mut checked = 0;
    let mut embedded = HashSet::new();
    let mut rows = Vec::new();
    for query in &source.queries {
        for molecule in query.route.molecules() {
            if molecule.path == RoutePath::target()
                || subtree_reaction_count(&molecule) < options.partial_min_reactions
            {
                continue;
            }
            checked += 1;
            let path = molecule.path.clone();
            let matches = embedding_rows(
                &source.source,
                query,
                &path,
                "internal_subroute",
                containers,
                index,
                options,
            );
            if !matches.is_empty() {
                embedded.insert((query.id.clone(), path));
            }
            rows.extend(matches);
        }
    }
    let embedded_query_ids: HashSet<_> = embedded.iter().map(|(id, _)| id).collect();
    (
        InternalSubrouteEmbeddingSummary {
            min_reactions: options.partial_min_reactions,
            checked_internal_subroutes: checked,
            embedded_internal_subroutes: embedded.len(),
            query_routes_with_embedding: embedded_query_ids.len(),
            embedding_occurrences: rows.len(),
        },
        rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn embedding_rows(
    source: &str,
    query: &QueryRoute,
    query_path: &RoutePath,
    match_kind: &str,
    containers: &[TrainingRouteRecord],
    index: &ContainerIndex,
    options: RouteEmbeddingAuditOptions,
) -> Vec<RouteEmbeddingLedgerRow> {
    let query_molecule = query
        .route
        .molecule_at(query_path)
        .expect("query path belongs to query route");
    let Some(root_reaction) = query_molecule.produced_by() else {
        return Vec::new();
    };
    let Some(occurrences) = index
        .by_reaction_root
        .get(&root_reaction.signature(options.match_level))
    else {
        return Vec::new();
    };
    let query_route_reactions = query.route.reactions().len();
    let query_subroute_reactions = subtree_reaction_count(&query_molecule);
    let embedding_options = EmbeddingOptions {
        match_level: options.match_level,
        allow_leaf_extension: options.allow_leaf_extension,
    };
    let mut rows = Vec::new();
    for occurrence in occurrences {
        let record = &containers[occurrence.record_index];
        if options.exclude_query_containers && record.id == query.id {
            continue;
        }
        let container_molecule = record
            .route
            .molecule_at(&occurrence.path)
            .expect("indexed path belongs to container route");
        let container_subtree_reactions = subtree_reaction_count(&container_molecule);
        let Some(matched) = route_embeds_at(
            query_molecule.clone(),
            container_molecule,
            embedding_options,
        ) else {
            continue;
        };
        rows.push(RouteEmbeddingLedgerRow {
            query_source: source.to_owned(),
            query_id: query.id.clone(),
            query_path: matched.query_path.to_string(),
            query_route_reactions,
            query_subroute_reactions,
            container_route_id: record.id.clone(),
            container_split: record.split.clone(),
            container_path: matched.container_path.to_string(),
            container_route_reactions: record.route.reactions().len(),
            container_subtree_reactions,
            match_kind: match_kind.to_owned(),
            matched_reactions: matched.matched_reactions,
            leaf_extension_query_paths: matched
                .leaf_extensions
                .iter()
                .map(|extension| extension.query_leaf_path.to_string())
                .collect(),
            leaf_extension_container_paths: matched
                .leaf_extensions
                .iter()
                .map(|extension| extension.container_path.to_string())
                .collect(),
        });
    }
    rows
}

fn summarize_full(rows: &[RouteEmbeddingLedgerRow]) -> FullEmbeddingSummary {
    let mut distances = BTreeMap::new();
    for row in rows {
        let depth = RoutePath::parse(&row.container_path)
            .expect("ledger paths are canonical")
            .depth();
        *distances.entry(depth).or_insert(0_usize) += 1;
    }
    FullEmbeddingSummary {
        query_routes_with_embedding: rows
            .iter()
            .map(|row| &row.query_id)
            .collect::<HashSet<_>>()
            .len(),
        embedding_occurrences: rows.len(),
        query_routes_with_root_shifted_embedding: rows
            .iter()
            .filter(|row| row.container_path != "rc:m:/")
            .map(|row| &row.query_id)
            .collect::<HashSet<_>>()
            .len(),
        query_routes_with_leaf_extended_embedding: rows
            .iter()
            .filter(|row| !row.leaf_extension_query_paths.is_empty())
            .map(|row| &row.query_id)
            .collect::<HashSet<_>>()
            .len(),
        root_distance_counts: distances.into_iter().collect(),
    }
}

#[derive(Clone, Copy)]
struct QueryCoverage {
    query_fraction: f64,
    container_fraction: f64,
    container_routes: usize,
    occurrences: usize,
}

fn summarize_coverage(
    query_ids: &[&str],
    rows: &[RouteEmbeddingLedgerRow],
    basis: String,
) -> CoverageSummary {
    let mut by_query: HashMap<&str, Vec<&RouteEmbeddingLedgerRow>> = HashMap::new();
    for row in rows {
        by_query.entry(&row.query_id).or_default().push(row);
    }
    let coverage: Vec<_> = query_ids
        .iter()
        .map(|id| query_coverage(by_query.get(id).map(Vec::as_slice).unwrap_or_default()))
        .collect();
    let embedded: Vec<_> = coverage
        .iter()
        .copied()
        .filter(|row| row.occurrences > 0)
        .collect();
    CoverageSummary {
        basis,
        all_query_routes: coverage.len(),
        embedded_query_routes: embedded.len(),
        all_query_fraction_stats: fraction_stats(
            &coverage
                .iter()
                .map(|row| row.query_fraction)
                .collect::<Vec<_>>(),
        ),
        embedded_query_fraction_stats: fraction_stats(
            &embedded
                .iter()
                .map(|row| row.query_fraction)
                .collect::<Vec<_>>(),
        ),
        all_container_fraction_stats: fraction_stats(
            &coverage
                .iter()
                .map(|row| row.container_fraction)
                .collect::<Vec<_>>(),
        ),
        embedded_container_fraction_stats: fraction_stats(
            &embedded
                .iter()
                .map(|row| row.container_fraction)
                .collect::<Vec<_>>(),
        ),
        embedded_mean_container_routes_per_query: mean(
            &embedded
                .iter()
                .map(|row| row.container_routes as f64)
                .collect::<Vec<_>>(),
        ),
        embedded_mean_occurrences_per_query: mean(
            &embedded
                .iter()
                .map(|row| row.occurrences as f64)
                .collect::<Vec<_>>(),
        ),
        matched_fraction_histogram: coverage_histogram(&coverage),
    }
}

fn query_coverage(rows: &[&RouteEmbeddingLedgerRow]) -> QueryCoverage {
    let Some(mut best) = rows.first().copied() else {
        return QueryCoverage {
            query_fraction: 0.0,
            container_fraction: 0.0,
            container_routes: 0,
            occurrences: 0,
        };
    };
    for candidate in &rows[1..] {
        if compare_coverage(candidate, best) == Ordering::Greater {
            best = candidate;
        }
    }
    QueryCoverage {
        query_fraction: ratio(best.matched_reactions, best.query_route_reactions),
        container_fraction: ratio(best.matched_reactions, best.container_route_reactions),
        container_routes: rows
            .iter()
            .map(|row| row.container_route_id.as_str())
            .collect::<HashSet<_>>()
            .len(),
        occurrences: rows.len(),
    }
}

fn compare_coverage(left: &RouteEmbeddingLedgerRow, right: &RouteEmbeddingLedgerRow) -> Ordering {
    let left_rank = [
        ratio(left.matched_reactions, left.query_route_reactions),
        ratio(left.matched_reactions, left.container_subtree_reactions),
        ratio(left.matched_reactions, left.container_route_reactions),
        left.matched_reactions as f64,
    ];
    let right_rank = [
        ratio(right.matched_reactions, right.query_route_reactions),
        ratio(right.matched_reactions, right.container_subtree_reactions),
        ratio(right.matched_reactions, right.container_route_reactions),
        right.matched_reactions as f64,
    ];
    left_rank
        .iter()
        .zip(right_rank)
        .find_map(|(left, right)| {
            let ordering = left.total_cmp(&right);
            (ordering != Ordering::Equal).then_some(ordering)
        })
        .unwrap_or(Ordering::Equal)
}

fn fraction_stats(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let midpoint = sorted.len() / 2;
    let median = if sorted.len() % 2 == 1 {
        sorted[midpoint]
    } else {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    };
    let p90_index = (0.9 * sorted.len() as f64).ceil() as usize - 1;
    (mean(values), median, sorted[p90_index])
}

fn coverage_histogram(rows: &[QueryCoverage]) -> Vec<(String, usize)> {
    let buckets = ["0%", "(0,25%]", "(25,50%]", "(50,75%]", "(75,100%)", "100%"];
    let mut counts = [0_usize; 6];
    for row in rows {
        let index = match row.query_fraction {
            0.0 => 0,
            value if value <= 0.25 => 1,
            value if value <= 0.50 => 2,
            value if value <= 0.75 => 3,
            value if value < 1.0 => 4,
            _ => 5,
        };
        counts[index] += 1;
    }
    buckets
        .into_iter()
        .zip(counts)
        .map(|(bucket, count)| (bucket.to_owned(), count))
        .collect()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn match_level_name(level: InchiKeyLevel) -> &'static str {
    match level {
        InchiKeyLevel::Full => "full",
        InchiKeyLevel::NoStereo => "no_stereo",
        InchiKeyLevel::Connectivity => "connectivity",
    }
}

fn enabled() -> bool {
    true
}

fn default_partial_min_reactions() -> usize {
    2
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{QueryRoute, QuerySource, RouteEmbeddingAuditOptions, build_route_embedding_audit};
    use crate::{model::Route, route_view::InchiKeyLevel, training::TrainingRouteRecord};

    fn route(value: serde_json::Value) -> Route {
        serde_json::from_value(value).unwrap()
    }

    fn chain(keys: &[&str]) -> Route {
        fn inchikey(label: &str) -> String {
            let letter = label.chars().next().expect("test label");
            format!(
                "{}-{}-{letter}",
                letter.to_string().repeat(14),
                letter.to_string().repeat(10)
            )
        }

        fn molecule(keys: &[&str]) -> serde_json::Value {
            if keys.len() == 1 {
                return json!({"smiles": keys[0], "inchikey": inchikey(keys[0])});
            }
            json!({
                "smiles": keys[0],
                "inchikey": inchikey(keys[0]),
                "product_of": {"reactants": [molecule(&keys[1..])]}
            })
        }
        route(json!({"target": molecule(keys)}))
    }

    fn options() -> RouteEmbeddingAuditOptions {
        RouteEmbeddingAuditOptions {
            match_level: InchiKeyLevel::Full,
            allow_leaf_extension: true,
            include_partial: false,
            partial_min_reactions: 2,
            exclude_query_containers: false,
        }
    }

    #[test]
    fn preserves_query_order_and_summarizes_shifted_embeddings() {
        let containers = vec![TrainingRouteRecord {
            id: "container".to_owned(),
            split: "training".to_owned(),
            route: chain(&["A", "B", "C", "D"]),
            sources: Vec::new(),
        }];
        let sources = vec![QuerySource {
            source: "bench".to_owned(),
            queries: vec![
                QueryRoute {
                    id: "shifted".to_owned(),
                    route: chain(&["B", "C"]),
                },
                QueryRoute {
                    id: "leaf".to_owned(),
                    route: chain(&["A", "B"]),
                },
            ],
        }];
        let audit =
            build_route_embedding_audit("release", &containers, &sources, options()).unwrap();
        assert_eq!(
            audit
                .ledger_rows
                .iter()
                .map(|row| row.query_id.as_str())
                .collect::<Vec<_>>(),
            ["shifted", "leaf"]
        );
        assert_eq!(
            audit.query_sets[0]
                .full_embeddings
                .query_routes_with_root_shifted_embedding,
            1
        );
    }

    #[test]
    fn exclusion_requires_query_ids_to_name_container_records() {
        let containers = vec![TrainingRouteRecord {
            id: "container".to_owned(),
            split: "training".to_owned(),
            route: chain(&["A", "B"]),
            sources: Vec::new(),
        }];
        let sources = vec![QuerySource {
            source: "bench".to_owned(),
            queries: vec![QueryRoute {
                id: "missing".to_owned(),
                route: chain(&["A", "B"]),
            }],
        }];
        let mut options = options();
        options.exclude_query_containers = true;
        let error =
            build_route_embedding_audit("release", &containers, &sources, options).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("query ids to match container record ids")
        );
    }
}
