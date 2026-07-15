use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::{
    VERSION,
    error::{EngineError, Result},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FileInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub path: String,
    #[serde(alias = "file_hash", rename = "sha256")]
    pub file_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ManifestOutputs {
    List(Vec<FileInfo>),
    ByLabel(BTreeMap<String, FileInfo>),
}

impl Default for ManifestOutputs {
    fn default() -> Self {
        Self::List(Vec::new())
    }
}

impl ManifestOutputs {
    pub fn iter(&self) -> Box<dyn Iterator<Item = &FileInfo> + '_> {
        match self {
            Self::List(files) => Box::new(files.iter()),
            Self::ByLabel(files) => Box::new(files.values()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    #[serde(default = "schema_version")]
    pub schema_version: String,
    #[serde(default = "retrocast_version")]
    pub retrocast_version: String,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    pub action: String,
    #[serde(default)]
    pub parameters: Map<String, Value>,
    #[serde(default)]
    pub directives: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_name: Option<String>,
    #[serde(default)]
    pub source_files: Vec<FileInfo>,
    #[serde(default)]
    pub output_files: ManifestOutputs,
    #[serde(default)]
    pub statistics: Map<String, Value>,
    #[serde(default)]
    pub summary: Map<String, Value>,
    #[serde(flatten)]
    pub extensions: Map<String, Value>,
}

impl Manifest {
    pub fn new(action: impl Into<String>) -> Self {
        Self {
            schema_version: schema_version(),
            retrocast_version: retrocast_version(),
            created_at: Utc::now(),
            action: action.into(),
            parameters: Map::new(),
            directives: Map::new(),
            release_name: None,
            source_files: Vec::new(),
            output_files: ManifestOutputs::default(),
            statistics: Map::new(),
            summary: Map::new(),
            extensions: Map::new(),
        }
    }

    pub fn output_files(&self) -> impl Iterator<Item = &FileInfo> {
        self.output_files.iter()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    Benchmark,
    Predictions,
    RouteCorpus,
    Stock,
    Unknown,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ManifestOutput {
    #[serde(default)]
    pub label: Option<String>,
    pub path: PathBuf,
    pub value: Value,
    pub content_type: ContentType,
    #[serde(default)]
    pub content_hash: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub fn create_manifest(
    action: impl Into<String>,
    sources: &[PathBuf],
    outputs: &[ManifestOutput],
    root_dir: &Path,
    parameters: Map<String, Value>,
    statistics: Map<String, Value>,
    directives: Map<String, Value>,
    summary: Map<String, Value>,
    release_name: Option<String>,
    keyed_output_files: bool,
) -> Result<Manifest> {
    let source_files = sources
        .iter()
        .map(|path| {
            if !path.exists() {
                return Err(EngineError::Provenance(format!(
                    "manifest source file not found: {}",
                    path.display()
                )));
            }
            Ok(FileInfo {
                label: None,
                path: tracked_file_path(path, root_dir),
                file_hash: file_hash(path)?,
                content_hash: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let output_files = outputs
        .iter()
        .map(|output| {
            let file_hash = if output.path.exists() {
                file_hash(&output.path).unwrap_or_else(|_| "error-hashing-file".to_owned())
            } else {
                "file-not-written".to_owned()
            };
            Ok(FileInfo {
                label: output.label.clone(),
                path: tracked_file_path(&output.path, root_dir),
                file_hash,
                content_hash: output
                    .content_hash
                    .clone()
                    .or_else(|| manifest_content_hash(&output.value, output.content_type)),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let output_files = if keyed_output_files {
        let mut by_label = BTreeMap::new();
        for output in output_files {
            let label = output.label.clone().ok_or_else(|| {
                EngineError::Provenance(
                    "keyed_output_files=True requires every output to include a label".to_owned(),
                )
            })?;
            by_label.insert(label, output);
        }
        ManifestOutputs::ByLabel(by_label)
    } else {
        ManifestOutputs::List(output_files)
    };

    Ok(Manifest {
        schema_version: schema_version(),
        retrocast_version: retrocast_version(),
        created_at: Utc::now(),
        action: action.into(),
        parameters,
        directives,
        release_name,
        source_files,
        output_files,
        statistics,
        summary,
        extensions: Map::new(),
    })
}

fn tracked_file_path(path: &Path, root_dir: &Path) -> String {
    let root = resolved_path(root_dir);
    let path = resolved_path(path);
    path.strip_prefix(&root)
        .unwrap_or(&path)
        .to_string_lossy()
        .into_owned()
}

fn resolved_path(path: &Path) -> PathBuf {
    if let Ok(resolved) = path.canonicalize() {
        return resolved;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name())
        && let Ok(parent) = parent.canonicalize()
    {
        return parent.join(name);
    }
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    }
}

fn manifest_content_hash(value: &Value, content_type: ContentType) -> Option<String> {
    if content_type == ContentType::Unknown {
        return None;
    }
    let payload = if content_type == ContentType::Stock {
        match value {
            Value::Object(values) => {
                let mut keys: Vec<_> = values.keys().cloned().collect();
                keys.sort_unstable();
                Value::Array(keys.into_iter().map(Value::String).collect())
            }
            _ => value.clone(),
        }
    } else {
        value.clone()
    };
    let canonical = canonical_json(&payload);
    Some(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

pub(crate) fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => python_json_string(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => {
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_unstable_by_key(|(key, _)| *key);
            format!(
                "{{{}}}",
                entries
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}:{}",
                        python_json_string(key),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

pub(crate) fn python_json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{001f}' || character >= '\u{007f}' => {
                let code = character as u32;
                if code <= 0xffff {
                    output.push_str(&format!("\\u{code:04x}"));
                } else {
                    let code = code - 0x1_0000;
                    let high = 0xd800 + (code >> 10);
                    let low = 0xdc00 + (code & 0x3ff);
                    output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum VerificationLevel {
    Pass,
    Fail,
    Warn,
    Info,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VerificationCategory {
    Graph,
    Phase1,
    Phase2,
    Header,
    Context,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VerificationIssue {
    pub level: VerificationLevel,
    pub path: PathBuf,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<VerificationCategory>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VerificationReport {
    pub manifest_path: PathBuf,
    pub is_valid: bool,
    #[serde(default)]
    pub issues: Vec<VerificationIssue>,
}

impl VerificationReport {
    pub fn new(manifest_path: impl Into<PathBuf>) -> Self {
        Self {
            manifest_path: manifest_path.into(),
            is_valid: true,
            issues: Vec::new(),
        }
    }

    pub fn add(
        &mut self,
        level: VerificationLevel,
        path: impl Into<PathBuf>,
        message: impl Into<String>,
        category: Option<VerificationCategory>,
    ) {
        if level == VerificationLevel::Fail {
            self.is_valid = false;
        }
        self.issues.push(VerificationIssue {
            level,
            path: path.into(),
            message: message.into(),
            category,
        });
    }
}

const PRIMARY_ARTIFACT_DIRS: [&str; 4] = ["0-assets", "1-benchmarks", "2-raw", "tmp"];

pub fn verify_manifest(
    manifest_path: &Path,
    root_dir: &Path,
    deep: bool,
    output_only: bool,
    lenient: bool,
) -> VerificationReport {
    let report_path = tracked_path_key(manifest_path, root_dir);
    let mut report = VerificationReport::new(report_path.clone());
    if !deep {
        match read_manifest(&resolve_tracked_path(manifest_path, root_dir)) {
            Ok(manifest) => verify_physical_integrity(
                &BTreeMap::from([(report_path, manifest)]),
                root_dir,
                &mut report,
                output_only,
                lenient,
            ),
            Err(error) => report.add(
                VerificationLevel::Fail,
                report_path,
                format!("Failed to load manifest: {error}"),
                Some(VerificationCategory::Phase2),
            ),
        }
        return report;
    }

    let graph = build_provenance_graph(manifest_path, root_dir, &mut report);
    if !report.is_valid {
        report.add(
            VerificationLevel::Fail,
            report_path,
            "Could not build provenance graph, aborting.",
            Some(VerificationCategory::Graph),
        );
        return report;
    }
    report.add(
        VerificationLevel::Pass,
        report_path.clone(),
        format!(
            "Successfully built provenance graph with {} manifests.",
            graph.len()
        ),
        Some(VerificationCategory::Graph),
    );
    verify_logical_chain(&graph, root_dir, &mut report);
    if !report.is_valid {
        report.add(
            VerificationLevel::Fail,
            report_path,
            "Logical chain verification failed, aborting physical check.",
            Some(VerificationCategory::Phase1),
        );
        return report;
    }
    verify_physical_integrity(&graph, root_dir, &mut report, output_only, lenient);
    report
}

fn build_provenance_graph(
    start_path: &Path,
    root_dir: &Path,
    report: &mut VerificationReport,
) -> BTreeMap<PathBuf, Manifest> {
    let mut graph = BTreeMap::new();
    let mut queue = VecDeque::from([start_path.to_path_buf()]);
    let mut visited = BTreeSet::new();
    report.add(
        VerificationLevel::Info,
        start_path,
        "Graph Discovery",
        Some(VerificationCategory::Header),
    );

    while let Some(manifest_path) = queue.pop_front() {
        let resolved = resolve_tracked_path(&manifest_path, root_dir);
        if !visited.insert(resolved.clone()) {
            continue;
        }
        let key = tracked_path_key(&manifest_path, root_dir);
        if !resolved.exists() {
            report.add(
                VerificationLevel::Fail,
                key,
                "Manifest file in dependency chain is MISSING.",
                Some(VerificationCategory::Graph),
            );
            continue;
        }
        match read_manifest(&resolved) {
            Ok(manifest) => {
                report.add(
                    VerificationLevel::Pass,
                    key.clone(),
                    format!("Loaded manifest for action '{}'.", manifest.action),
                    Some(VerificationCategory::Graph),
                );
                for source in &manifest.source_files {
                    let source_path = Path::new(&source.path);
                    if !is_primary(source_path) {
                        let parent = resolve_tracked_path(
                            &source_path
                                .parent()
                                .unwrap_or_else(|| Path::new(""))
                                .join("manifest.json"),
                            root_dir,
                        );
                        if !visited.contains(&parent) {
                            report.add(
                                VerificationLevel::Info,
                                source_path,
                                "Source is a generated artifact, adding its manifest to the queue.",
                                Some(VerificationCategory::Graph),
                            );
                            queue.push_back(parent);
                        }
                    }
                }
                graph.insert(key, manifest);
            }
            Err(error) => report.add(
                VerificationLevel::Fail,
                key,
                format!("Failed to load or parse manifest: {error}"),
                Some(VerificationCategory::Graph),
            ),
        }
    }
    graph
}

fn verify_logical_chain(
    graph: &BTreeMap<PathBuf, Manifest>,
    root_dir: &Path,
    report: &mut VerificationReport,
) {
    report.add(
        VerificationLevel::Info,
        report.manifest_path.clone(),
        "Phase 1 - Verifying manifest chain consistency",
        Some(VerificationCategory::Header),
    );
    for (child_path, child) in graph {
        report.add(
            VerificationLevel::Info,
            child_path,
            format!("Inspecting links for manifest '{}'...", child.action),
            Some(VerificationCategory::Context),
        );
        for source in &child.source_files {
            let source_path = Path::new(&source.path);
            if is_primary(source_path) {
                report.add(
                    VerificationLevel::Pass,
                    source_path,
                    "Source is a primary artifact.",
                    Some(VerificationCategory::Phase1),
                );
                continue;
            }
            let parent_path = tracked_path_key(
                &source_path
                    .parent()
                    .unwrap_or_else(|| Path::new(""))
                    .join("manifest.json"),
                root_dir,
            );
            let Some(parent) = graph.get(&parent_path) else {
                report.add(
                    VerificationLevel::Warn,
                    source_path,
                    format!(
                        "Parent manifest '{}' not found; cannot verify link.",
                        parent_path.display()
                    ),
                    Some(VerificationCategory::Phase1),
                );
                continue;
            };
            match parent
                .output_files()
                .find(|output| output.path == source.path)
            {
                None => report.add(
                    VerificationLevel::Fail,
                    source_path,
                    format!(
                        "Provenance broken. Not declared as output in parent manifest ('{}').",
                        parent.action
                    ),
                    Some(VerificationCategory::Phase1),
                ),
                Some(output) if output.file_hash != source.file_hash => report.add(
                    VerificationLevel::Fail,
                    source_path,
                    "Provenance broken. Hash mismatch between parent and child manifests.",
                    Some(VerificationCategory::Phase1),
                ),
                Some(_) => report.add(
                    VerificationLevel::Pass,
                    source_path,
                    format!(
                        "Link to parent manifest ('{}') is consistent.",
                        parent.action
                    ),
                    Some(VerificationCategory::Phase1),
                ),
            }
        }
    }
}

fn verify_physical_integrity(
    graph: &BTreeMap<PathBuf, Manifest>,
    root_dir: &Path,
    report: &mut VerificationReport,
    output_only: bool,
    lenient: bool,
) {
    report.add(
        VerificationLevel::Info,
        report.manifest_path.clone(),
        "Phase 2 - Verifying on-disk file integrity",
        Some(VerificationCategory::Header),
    );
    let mut expected = BTreeMap::new();
    for manifest in graph.values() {
        for output in manifest.output_files() {
            expected.insert(PathBuf::from(&output.path), output.file_hash.clone());
        }
    }
    if !output_only {
        for manifest in graph.values() {
            for source in &manifest.source_files {
                expected
                    .entry(PathBuf::from(&source.path))
                    .or_insert_with(|| source.file_hash.clone());
            }
        }
    }
    for (path, expected_hash) in expected {
        let absolute = resolve_tracked_path(&path, root_dir);
        if !absolute.exists() {
            report.add(
                if lenient {
                    VerificationLevel::Warn
                } else {
                    VerificationLevel::Fail
                },
                path,
                "File is MISSING from disk.",
                Some(VerificationCategory::Phase2),
            );
            continue;
        }
        match file_hash(&absolute) {
            Ok(actual) if actual == expected_hash => report.add(
                VerificationLevel::Pass,
                path,
                "On-disk file hash matches manifest record.",
                Some(VerificationCategory::Phase2),
            ),
            Ok(_) => report.add(
                VerificationLevel::Fail,
                path,
                "HASH MISMATCH (Disk vs. Manifest).",
                Some(VerificationCategory::Phase2),
            ),
            Err(error) => report.add(
                VerificationLevel::Fail,
                path,
                format!("Could not hash file: {error}"),
                Some(VerificationCategory::Phase2),
            ),
        }
    }
}

fn read_manifest(path: &Path) -> Result<Manifest> {
    let payload = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
}

fn is_primary(path: &Path) -> bool {
    path.components().any(|component| {
        PRIMARY_ARTIFACT_DIRS.contains(&component.as_os_str().to_string_lossy().as_ref())
    })
}

fn resolve_tracked_path(path: &Path, root_dir: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let mut candidates = vec![root_dir.join(path)];
    candidates.extend(root_dir.ancestors().skip(1).map(|parent| parent.join(path)));
    if let Ok(current) = std::env::current_dir() {
        candidates.push(current.join(path));
    }
    candidates
        .iter()
        .find(|candidate| candidate.exists())
        .cloned()
        .unwrap_or_else(|| candidates[0].clone())
}

fn tracked_path_key(path: &Path, root_dir: &Path) -> PathBuf {
    let resolved = resolve_tracked_path(path, root_dir);
    resolved
        .strip_prefix(root_dir)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| {
            if path.is_absolute() {
                resolved
            } else {
                path.to_path_buf()
            }
        })
}

pub fn file_hash(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn content_hash(value: &Value) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical_json(value).as_bytes())
    ))
}

fn schema_version() -> String {
    "2".to_owned()
}

fn retrocast_version() -> String {
    VERSION.to_owned()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Manifest, ManifestOutputs, content_hash};

    #[test]
    fn content_hash_is_object_order_independent() {
        assert_eq!(
            content_hash(&json!({"a": 1, "b": 2})).unwrap(),
            content_hash(&json!({"b": 2, "a": 1})).unwrap()
        );
    }

    #[test]
    fn manifest_preserves_unknown_fields() {
        let manifest: Manifest = serde_json::from_value(json!({
            "action": "test",
            "future_field": {"kept": true}
        }))
        .unwrap();
        assert_eq!(manifest.extensions["future_field"]["kept"], true);
        assert!(matches!(manifest.output_files, ManifestOutputs::List(_)));
    }
}
