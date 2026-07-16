//! Hosted dataset resolution, transport, and checksum parsing.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use reqwest::{StatusCode, blocking::Client};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use url::Url;

const USER_AGENT: &str = "retrocast/2";
const PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'!')
    .add(b'"')
    .add(b'#')
    .add(b'$')
    .add(b'%')
    .add(b'&')
    .add(b'\'')
    .add(b'(')
    .add(b')')
    .add(b'*')
    .add(b'+')
    .add(b',')
    .add(b'/')
    .add(b':')
    .add(b';')
    .add(b'<')
    .add(b'=')
    .add(b'>')
    .add(b'?')
    .add(b'@')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

#[derive(Debug, Error, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetError {
    #[error("invalid dataset URL {url:?}: {message}")]
    InvalidUrl { url: String, message: String },
    #[error("dataset URL is not a local file: {url}")]
    InvalidFileUrl { url: String },
    #[error("hosted endpoint returned HTTP {status} for {url}")]
    HttpStatus { url: String, status: u16 },
    #[error("failed to reach hosted endpoint {url}: {message}")]
    Unreachable { url: String, message: String },
    #[error("failed to read hosted file {path}: {message}")]
    Read { path: PathBuf, message: String },
    #[error("failed to write hosted file {path}: {message}")]
    Write { path: PathBuf, message: String },
    #[error("invalid JSON returned by hosted endpoint {url}: {message}")]
    InvalidJson { url: String, message: String },
    #[error("invalid dataset checksum line: {line:?}")]
    InvalidChecksum { line: String },
    #[error("{message}")]
    Policy {
        category: DatasetErrorCategory,
        code: String,
        message: String,
        context: Value,
        retryable: bool,
    },
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetErrorCategory {
    Configuration,
    Resolution,
    Download,
    Verification,
    ArtifactFormat,
}

pub type DatasetResult<T> = std::result::Result<T, DatasetError>;

pub fn build_url(base_url: &str, segments: &[String]) -> String {
    let mut result = base_url.trim_end_matches('/').to_owned();
    for segment in segments {
        result.push('/');
        result.push_str(&utf8_percent_encode(segment, PATH_SEGMENT).to_string());
    }
    result
}

pub fn load_json_url(url: &str) -> DatasetResult<Value> {
    let bytes = fetch_bytes(url)?;
    serde_json::from_slice(&bytes).map_err(|error| DatasetError::InvalidJson {
        url: url.to_owned(),
        message: error.to_string(),
    })
}

pub fn download_url_to_path(url: &str, destination: &Path) -> DatasetResult<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|error| DatasetError::Write {
            path: destination.to_path_buf(),
            message: error.to_string(),
        })?;
    }
    let temporary = temporary_path(destination);
    let result = (|| {
        let mut source = open_source(url)?;
        let file = File::create(&temporary).map_err(|error| DatasetError::Write {
            path: destination.to_path_buf(),
            message: error.to_string(),
        })?;
        let mut writer = BufWriter::new(file);
        std::io::copy(&mut source, &mut writer).map_err(|error| DatasetError::Write {
            path: destination.to_path_buf(),
            message: error.to_string(),
        })?;
        writer.flush().map_err(|error| DatasetError::Write {
            path: destination.to_path_buf(),
            message: error.to_string(),
        })?;
        std::fs::rename(&temporary, destination).map_err(|error| DatasetError::Write {
            path: destination.to_path_buf(),
            message: error.to_string(),
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

pub fn load_sha256sums(path: &Path) -> DatasetResult<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path).map_err(|error| DatasetError::Read {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    parse_sha256sums(&text)
}

pub fn parse_sha256sums(text: &str) -> DatasetResult<Vec<(String, String)>> {
    let mut checksums = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Some((sha256, filename)) = line.split_once(char::is_whitespace) else {
            return Err(DatasetError::InvalidChecksum {
                line: line.to_owned(),
            });
        };
        let filename = filename.trim();
        if filename.is_empty() {
            return Err(DatasetError::InvalidChecksum {
                line: line.to_owned(),
            });
        }
        checksums.push((filename.to_owned(), sha256.to_owned()));
    }
    Ok(checksums)
}

fn fetch_bytes(url: &str) -> DatasetResult<Vec<u8>> {
    let mut reader = open_source(url)?;
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| DatasetError::Unreachable {
            url: url.to_owned(),
            message: error.to_string(),
        })?;
    Ok(bytes)
}

fn open_source(url: &str) -> DatasetResult<Box<dyn Read>> {
    let parsed = Url::parse(url).map_err(|error| DatasetError::InvalidUrl {
        url: url.to_owned(),
        message: error.to_string(),
    })?;
    if parsed.scheme() == "file" {
        let path = parsed
            .to_file_path()
            .map_err(|_| DatasetError::InvalidFileUrl {
                url: url.to_owned(),
            })?;
        let file = File::open(&path).map_err(|error| DatasetError::Read {
            path: path.clone(),
            message: error.to_string(),
        })?;
        return Ok(Box::new(BufReader::new(file)));
    }
    let response = Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(|error| DatasetError::Unreachable {
            url: url.to_owned(),
            message: error.to_string(),
        })?
        .get(url)
        .send()
        .map_err(|error| DatasetError::Unreachable {
            url: url.to_owned(),
            message: error.to_string(),
        })?;
    if response.status() != StatusCode::OK {
        return Err(DatasetError::HttpStatus {
            url: url.to_owned(),
            status: response.status().as_u16(),
        });
    }
    Ok(Box::new(response))
}

fn temporary_path(destination: &Path) -> PathBuf {
    let extension = destination
        .extension()
        .map(|value| format!("{}.tmp", value.to_string_lossy()))
        .unwrap_or_else(|| "tmp".to_owned());
    destination.with_extension(extension)
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrainingDataRequest {
    pub dataset: String,
    pub artifact: Option<String>,
    pub split: Option<String>,
    pub release: String,
    pub format: Option<String>,
    #[serde(default)]
    pub omit: Vec<String>,
    pub cache_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub base_url: String,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TrainingSetRequest {
    pub dataset: String,
    pub artifact: String,
    pub split: String,
    pub release: String,
    pub format: String,
    pub cache_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub base_url: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HostedDataRequest {
    pub target: String,
    pub cache_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub base_url: String,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HostedFileRequest {
    pub relative_path: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub base_url: String,
}

#[derive(Clone, Copy)]
struct ArtifactSpec {
    splits: &'static [&'static str],
    formats: &'static [&'static str],
}

pub fn download_training_set(request: &TrainingSetRequest) -> DatasetResult<PathBuf> {
    validate_training_request(
        &request.dataset,
        &request.artifact,
        &request.split,
        &request.format,
    )?;
    let release = resolve_release(&request.dataset, &request.release, &request.base_url)?;
    let root = training_root(
        &request.dataset,
        &release,
        request.cache_dir.as_deref(),
        request.output_dir.as_deref(),
    );
    let filename = training_filename(&request.artifact, &request.split, &request.format)?;
    let checksums_path = root.join("SHA256SUMS");
    let checksums_url = build_url(
        &request.base_url,
        &[
            request.dataset.clone(),
            release.clone(),
            "SHA256SUMS".to_owned(),
        ],
    );
    let artifact_root = root.join(&request.artifact);
    let artifact_path = artifact_root.join(&filename);
    download_verified(
        &artifact_path,
        &checksums_path,
        &checksums_url,
        &format!("{}/{}", request.artifact, filename),
        &build_url(
            &request.base_url,
            &[
                request.dataset.clone(),
                release.clone(),
                request.artifact.clone(),
                filename.clone(),
            ],
        ),
    )?;
    download_verified(
        &artifact_root.join("manifest.json"),
        &checksums_path,
        &checksums_url,
        &format!("{}/manifest.json", request.artifact),
        &build_url(
            &request.base_url,
            &[
                request.dataset.clone(),
                release,
                request.artifact.clone(),
                "manifest.json".to_owned(),
            ],
        ),
    )?;
    Ok(artifact_path)
}

pub fn download_training_data(request: &TrainingDataRequest) -> DatasetResult<Vec<PathBuf>> {
    validate_training_filter(request)?;
    let release = resolve_release(&request.dataset, &request.release, &request.base_url)?;
    let root = training_root(
        &request.dataset,
        &release,
        request.cache_dir.as_deref(),
        request.output_dir.as_deref(),
    );
    let checksums_path = root.join("SHA256SUMS");
    let checksums_url = build_url(
        &request.base_url,
        &[
            request.dataset.clone(),
            release.clone(),
            "SHA256SUMS".to_owned(),
        ],
    );
    download_checksums(&checksums_url, &checksums_path)?;
    let published = load_sha256sums(&checksums_path)?;
    let selected: Vec<_> = published
        .into_iter()
        .filter(|(key, _)| {
            training_file_matches(
                key,
                request.artifact.as_deref(),
                request.split.as_deref(),
                request.format.as_deref(),
                &request.omit,
            )
        })
        .collect();
    if selected.is_empty() {
        return Err(policy(
            DatasetErrorCategory::Resolution,
            "dataset.no_matching_files",
            "no published training data files match request",
            json!({"dataset": request.dataset, "release": release}),
        ));
    }
    let paths: Vec<_> = selected.iter().map(|(key, _)| root.join(key)).collect();
    if !request.dry_run {
        for ((key, _), path) in selected.iter().zip(&paths) {
            let mut segments = vec![request.dataset.clone(), release.clone()];
            segments.extend(key.split('/').map(str::to_owned));
            download_verified(
                path,
                &checksums_path,
                &checksums_url,
                key,
                &build_url(&request.base_url, &segments),
            )?;
        }
    }
    Ok(paths)
}

pub fn download_hosted_data(request: &HostedDataRequest) -> DatasetResult<Vec<PathBuf>> {
    let root = hosted_root(request.cache_dir.as_deref(), request.output_dir.as_deref());
    let checksums_path = root.join("SHA256SUMS");
    let checksums_url = build_url(&request.base_url, &["SHA256SUMS".to_owned()]);
    download_checksums(&checksums_url, &checksums_path)?;
    let published = load_sha256sums(&checksums_path)?;
    let selected: Vec<_> = published
        .into_iter()
        .filter(|(key, _)| hosted_target_matches(&request.target, key).unwrap_or(false))
        .collect();
    if !is_hosted_target(&request.target) {
        return Err(policy(
            DatasetErrorCategory::Configuration,
            "dataset.unsupported_target",
            format!("unsupported hosted data target: {}", request.target),
            json!({"target": request.target}),
        ));
    }
    if selected.is_empty() {
        return Err(policy(
            DatasetErrorCategory::Resolution,
            "dataset.no_matching_files",
            format!("no hosted data files match target: {}", request.target),
            json!({"target": request.target}),
        ));
    }
    let paths: Vec<_> = selected.iter().map(|(key, _)| root.join(key)).collect();
    if !request.dry_run {
        for ((key, _), path) in selected.iter().zip(&paths) {
            let segments: Vec<_> = key.split('/').map(str::to_owned).collect();
            download_verified(
                path,
                &checksums_path,
                &checksums_url,
                key,
                &build_url(&request.base_url, &segments),
            )?;
        }
    }
    Ok(paths)
}

pub fn download_hosted_file(request: &HostedFileRequest) -> DatasetResult<PathBuf> {
    let root = hosted_root(request.cache_dir.as_deref(), request.output_dir.as_deref());
    let local_path = root.join(&request.relative_path);
    let checksums_path = root.join("SHA256SUMS");
    let checksums_url = build_url(&request.base_url, &["SHA256SUMS".to_owned()]);
    let key = request
        .relative_path
        .iter()
        .map(|part| part.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let segments = request
        .relative_path
        .iter()
        .map(|part| part.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    download_verified(
        &local_path,
        &checksums_path,
        &checksums_url,
        &key,
        &build_url(&request.base_url, &segments),
    )?;
    Ok(local_path)
}

fn validate_training_filter(request: &TrainingDataRequest) -> DatasetResult<()> {
    if request.dataset != "paroutes" {
        return Err(configuration(
            "dataset.unsupported_dataset",
            format!("unsupported training dataset: {}", request.dataset),
            json!({"dataset": request.dataset, "supported_datasets": ["paroutes"]}),
        ));
    }
    if let Some(artifact) = request.artifact.as_deref()
        && artifact_spec(artifact).is_none()
    {
        return Err(configuration(
            "dataset.unsupported_artifact",
            format!("unsupported training artifact: {artifact}"),
            json!({"artifact": artifact}),
        ));
    }
    if let Some(split) = request.split.as_deref()
        && !matches!(split, "all" | "training" | "validation")
    {
        return Err(configuration(
            "dataset.unsupported_split",
            format!("unsupported training split: {split}"),
            json!({"split": split}),
        ));
    }
    if let Some(format) = request.format.as_deref()
        && !matches!(format, "jsonl" | "rsmi")
    {
        return Err(configuration(
            "dataset.unsupported_format",
            format!("unsupported training dataset format: {format}"),
            json!({"format": format}),
        ));
    }
    for omitted in &request.omit {
        if !matches!(
            omitted.as_str(),
            "all" | "training" | "validation" | "jsonl" | "rsmi"
        ) {
            return Err(configuration(
                "dataset.unsupported_omit_part",
                format!("unsupported omit part: {omitted}"),
                json!({"omit_part": omitted}),
            ));
        }
    }
    Ok(())
}

pub fn validate_training_request(
    dataset: &str,
    artifact: &str,
    split: &str,
    format: &str,
) -> DatasetResult<()> {
    let request = TrainingDataRequest {
        dataset: dataset.to_owned(),
        artifact: Some(artifact.to_owned()),
        split: Some(split.to_owned()),
        release: String::new(),
        format: Some(format.to_owned()),
        omit: Vec::new(),
        cache_dir: None,
        output_dir: None,
        base_url: String::new(),
        dry_run: false,
    };
    validate_training_filter(&request)?;
    let spec = artifact_spec(artifact).expect("validated artifact");
    if !spec.splits.contains(&split) {
        return Err(configuration(
            "dataset.split_mismatch",
            format!("artifact '{artifact}' does not support split '{split}'"),
            json!({"artifact": artifact, "split": split, "supported_splits": spec.splits}),
        ));
    }
    if !spec.formats.contains(&format) {
        return Err(configuration(
            "dataset.format_mismatch",
            format!("artifact '{artifact}' does not support format '{format}'"),
            json!({"artifact": artifact, "format": format, "supported_formats": spec.formats}),
        ));
    }
    Ok(())
}

fn artifact_spec(artifact: &str) -> Option<ArtifactSpec> {
    let all = &["all"];
    let splits = &["all", "training", "validation"];
    let jsonl = &["jsonl"];
    let both = &["jsonl", "rsmi"];
    Some(match artifact {
        "n1-routes" | "n5-routes" => ArtifactSpec {
            splits: all,
            formats: jsonl,
        },
        "route-holdout-n1-n5" | "reaction-holdout-n1-n5" => ArtifactSpec {
            splits,
            formats: jsonl,
        },
        "n1-single-step-reactions" | "n5-single-step-reactions" => ArtifactSpec {
            splits: all,
            formats: both,
        },
        "single-step-route-holdout-n1-n5" | "single-step-reaction-holdout-n1-n5" => ArtifactSpec {
            splits,
            formats: both,
        },
        _ => return None,
    })
}

pub fn training_filename(artifact: &str, split: &str, format: &str) -> DatasetResult<String> {
    validate_training_request("paroutes", artifact, split, format)?;
    Ok(match format {
        "jsonl" => format!("{split}.jsonl.gz"),
        "rsmi" => format!("{split}.rsmi.txt.gz"),
        _ => unreachable!(),
    })
}

pub fn training_file_matches(
    key: &str,
    artifact: Option<&str>,
    split: Option<&str>,
    format: Option<&str>,
    omit: &[String],
) -> bool {
    if artifact.is_some_and(|artifact| !key.starts_with(&format!("{artifact}/"))) {
        return false;
    }
    let filename = Path::new(key)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(key);
    if filename == "manifest.json" {
        return artifact.is_some() || (split.is_none() && format.is_none());
    }
    if split.is_some_and(|part| !file_part_matches(filename, part)) {
        return false;
    }
    if format == Some("jsonl") && !filename.ends_with(".jsonl.gz") {
        return false;
    }
    if format == Some("rsmi") && !filename.ends_with(".rsmi.txt.gz") {
        return false;
    }
    !omit.iter().any(|part| file_part_matches(filename, part))
}

fn file_part_matches(filename: &str, part: &str) -> bool {
    filename == format!("{part}.jsonl.gz")
        || filename == format!("{part}.rsmi.txt.gz")
        || filename.contains(&format!(".{part}."))
}

pub fn resolve_release(dataset: &str, release: &str, base_url: &str) -> DatasetResult<String> {
    if release != "latest" {
        return Ok(release.to_owned());
    }
    let url = build_url(base_url, &[dataset.to_owned(), "latest.json".to_owned()]);
    let payload = load_json_url(&url).map_err(|error| transport_policy(error, true))?;
    let Some(resolved_dataset) = payload.get("dataset").and_then(Value::as_str) else {
        return Err(policy(
            DatasetErrorCategory::ArtifactFormat,
            "dataset.invalid_latest_payload",
            format!("invalid latest release payload for training dataset '{dataset}'"),
            json!({"dataset": dataset, "url": url}),
        ));
    };
    let Some(latest_release) = payload.get("latest_release").and_then(Value::as_str) else {
        return Err(policy(
            DatasetErrorCategory::ArtifactFormat,
            "dataset.invalid_latest_payload",
            format!("invalid latest release payload for training dataset '{dataset}'"),
            json!({"dataset": dataset, "url": url}),
        ));
    };
    if resolved_dataset != dataset {
        return Err(policy(
            DatasetErrorCategory::Resolution,
            "dataset.latest_dataset_mismatch",
            format!(
                "latest release pointer dataset mismatch: expected '{dataset}', got '{resolved_dataset}'"
            ),
            json!({"dataset": dataset, "resolved_dataset": resolved_dataset, "url": url}),
        ));
    }
    Ok(latest_release.to_owned())
}

fn download_verified(
    local_path: &Path,
    checksums_path: &Path,
    checksums_url: &str,
    checksum_key: &str,
    download_url: &str,
) -> DatasetResult<String> {
    let expected = resolve_expected(checksums_path, checksums_url, checksum_key)?;
    if local_path.exists()
        && crate::provenance::file_hash(local_path).ok().as_deref() == Some(expected.as_str())
    {
        return Ok(expected);
    }
    download_url_to_path(download_url, local_path)
        .map_err(|error| transport_policy(error, false))?;
    let actual = crate::provenance::file_hash(local_path).map_err(|error| {
        policy(
            DatasetErrorCategory::Download,
            "dataset.cache_read_failed",
            error.to_string(),
            json!({"path": local_path}),
        )
    })?;
    if actual != expected {
        let _ = std::fs::remove_file(local_path);
        let _ = std::fs::remove_file(checksums_path);
        return Err(policy(
            DatasetErrorCategory::Verification,
            "dataset.hash_mismatch",
            format!("downloaded dataset file failed integrity verification: {checksum_key}"),
            json!({"expected_sha256": expected, "actual_sha256": actual, "key": checksum_key}),
        ));
    }
    Ok(expected)
}

pub fn resolve_expected(path: &Path, url: &str, key: &str) -> DatasetResult<String> {
    let mut checksums = if path.exists() {
        load_sha256sums(path)?
    } else {
        download_checksums(url, path)?
    };
    if let Some((_, value)) = checksums.iter().find(|(candidate, _)| candidate == key) {
        return Ok(value.clone());
    }
    download_checksums(url, path)?;
    checksums = load_sha256sums(path)?;
    checksums
        .into_iter()
        .find_map(|(candidate, value)| (candidate == key).then_some(value))
        .ok_or_else(|| {
            policy(
                DatasetErrorCategory::Resolution,
                "dataset.file_not_published",
                format!("dataset file is not published: {key}"),
                json!({"key": key}),
            )
        })
}

fn download_checksums(url: &str, path: &Path) -> DatasetResult<Vec<(String, String)>> {
    download_url_to_path(url, path).map_err(|error| transport_policy(error, false))?;
    load_sha256sums(path)
}

pub fn training_root(
    dataset: &str,
    release: &str,
    cache: Option<&Path>,
    output: Option<&Path>,
) -> PathBuf {
    if let Some(output) = output {
        output.join(release)
    } else if let Some(cache) = cache {
        cache.join(dataset).join(release)
    } else {
        default_cache_root()
            .join("training-sets")
            .join(dataset)
            .join(release)
    }
}

pub fn hosted_root(cache: Option<&Path>, output: Option<&Path>) -> PathBuf {
    output
        .or(cache)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_cache_root().join("data"))
}

fn default_cache_root() -> PathBuf {
    if let Some(value) = std::env::var_os("RETROCAST_CACHE_DIR") {
        return PathBuf::from(value);
    }
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache")
        .join("retrocast")
}

fn is_hosted_target(target: &str) -> bool {
    matches!(
        target,
        "all"
            | "benchmarks"
            | "definitions"
            | "stocks"
            | "raw"
            | "processed"
            | "scored"
            | "results"
            | "mkt-lin-500"
            | "mkt-cnv-160"
            | "mkt-cnv-160-depth"
            | "mkt-cnv-160-leaf"
            | "mkt-cnv-160-leaf-depth"
            | "ref-lin-600"
            | "ref-cnv-400"
            | "ref-lng-84"
    )
}

fn hosted_target_matches(target: &str, key: &str) -> Option<bool> {
    Some(match target {
        "all" => true,
        "benchmarks" => key.starts_with("1-benchmarks"),
        "definitions" => key.starts_with("1-benchmarks/definitions"),
        "stocks" => key.starts_with("1-benchmarks/stocks"),
        "raw" => key.starts_with("2-raw"),
        "processed" => key.starts_with("3-processed"),
        "scored" => key.starts_with("4-scored"),
        "results" => key.starts_with("5-results"),
        "mkt-lin-500" => benchmark_dependency(key, &["mkt-lin-500.", "buyables-stock"]),
        "mkt-cnv-160" => benchmark_dependency(key, &["mkt-cnv-160.", "buyables-stock"]),
        "mkt-cnv-160-depth" => benchmark_dependency(key, &["mkt-cnv-160-depth.", "buyables-stock"]),
        "mkt-cnv-160-leaf" => benchmark_dependency(key, &["mkt-cnv-160-leaf.", "buyables-stock"]),
        "mkt-cnv-160-leaf-depth" => {
            benchmark_dependency(key, &["mkt-cnv-160-leaf-depth.", "buyables-stock"])
        }
        "ref-lin-600" => benchmark_dependency(key, &["ref-lin-600", "n5-stock"]),
        "ref-cnv-400" => benchmark_dependency(key, &["ref-cnv-400", "n5-stock"]),
        "ref-lng-84" => benchmark_dependency(key, &["ref-lng-84", "n1-n5-stock"]),
        _ => return None,
    })
}

fn benchmark_dependency(key: &str, parts: &[&str]) -> bool {
    key.starts_with("1-benchmarks/") && parts.iter().any(|part| key.contains(part))
}

fn configuration(code: &str, message: String, context: Value) -> DatasetError {
    policy(DatasetErrorCategory::Configuration, code, message, context)
}

fn policy(
    category: DatasetErrorCategory,
    code: &str,
    message: impl Into<String>,
    context: Value,
) -> DatasetError {
    DatasetError::Policy {
        category,
        code: code.to_owned(),
        message: message.into(),
        context,
        retryable: false,
    }
}

fn transport_policy(error: DatasetError, metadata: bool) -> DatasetError {
    match error {
        DatasetError::HttpStatus { url, status } => policy(
            DatasetErrorCategory::Resolution,
            if metadata {
                "dataset.metadata_http_error"
            } else {
                "dataset.file_http_error"
            },
            format!("hosted endpoint returned HTTP {status} for {url}"),
            json!({"url": url, "status": status}),
        ),
        DatasetError::InvalidJson { url, .. } => policy(
            DatasetErrorCategory::ArtifactFormat,
            "dataset.invalid_metadata_json",
            format!("invalid json returned by hosted metadata endpoint {url}"),
            json!({"url": url}),
        ),
        error => policy(
            DatasetErrorCategory::Download,
            if metadata {
                "dataset.metadata_unreachable"
            } else {
                "dataset.file_unreachable"
            },
            error.to_string(),
            json!({}),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_segments_are_encoded_without_losing_file_suffixes() {
        assert_eq!(
            build_url(
                "https://example.test/root/",
                &["pa routes".to_owned(), "x.json.gz".to_owned()]
            ),
            "https://example.test/root/pa%20routes/x.json.gz"
        );
    }

    #[test]
    fn checksum_parser_preserves_publication_order() {
        assert_eq!(
            parse_sha256sums("b second\na first\n").unwrap(),
            [
                ("second".to_owned(), "b".to_owned()),
                ("first".to_owned(), "a".to_owned())
            ]
        );
    }

    #[test]
    fn training_selection_keeps_manifest_with_an_artifact_filter() {
        assert!(training_file_matches(
            "n1-routes/manifest.json",
            Some("n1-routes"),
            Some("all"),
            Some("jsonl"),
            &[]
        ));
        assert!(!training_file_matches(
            "n1-routes/all.jsonl.gz",
            Some("n1-routes"),
            Some("all"),
            Some("jsonl"),
            &["all".to_owned()]
        ));
    }

    #[test]
    fn benchmark_target_selects_definition_and_stock_dependency() {
        assert_eq!(
            hosted_target_matches(
                "mkt-lin-500",
                "1-benchmarks/definitions/mkt-lin-500.json.gz"
            ),
            Some(true)
        );
        assert_eq!(
            hosted_target_matches("mkt-lin-500", "1-benchmarks/stocks/buyables-stock.csv.gz"),
            Some(true)
        );
    }
}
