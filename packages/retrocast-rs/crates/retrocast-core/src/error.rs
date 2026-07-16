use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("invalid SMILES {smiles:?}: {message}")]
    InvalidSmiles { smiles: String, message: String },
    #[error("invalid InChIKey {inchikey:?}: {message}")]
    InvalidInchiKey { inchikey: String, message: String },
    #[error("cannot upscale InChIKey {inchikey:?} to {target_level}")]
    InchiKeyUpscale {
        inchikey: String,
        target_level: &'static str,
    },
    #[error("chemistry error for {smiles:?}: {message}")]
    Chemistry { smiles: String, message: String },
    #[error("AiZynthFinder schema error: {0}")]
    AdapterSchema(String),
    #[error("{adapter} schema error: {message}")]
    AdapterSchemaContext {
        adapter: &'static str,
        message: String,
        context: Map<String, Value>,
    },
    #[error("{adapter} adapter error ({code}): {message}")]
    AdapterLogic {
        adapter: &'static str,
        code: &'static str,
        message: String,
    },
    #[error("AiZynthFinder route is not a tree: {0}")]
    RouteShape(String),
    #[error(
        "{adapter} produced mismatched SMILES for target {target_id}. expected canonical: {expected}, but adapter produced: {actual}"
    )]
    TargetMismatch {
        adapter: &'static str,
        target_id: String,
        expected: String,
        actual: String,
    },
    #[error("unsupported task constraint: {0}")]
    UnsupportedConstraint(String),
    #[error("unknown adapter {name:?}; available adapters: {available}")]
    UnknownAdapter { name: String, available: String },
    #[error("unregistered stock: {0}")]
    UnregisteredStock(String),
    #[error("unsafe {label} path component: {value:?}")]
    UnsafePathComponent { label: String, value: String },
    #[error("invalid worker count: {0}")]
    InvalidWorkers(usize),
    #[error("bootstrap resamples must be positive, got {0}")]
    InvalidBootstrapResamples(usize),
    #[error("provenance error: {0}")]
    Provenance(String),
    #[error("JSONL row {line_number}: {message}")]
    Jsonl { line_number: usize, message: String },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Schema(#[from] crate::schema::ScalarError),
}

pub type Result<T> = std::result::Result<T, EngineError>;
