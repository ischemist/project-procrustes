pub mod adapt;
pub mod adapters;
pub mod analyze;
pub mod chem;
pub mod curation;
pub mod dataset;
pub mod embedding;
pub mod embedding_audit;
pub mod error;
pub mod io;
pub mod model;
pub mod pipeline;
pub mod provenance;
pub mod route;
pub mod route_path;
pub mod route_view;
pub mod sampling;
pub mod schema;
pub mod score;
pub mod stats;
pub mod training;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use rayon::ThreadPoolBuilder;

use crate::error::{EngineError, Result};

pub fn with_pool<T: Send>(workers: usize, operation: impl FnOnce() -> T + Send) -> Result<T> {
    if workers == 0 {
        return Err(EngineError::InvalidWorkers(workers));
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("retrocast-{index}"))
        .build()
        .map_err(|error| {
            EngineError::AdapterSchema(format!("failed to build worker pool: {error}"))
        })?;
    Ok(pool.install(operation))
}
