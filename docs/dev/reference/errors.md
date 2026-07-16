---
icon: lucide/triangle-alert
---

# Error Handling

`retrocast-core` owns one `EngineError` boundary. The CLI renders it and exits non-zero. PyO3 translates it into a built-in Python exception. The published package does not contain a second Python exception hierarchy.

Expected route-local failures are different: they become `FailureRecord` data and keep their planner rank.

## Interface Boundaries

=== "Python 0.8.x"

    | Exception | Meaning |
    | --- | --- |
    | `ValueError` | Invalid SMILES, InChIKey, chemical level, or typed artifact data |
    | `OSError` | An artifact path could not be read or written |
    | `RuntimeError` | Adapter, schema, worker-pool, scoring, analysis, or another native failure |

    ```python
    try:
        predictions = retrocast.ingest(raw, "aizynthfinder", task)
    except (ValueError, OSError, RuntimeError) as error:
        logger.exception("RetroCast ingest failed: %s", error)
        raise
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::error::{EngineError, Result};

    fn run() -> Result<()> {
        let predictions = ingest(raw, adapter, &task, mode, limit, workers)?;
        // ...
        Ok(())
    }
    ```

=== "Python 0.7.1"

    ```python
    from retrocast.exceptions import AdapterError, ChemError, RetroCastException

    try:
        predictions = ingest_candidates(raw_payload, adapter, task)
    except (AdapterError, ChemError) as error:
        logger.error("%s: %s", error.code, error)
        raise
    ```

Do not branch on Python exception prose. Rust code should match a specific `EngineError` variant only when it can recover meaningfully.

## Failure Records

A malformed route entry can be counted without aborting the rest of the planner artifact:

```json
{
  "rank": 2,
  "failure": {
    "code": "adapter.schema_invalid",
    "target_id": "target-1",
    "context": {"adapter": "aizynthfinder"}
  }
}
```

The distinction is:

- a route-local failure becomes a ranked candidate when processing can continue
- an invalid artifact envelope, unsafe path, unknown adapter, or impossible workflow request aborts the call

Inspect `FailureRecord.code` when route-level accounting matters:

=== "Python 0.8.x"

    ```python
    failures = [
        candidate["failure"]["code"]
        for target_candidates in predictions.to_dict().values()
        for candidate in target_candidates
        if candidate.get("failure") is not None
    ]
    ```

=== "Rust 0.8.x"

    ```rust
    let failures = predictions
        .values()
        .flatten()
        .filter_map(|candidate| candidate.failure.as_ref())
        .map(|failure| failure.code.as_str())
        .collect::<Vec<_>>();
    ```

=== "Python 0.7.1"

    ```python
    failures = [
        candidate.failure.code
        for target_candidates in predictions.values()
        for candidate in target_candidates
        if candidate.failure is not None
    ]
    ```

## Stable Code Families

Serialized failure codes are the durable route-level contract. Human-readable messages may become clearer without changing the code.

| Family | Examples | Meaning |
| --- | --- | --- |
| `input.*` | `input.invalid` | External CLI or configuration input is invalid. |
| `security.*` | `security.path_invalid`, `security.path_traversal` | A path or filename is unsafe. |
| `config.*` | `config.invalid_value` | Configuration cannot be interpreted safely. |
| `chem.*` | `chem.invalid_smiles`, `chem.runtime_error` | Chemistry parsing, identity, or RDKit failed. |
| `curation.*` | `curation.route_embedding_query_invalid` | Valid data is invalid for the requested curation operation. |
| `schema.*` | `schema.logic_error` | Structurally valid data is logically impossible. |
| `benchmark.*` | `benchmark.validation_failed` | Benchmark construction or uniqueness failed. |
| `adapter.*` | `adapter.schema_invalid`, `adapter.target_mismatch`, `adapter.cycle_detected` | Raw planner output failed an adapter contract. |
| `io.*` | `io.not_found`, `io.decode_failed`, `io.invalid_artifact_shape` | Artifact reading, decoding, shape, or writing failed. |
| `serialization.*` | `serialization.syntheseus_failed` | Conversion to an external format failed. |
| `workflow.*` | `workflow.error` | Orchestration failed at a workflow boundary. |
| `dataset.*` | `dataset.resolution_failed`, `dataset.download_failed`, `dataset.verification_failed` | Hosted data resolution or integrity checking failed. |
| `validity.*` | `validity.unsupported_tier` | No validator implements the requested validity tier. |

Not every family is currently emitted by every workflow. Adding a new code is a schema decision; repurposing an existing code is a breaking change.

## Adapter Policy

An invalid artifact envelope is fatal. An invalid route entry becomes a failed candidate when the adapter can continue to later entries.

Use `adapter.target_mismatch` for a valid route whose canonical root differs from the requested target. Use `adapter.schema_invalid` when the raw payload does not match the adapter's declared shape. Chemistry failures keep their `chem.*` identity.

## Workflow Accounting

Candidate statistics aggregate failures by code:

```json
{
  "statistics": {
    "failures_by_code": {
      "adapter.schema_invalid": 2,
      "chem.runtime_error": 1
    }
  }
}
```

Manifests persist those counts so an ingest run can be audited without reparsing the raw planner output.

## Rust Rules

- Return `Result<T, EngineError>` across core boundaries.
- Prefer a specific enum variant to an unstructured string.
- Preserve source errors where the original cause matters.
- Convert an adapter error into `FailureRecord` only at the boundary that owns rank and target context.
- Validate path components before joining managed artifact paths.
- Never panic on caller-controlled data.

## CLI Policy

The CLI is the process boundary. It prints a diagnostic and exits non-zero for fatal failures. Route-local failures do not make ingest fail when they have been preserved as candidate data.

## Tests

Test the `EngineError` variant or serialized failure code. Assert exact messages only when CLI copy is itself the contract.

Adversarial coverage should include malformed JSON, invalid chemistry, unsafe paths, zero workers, partial route corruption, deterministic concurrent execution, and Python exception translation.

See [adapter errors](adapters.md#adapter-errors) for route-local examples.
