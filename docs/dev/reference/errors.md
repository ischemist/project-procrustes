---
icon: lucide/triangle-alert
---

# Error Handling

`retrocast-core` has one `EngineError` boundary. The CLI renders it and exits non-zero. PyO3 translates it into a built-in Python exception; the published package does not contain a parallel Python exception hierarchy.

## Python Boundary

| Python exception | Meaning |
| --- | --- |
| `ValueError` | Invalid SMILES, InChIKey, chemical level, or typed artifact data |
| `OSError` | An artifact path could not be read or written |
| `RuntimeError` | Adapter, schema, worker-pool, scoring, analysis, or other native engine failure |

Do not branch on Python exception prose. If route-local accounting matters, inspect the `FailureRecord.code` preserved in predictions or evaluation artifacts.

## Failure Records

Expected route-local failures are data, not thrown exceptions. They occupy their planner rank:

```json
{
  "rank": 2,
  "failure": {
    "code": "adapter.schema_invalid",
    "target_id": "target-1",
    "context": { "adapter": "aizynthfinder" }
  }
}
```

This distinction is deliberate:

- an invalid route entry becomes a failed candidate when the rest of the artifact can continue
- an invalid artifact envelope, unsafe path, or impossible workflow request aborts the call

## Rust Rules

- Return `Result<T, EngineError>` across core boundaries.
- Prefer a specific enum variant over a formatted string.
- Preserve source errors with `#[source]` where the original cause matters.
- Convert an adapter error into `FailureRecord` only at the adaptation boundary that owns rank and target context.
- Validate path components before joining them into managed artifact paths.
- Never panic on caller-controlled data.

## Stable Codes

Serialized failure codes use stable families such as `adapter.*`, `chem.*`, `schema.*`, `io.*`, `dataset.*`, and `workflow.*`. Human messages may become clearer without changing those codes.

Candidate statistics and manifests aggregate route-local failures under `failures_by_code`, so parity tests should compare codes and context rather than exact prose.

## Tests

Test the `EngineError` variant or serialized failure code. Assert exact messages only when command-line copy itself is the contract. Adversarial tests should cover malformed JSON, invalid chemistry, unsafe paths, zero workers, partial route corruption, and concurrent execution.
