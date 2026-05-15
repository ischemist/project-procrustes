---
icon: lucide/triangle-alert
---


# Error Handling

RetroCast treats expected boundary failures as typed exceptions with stable codes. Messages are for humans; codes and context are the contract.

!!! info "Codes are the public contract"

    Do not branch on exception prose. Branch on `error.code`.

## Rules

- Raise `RetroCastException` subclasses only at package, CLI, I/O, adapter, workflow, and benchmark boundaries.
- Preserve causes with `raise ... from exc` when wrapping filesystem, JSON, gzip, Pydantic, RDKit, or serializer failures.
- Let ordinary `ValueError`, `TypeError`, and `RuntimeError` represent local programmer errors.
- Library code raises; CLI and workflow boundaries decide whether to abort, skip, count, or continue.
- Never branch on exception prose. Branch on `error.code`.

## Shape

All boundary-facing exceptions expose:

```json
{
    "code": "adapter.schema_invalid",
    "message": "raw data for target 'x' failed DMS schema validation",
    "context": {"adapter": "dms", "target_id": "x"},
    "retryable": false
}
```

Use `error.to_dict()` when persisting or reporting the structured form.

## Examples

Wrap expected boundary failures with a stable code and context:

```python
from pydantic import ValidationError

from retrocast.adapters.errors import adapter_schema_error

try:
    validated = MyRawOutput.model_validate(raw_target_data)
except ValidationError as exc:
    raise adapter_schema_error("dms", target.id, "invalid route list") from exc
```

Handle failures by class and code, not by message text:

```python
from retrocast.workflow.adapt import adapt_target_routes
from retrocast.exceptions import AdapterError, ChemError

try:
    routes = list(adapt_target_routes(adapter, raw_payload, target))
except (AdapterError, ChemError) as exc:
    stats.record_failure(exc.code, target_id=target_id)
    routes = []
```

## Taxonomy

| Family | Base class | Examples | Meaning |
| --- | --- | --- | --- |
| `input.*` | `InputError` | `input.invalid` | External CLI/config input is invalid |
| `security.*` | `SecurityError` | `security.path_invalid`, `security.path_traversal` | Path, filename, or filesystem boundary input is unsafe |
| `config.*` | `ConfigurationError` | `config.invalid_color` | Config values cannot be interpreted safely |
| `chem.*` | `ChemError` | `chem.invalid_smiles`, `chem.runtime_error` | Chemistry parsing, identity, or backend failure |
| `schema.*` | `SchemaLogicError` | `schema.logic_error` | Data is structurally valid but logically impossible |
| `benchmark.*` | `BenchmarkError` | `benchmark.validation_failed` | Benchmark construction or uniqueness contract failed |
| `adapter.*` | `AdapterError` | `adapter.schema_invalid`, `adapter.target_mismatch`, `adapter.cycle_detected`, `adapter.route_string_invalid` | Raw model output failed adapter boundary contracts |
| `io.*` | `DataIOError` | `io.not_found`, `io.decode_failed`, `io.invalid_artifact_shape`, `io.unsupported_format`, `io.write_failed` | Artifact read, decode, shape, format, or write failure |
| `serialization.*` | `RetroCastSerializationError` | `serialization.syntheseus_failed` | Conversion to external route formats failed |
| `workflow.*` | `WorkflowError` | `workflow.error` | Orchestration failed at a workflow boundary |

## Adapter Policy

Adapter schema errors are fatal for one raw payload or entry and counted by the workflow. Individual route transform errors may be logged and skipped when the adapter can keep processing the rest of the input artifact.

Route root mismatches should use `adapter.target_mismatch`; malformed raw payloads should use `adapter.schema_invalid`.

## Workflow Accounting

Batch workflows that continue after expected failures must count them by stable code.

For `ingest`, `ingest_model_predictions()` logs the sorted failure summary and returns `stats`. The ingest CLI path then writes `stats.to_manifest_dict()` into the manifest `statistics` block, so counts land under `statistics.failures_by_code`.

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

Training-set curation preserves the adaptation summary shape instead of using the ingest `statistics` block. In those release manifests, the counts live under `summary.adaptation....failures_by_code`.

## CLI Policy

The CLI is the process boundary. It formats known `RetroCastException` instances with code and context, logs unexpected errors with tracebacks, and exits non-zero for fatal command failures.

## Tests

Test the exception class and `code`; assert exact message text only when copy itself is the public contract. When wrapping matters, assert `__cause__` is present.

See [adapter errors](adapters.md#adapter-error-examples) for concrete examples of route-local adapter failures.
