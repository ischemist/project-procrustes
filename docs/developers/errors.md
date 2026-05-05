# error handling

retrocast treats expected boundary failures as typed exceptions with stable codes.
messages are for humans; codes and context are the contract.

## rules

- raise `RetroCastException` subclasses only at package, cli, io, adapter, workflow, and benchmark boundaries.
- preserve causes with `raise ... from exc` when wrapping filesystem, json, gzip, pydantic, rdkit, or serializer failures.
- let ordinary `ValueError`, `TypeError`, and `RuntimeError` represent local programmer errors.
- library code raises; cli and workflow boundaries decide whether to abort, skip, count, or continue.
- never branch on exception prose. branch on `error.code`.

## shape

all boundary-facing exceptions expose:

```json
{
    "code": "adapter.schema_invalid",
    "message": "raw data for target 'x' failed dms schema validation",
    "context": {"adapter": "dms", "target_id": "x"},
    "retryable": false,
}
```

use `error.to_dict()` when persisting or reporting the structured form.

## taxonomy

| family | base class | examples | meaning |
| --- | --- | --- | --- |
| `input.*` | `InputError` | `security.path_invalid` | external cli/config/path input is invalid |
| `config.*` | `ConfigurationError` | `config.invalid_color` | config values cannot be interpreted safely |
| `chem.*` | `ChemError` | `chem.invalid_smiles`, `chem.runtime_error` | chemistry parsing, identity, or backend failure |
| `schema.*` | `SchemaLogicError` | `schema.logic_error` | data is structurally valid but logically impossible |
| `benchmark.*` | `BenchmarkError` | `benchmark.validation_failed` | benchmark construction or uniqueness contract failed |
| `adapter.*` | `AdapterError` | `adapter.schema_invalid`, `adapter.target_mismatch` | raw model output failed adapter boundary contracts |
| `io.*` | `DataIOError` | `io.not_found`, `io.decode_failed`, `io.invalid_artifact_shape`, `io.write_failed` | artifact read, decode, shape, or write failure |
| `serialization.*` | `RetroCastSerializationError` | `serialization.syntheseus_failed` | conversion to external route formats failed |
| `workflow.*` | `WorkflowError` | `workflow.error` | orchestration failed at a workflow boundary |

## adapter policy

adapter schema errors are fatal for one target payload and counted by the workflow.
individual route transform errors may be logged and skipped when the adapter can keep processing the rest of the target.
route root mismatches should use `adapter.target_mismatch`; malformed raw payloads should use `adapter.schema_invalid`.

## workflow accounting

batch workflows that continue after expected failures must count them by stable code.
ingestion persists `failures_by_code` in manifests so runs can be compared without log scraping.

## cli policy

the cli is the process boundary. it formats known `RetroCastException` instances with code and context, logs unexpected errors with tracebacks, and exits non-zero for fatal command failures.

## tests

test the exception class and `code`; assert exact message text only when copy itself is the public contract.
when wrapping matters, assert `__cause__` is present.
