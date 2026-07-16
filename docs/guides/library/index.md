---
icon: lucide/cable
---

# Python Library

The published Python package is a direct binding to `retrocast-core`. The wheel contains the native extension, a generated import shim, and bundled RDKit C++ libraries; it does not contain the old Python implementation.

Inputs at the Python boundary are JSON-compatible dictionaries and lists. Corpus-sized intermediate values stay in Rust:

```python
import json

import retrocast

raw = json.loads(raw_path.read_text())
task = json.loads(benchmark_path.read_text())
stocks = {"buyables": list(stock_inchikeys)}

predictions = retrocast.ingest(raw, "aizynthfinder", task, workers=12)
evaluation = retrocast.score(predictions, task, stocks, workers=12)
report = retrocast.analyze(evaluation, n_boot=10_000, workers=12)
```

`predictions` is a `NativePredictions` handle. `score` consumes it and moves the route graph into a `NativeEvaluation`, avoiding a second corpus-sized representation. `report` is a normal Python dictionary because analysis output is small.

Call `.write(path)` on a native handle to write schema-v2 JSON or JSON gzip directly from Rust. `.to_dict()` and `.json()` create explicit snapshots for inspection.

For large files, avoid constructing the raw payload in Python:

```python
predictions = retrocast.ingest_file(
    raw_path,
    "aizynthfinder",
    benchmark_path,
    workers=12,
)
```

The all-in-one file API keeps all three stages native and writes their artifacts:

```python
stats = retrocast.pipeline(
    raw_path,
    benchmark_path,
    stock_path,
    output_dir,
    adapter="aizynthfinder",
    workers=12,
)
```

See the [Library Reference](reference.md) for signatures and ownership behavior, and [Schema Design](/dev/rationale/schema-design) for artifact shapes.
