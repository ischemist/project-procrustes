---
icon: lucide/cable
---

# Python Library

RetroCast is a modular Python library for standardizing retrosynthesis planner
outputs, collecting routes onto benchmarks, scoring predictions, and analyzing
model performance without depending on the CLI directory layout.

!!! tip "When to use the Python API"

    - Interactive notebooks and exploratory analysis
    - Custom evaluation loops
    - Integration with existing research pipelines
    - Programmatic access to metrics without file I/O

## Installation

=== "uv (recommended)"

    ```bash
    uv add retrocast
    ```

    For visualization support:

    ```bash
    uv add retrocast[viz]
    ```

=== "pip"

    ```bash
    pip install retrocast
    ```

    For visualization support:

    ```bash
    pip install retrocast[viz]
    ```

## Choose A Guide

| Task | Start here |
| --- | --- |
| Convert raw planner output into canonical `Route` objects | [Adaptation](adaptation.md) |
| Score benchmark-keyed routes against stock | [Evaluation](evaluation.md) |
| Compute confidence intervals and aggregate metrics | [Statistics](statistics.md) |
| Plot model diagnostics and comparisons | [Visualization](visualization.md) |
| Look up public functions quickly | [Reference](reference.md) |

## v0.6 Mental Model

Before v0.6, RetroCast's public mental model centered on `ingest`: take raw
predictions, adapt them, and write benchmark-keyed `routes.json.gz`. That is
still useful, but it hides two different operations.

In v0.6, those operations are exposed as separate library workflows:

| Workflow | Input | Output | Requires benchmark targets? |
| --- | --- | --- | --- |
| Adaptation | Raw provider output | Canonical `Route` objects | No |
| Benchmark collection | Canonical `Route` objects plus a benchmark | Target-keyed route mapping | Yes |

Use adaptation when you want the general library promise: "give me whatever the
planner emitted and I will give you standard `Route` objects." Use benchmark
collection only when you need the `routes.json.gz` shape for benchmark scoring
or aggregate target-level statistics.
