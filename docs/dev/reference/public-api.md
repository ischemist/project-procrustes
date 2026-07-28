---
icon: lucide/package-check
---

# Public API

The PyPI distribution exposes one import surface: the Maturin-built `retrocast` extension. There are no Python subpackages or compatibility implementation.

## Policy

- Friendly workflow names (`adapt`, `ingest`, `score`, `analyze`, `evaluate`) accept JSON-compatible Python values or native handles.
- Corpus-sized stages return opaque Rust-owned handles. Materialization must be explicit through `.to_dict()`, `.json()`, or `.write()`.
- File entry points pass paths into Rust; Python must not read or decompress corpus artifacts first.
- Producer functions expose task, stock, execution-stat, artifact, and provenance contracts as JSON-compatible Python values.
- Chemistry helpers call the same RDKit C++ bridge as the CLI.
- Low-level `*_json` functions are binding plumbing and are not the preferred user API.
- `retrocast.__all__` and `retrocast.pyi` define the curated Python surface. Names absent from both are internal even when extension-module introspection can see them.
- Adding or removing a top-level function is a public API change even when the Rust core function already exists.

The frozen `packages/retrocast-py` tree has its own v0.7.1 package surface. It is a differential-testing oracle, not a fallback imported by the native distribution.

## Review Check

When a binding is added, ask whether Python needs a friendly value-level function, a file-level function, or both. Do not add a Python implementation merely to reshape data that Rust already owns.
