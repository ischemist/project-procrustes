# RetroCast 0.7.1 Python oracle

This directory preserves the pure-Python implementation from the `v0.7.1` tag for differential testing while the Rust engine is battle-tested.

It is not a release line and must not be published. New features and fixes belong in `packages/retrocast-rs`. Run this package in a separate environment because it and the published Rust-backed distribution both provide the `retrocast` import namespace.

```console
cd packages/retrocast-py
uv sync --extra viz
uv run pytest
```
