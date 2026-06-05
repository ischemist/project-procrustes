---
icon: lucide/package-check
---

# Public API

RetroCast treats `__all__` entries and package-level re-exports as public API decisions. Add them deliberately; do not update re-export lists just because a new function, class, or helper exists.

## Policy

- `retrocast` is the quickstart surface. It should expose the core schema objects, main workflow verbs, `get_adapter`, and `__version__`.
- Subpackage `__init__.py` files may expose stable domain concepts that callers naturally expect at that namespace.
- Stable leaf utilities may be re-exported when the package name is the user-facing concept. For example, `retrocast.io.save_jsonl_gz` is preferable to making users know about `retrocast.io.blob`.
- Implementation plumbing stays in its owning module. Registry compatibility data, scoring substeps, release-build internals, audit helpers, and migration-only code should not get omnibus re-exports.
- Curation and release-prep modules default to concrete imports from their owning files. Their package roots are not convenience buckets.
- Heavy optional dependencies should not be imported eagerly through a package root. Use concrete imports or lazy package attributes.
- Moving or removing an exported name is an API change. Prefer a documented deprecation path unless the surface is explicitly experimental or internal.

## Review Check

When a PR adds or removes a package export, ask:

> does this create a public contract we actually want?

If the answer is no, import from the owning module. If the answer is yes, the export should be stable, cheap to import, useful outside its defining module, and doc-worthy.
