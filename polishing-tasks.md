# polishing tasks

status: complete

## completed decisions

the adapter workflow now uses the 2-axis vocabulary:

- raw vs canonical
- target-keyed vs not target-keyed

the public raw-input nouns are:

- `provider_output`: one raw blob emitted by a planner, service, script, notebook, or dataset dump. the adapter owns splitting it into `RawRouteEntry` values.
- `target_keyed_provider_output`: raw provider output keyed by target id or target smiles. the workflow owns resolving keys against a benchmark before dispatching per-target payloads to the adapter.

the canonical output nouns are:

- `routes`: in-memory ordered `list[Route]`
- `routes_by_target`: in-memory `dict[target_id, list[Route]]`
- `route_corpus`: persisted unkeyed ordered route artifact, such as `jsonl.gz`
- `target_keyed_routes`: persisted or in-memory benchmark-aligned route mapping

## completed implementation

- renamed `adapt_route_corpus(...)` to `adapt_provider_output(...)`
- renamed `adapt_benchmark_keyed_route_corpus(...)` to `adapt_target_keyed_provider_output(...)`
- added `adapt_route(...)` for target-free single-route adaptation
- kept deprecated aliases for both old names with `RetroCastFutureWarning`
- deprecated `adapt_single_route(...)` with guidance to use `adapt_route(...)`
- removed `dict`-shape inference from workflow and ad hoc cli adaptation
- added explicit `--input-kind provider-output | target-keyed-provider-output`
- defaulted ad hoc `retrocast adapt` to `provider-output`
- defaulted project `retrocast ingest` to `target-keyed-provider-output`
- made `adapt_single_route(...)` delegate through `adapt_target_routes(...)`
- preserved public `ADAPTER_MAP` as instance-valued
- added `ADAPTER_TYPES` for the class-valued registry used by `get_adapter(...)`
- simplified collector policies to `ignore | error`
- kept `skip` and `report` as warning aliases for the old ignore behavior
- removed the unused `matched_by_target_hint` stats field
- kept `Route.rank` removed from the canonical `Route` model
- added regression coverage showing legacy serialized `rank` fields are ignored on load
- updated docs to describe list order as the canonical route ordering signal

## compatibility stance

library callable renames use `FutureWarning` through `RetroCastFutureWarning`.

deprecated names scheduled for removal in `0.7`:

- `adapt_route_corpus(...)`
- `adapt_benchmark_keyed_route_corpus(...)`
- `adapt_single_route(...)`
- `on_unmatched="skip" | "report"`
- `on_ambiguous="skip" | "report"`

`Route.rank` is not restored and does not get a runtime shim. old serialized route artifacts with a `rank` field still load because the field is ignored by the `Route` schema.

## verification

- `uv run ruff check src tests`
- `uv run pytest -q`

latest result:

- ruff passed
- pytest passed: `870 passed`
