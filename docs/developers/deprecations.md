---
icon: lucide/calendar-clock
---

# Deprecation Schedule

RetroCast uses warnings to keep pre-1.0 cleanup visible without forcing every downstream project to migrate immediately. A deprecated API remains usable until the removal version listed here, emits `RetroCastFutureWarning`, and should have a documented replacement before removal.

## Policy

- Treat the removal version in the warning as the public contract.
- Prefer one minor release of warning time for narrow aliases and two minor releases for workflow/API shape changes.
- Do not remove deprecated behavior in patch releases.
- Update this page in the same PR that adds, delays, or removes a deprecation.

## Scheduled Removals

| removal | deprecated surface | replacement |
| --- | --- | --- |
| `0.9.0` | legacy adapter slugs such as `aizynth`, `dms`, `dreamretro`, and `ursa-llm` | canonical adapter slugs from `retrocast list-adapters`, such as `aizynthfinder`, `directmultistep`, `dreamretroer`, and `ursa` |
| `0.9.0` | legacy adapter class aliases such as `AizynthAdapter`, `DMSAdapter`, `DreamRetroAdapter`, `TtlRetroAdapter`, `RetrochimeraAdapter`, `SynLLaMaAdapter`, `SynLlaMaAdapter`, and `UrsaLlmAdapter` | canonical adapter classes such as `AiZynthFinderAdapter`, `DirectMultiStepAdapter`, `DreamRetroErAdapter`, `MultiStepTTLAdapter`, `RetroChimeraAdapter`, `SynLlamaAdapter`, and `UrsaAdapter` |
| `0.9.0` | `adapt_single_route(...)` and `adapt_routes(...)` | `adapt_route(...)`, `adapt_provider_output(...)`, or `adapt_target_keyed_provider_output(...)`, depending on input shape |
| `0.9.0` | `adapt_route_corpus(...)` and `adapt_benchmark_keyed_route_corpus(...)` | `adapt_provider_output(...)` and `adapt_target_keyed_provider_output(...)` |
| `0.9.0` | collection policies `on_unknown_target="skip"`, `on_unknown_target="report"`, `on_ambiguous_target="skip"`, and `on_ambiguous_target="report"` | `on_unknown_target="ignore"` or `on_ambiguous_target="ignore"` |

## Migration Notes

Adapter names now distinguish public model names from compatibility aliases. CLI users should pass canonical slugs; library users should import canonical classes. The aliases still work until `0.9.0`, but new docs and examples should not introduce them.

The adaptation API is moving away from target-local helpers toward explicit workflow names. Use `adapt_route(...)` for a single raw route, `adapt_provider_output(...)` when adapting a streamable provider artifact into `PredictedRoute` objects, and `adapt_target_keyed_provider_output(...)` when the raw artifact is already keyed by benchmark target.
