---
icon: lucide/calendar-clock
---

# Deprecation Schedule

RetroCast uses warnings to keep pre-1.0 changes visible without forcing every downstream project to migrate immediately. A deprecated API remains usable until the removal version listed here, emits `RetroCastFutureWarning`, and should have a documented replacement before removal.

## Internal Policy

- Treat the removal version in the warning as the public contract.
- Prefer one minor release of warning time for narrow aliases and two minor releases for workflow/API shape changes.
- Do not remove deprecated behavior in patch releases.
- Update this page in the same PR that adds, delays, or removes a deprecation.

## Scheduled Removals

| removal | deprecated surface | replacement |
| --- | --- | --- |
| `0.9.0` | legacy adapter slugs such as `aizynth`, `dms`, and `dreamretro` | canonical adapter slugs from `retrocast list-adapters`, such as `aizynthfinder`, `directmultistep`, and `dreamretroer` |
