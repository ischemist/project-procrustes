---
icon: lucide/git-branch
---

# Solv-N Evaluation

This page explains the design rationale behind the Solv-N evaluation model.

## Historical Context

RetroCast originally used "solvability" in the common retrosynthesis sense: a
route was solved if it terminated in the selected stock. That metric is really
stock-termination rate, not universal chemical solvability.

Two external pieces motivate the terminology shift:

- [arxiv:2512.07079](https://arxiv.org/abs/2512.07079) highlighted that reported solvability is often no more than stock-termination rate.
- The [Syntax of Matter preprint](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001278/v3) introduces the Solv-N / Tier-N framing this migration targets.

Those links belong in rationale and concepts docs, not runtime deprecation
warnings.

## Core Separation

The design separates four concepts.

`Tier-i validity` is intrinsic chemical validity. It starts at the reaction level
and lifts to a route only when every reaction in the route passes the tier.

`MetricScope` is the set of user/task constraints that determines whether a
candidate is eligible for a metric. Stock termination is the default scope for
ordinary retrosynthesis benchmarks, but it is not the only possible scope.

`Solv-i[scope]` is the conjunction of intrinsic validity and scope satisfaction:

```text
Solv-i[scope] = Tier-i validity + MetricScope
```

`Acceptable-route reconstruction` is reference matching. It is a conservative
proxy used because higher-tier validity is hard to automate, not the definition
of Solv-i.

## Why Solv-i Is Parameterized

Solvability is always relative to boundary conditions. A route can be solvable
against one stock and unsolvable against another. Future benchmark tasks may add
other constraints, such as requiring a specific starting material or maximum
route depth.

Using `Solv-i[scope]` keeps this explicit:

```text
Solv-1[stock]
Solv-1[stock + required starting material]
Solv-1[stock + required starting material + max depth]
```

The default shorthand remains useful:

```text
Solv-i := Solv-i[stock]
```

but only when the report or artifact declares that the active scope is `stock`.

This avoids inventing parallel names like `constrained_solv_i` while still
making cross-benchmark comparisons safe.

## Why Stock Is Not Special

Stock termination is historically central because retrosynthesis planners are
usually evaluated against an allowed stock. Mathematically, it is a constraint in
the same family as "must include this starting material" or "must have depth <=
4."

Treating stock as the default `MetricScope` gives RetroCast continuity with
existing benchmarks while keeping the schema general enough for future tasks.

## Why Top-K Uses Scope, Not Solv-i

Top-K acceptable-route accuracy asks:

```text
among routes eligible under the user's task constraints, did the model recover a benchmark-acceptable route?
```

For current RetroCast benchmarks, the task constraint is stock termination, so
the default metric is effectively `Top-K[stock]`.

Top-K should not be filtered by Solv-i as a headline metric. A route can satisfy
Solv-3 without resembling a benchmark reference route. Conversely,
acceptable-route matching is a conservative proxy for unmeasured validity, not
the same thing as validity itself.

Keeping Top-K scoped by user constraints, rather than by Tier/Solv predicates,
preserves that distinction.

## Why MRR Is A Raw-Rank Diagnostic

MRR-style metrics answer a different question from user-centric reconstruction:

```text
how early did the model emit a candidate satisfying this predicate?
```

That makes `mrr_tier_i` and `mrr_solv_i[scope]` useful diagnostics for ranking
behavior. They should use raw model rank, not effective rank after filtering,
because the point is to measure how much unusable material appears before the
first valid or solvable candidate.

## Why Candidate Records Exist

Route-only artifacts cannot honestly measure candidate-level Tier-0 behavior.

Adaptation canonicalizes and validates routes. Malformed sequence-model outputs
can disappear before scoring. If failed rank slots are dropped, Tier-0 candidate
validity and raw-rank MRR become inflated.

Candidate records preserve the raw ranked candidate stream:

- successful candidates keep a canonical `Route`
- failed candidates keep `route=None` plus a typed failure record
- raw rank remains intact

This preserves denominator integrity for metrics that need it without forcing
route-standardization users to work with failed candidates.

## Why Failed Candidates Are Not Empty Routes

An empty `Route` would pretend RetroCast has chemistry where it only has a
failed model emission. That would pollute route traversal, hashing, and
deduplication semantics.

The honest representation is a candidate with no route:

```python
CandidateRecord(route=None, adapter_failure=failure)
```

That keeps rank and failure information while preserving the meaning of `Route`.

## Why Validity Is Not Stored On ReactionStep

`Route`, `Molecule`, and `ReactionStep` describe canonical chemistry. Validity is
an evaluator result.

Embedding validity on `ReactionStep` would make the chemistry object depend on
the selected scope, validator set, mapper version, optional dependencies, and
scoring run. It would also pollute content hashes and deduplication.

Validity belongs in the evaluation sidecar. The sidecar can carry stable
reaction identifiers for debugging without changing canonical chemistry.

## Why Pydantic At Boundaries

Solv-N records cross artifact, CLI, and library boundaries. They need stable JSON
shapes, readable validation errors, defaults, and compatibility behavior.
Pydantic is the right boundary tool.

Dataclasses remain appropriate for small internal validator values when they
make computation clearer. Persisted records should be Pydantic.

## Why Typed Failures And Check Results Differ

Candidate adaptation failures should reuse RetroCast's typed error contract:
stable code, message, context, and retryability.

Validator outcomes are different. A failed validator check is a measured result,
not necessarily an exception. It should use a check record with a stable code and
status.

This separation lets RetroCast distinguish "the adapter could not construct a
route" from "the constructed route failed a validity check."
