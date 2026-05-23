# Solv-N Target Design

This document describes the target state for RetroCast after the Solv-N
migration. It is a design target, not a record of the current implementation.

## Goals

RetroCast should separate four ideas currently collapsed into "solvability":

- `Tier-i validity`: intrinsic validity of reactions and routes.
- `MetricScope`: user/task feasibility constraints.
- `Solv-i[scope]`: `Tier-i validity + MetricScope`.
- `Top-K reconstruction`: acceptable-route recovery after filtering by a scope.

The default retrosynthesis scope is stock termination, so the familiar shorthand
is:

```text
solv_i = solv_i[stock]
top_k = top_k[stock]
```

Persisted artifacts must store the scope explicitly. Reports may use the
shorthand only when the table or section declares the default scope.

## Metric Taxonomy

### Tier Validity

Tier validity is evaluated first at the reaction level.

- `tier_0_validity`: syntactic validity. Molecules parse and sanitize; reaction records are well formed; reaction SMILES or mapping fields are syntactically valid when present.
- `tier_1_validity`: topological validity. The reaction is a legal graph transformation under the selected validator protocol.
- `tier_2_validity`: selectivity and plausibility. Deferred until validators exist.
- `tier_3_validity`: executability. Deferred until validators exist.

A route passes `tier_i_validity` if every reaction in the route passes
`tier_i_validity`. Tier validity is independent of scope constraints.

For tier-0, `ReactionStep.reactants` means structural precursor children in the
route tree. It does not mean reagents, catalysts, solvents, or conditions. Those
fields may be syntax-checked when present, but their semantics belong to higher
tiers.

### Metric Scopes

A `MetricScope` is the set of user/task constraints used to decide whether a
candidate is eligible for user-centric metrics.

Stock-only scope:

```json
{
  "id": "stock",
  "constraints": [
    {
      "type": "stock_termination",
      "stock_name": "buyables-stock",
      "match_level": "full"
    }
  ]
}
```

Stock plus required starting material:

```json
{
  "id": "stock-required-sm",
  "constraints": [
    {
      "type": "stock_termination",
      "stock_name": "buyables-stock",
      "match_level": "full"
    },
    {
      "type": "required_leaf",
      "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
      "match_level": "full"
    }
  ]
}
```

Stock plus required starting material plus depth:

```json
{
  "id": "stock-required-sm-depth-le-4",
  "constraints": [
    {
      "type": "stock_termination",
      "stock_name": "buyables-stock",
      "match_level": "full"
    },
    {
      "type": "required_leaf",
      "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
      "match_level": "full"
    },
    {
      "type": "max_depth",
      "max_depth": 4
    }
  ]
}
```

`STR` is not special in the schema. It is simply the default constraint for
ordinary retrosynthesis benchmarks.

### Solv-i

Solvability is parameterized by scope:

```text
solv_i[scope] = tier_i_validity and constraints_pass(scope)
```

For ordinary RetroCast benchmarks:

```text
solv_i[stock] = tier_i_validity and stock_termination
```

Do not introduce a separate public name such as `constrained_solv_i`.
Parameterization is the disambiguation.

### Reconstruction Accuracy

Acceptable-route matching answers a different question from Solv-i:

```text
among routes eligible under this scope, did the model recover a benchmark-acceptable route?
```

The canonical metric name is `top_k[scope]`. In ordinary stock-only reports,
`top_k` is shorthand for `top_k[stock]`, matching the current RetroCast
effective-rank behavior.

Do not make `top_k[solv_i]` a headline metric. It mixes reference matching with
the tier-validity predicate that reference matching is meant to complement. A
tier-valid route need not resemble a benchmark reference route.

### Ranking Metrics

Ranking metrics are raw-rank diagnostics over a predicate:

```text
mrr_tier_i = mean reciprocal raw rank of the first tier-i-valid candidate
mrr_solv_i[scope] = mean reciprocal raw rank of the first solv-i[scope] candidate
```

If the paper-facing name `MRR-V_i` is used, it maps to `mrr_tier_i`, not to
reference-route reconstruction.

Ranking metrics require rank-preserving candidate artifacts. Route-only artifacts
can still support `top_k[stock]`, but they cannot support honest raw-rank
validity diagnostics.

### Diagnostic Rates

Diagnostics are useful but should not be confused with target success:

- `adaptation_success_rate`
- `candidate_tier_i_pass_rate`
- `candidate_constraint_pass_rate[scope]`
- `saved_route_tier_i_pass_rate`
- failure counts by stable code
- per-reaction failure localization

## Model Ownership

Use Pydantic models at artifact and public-library boundaries.

- `src/retrocast/models/chem.py`: canonical chemistry only. No validity fields.
- `src/retrocast/models/validity.py`: validity and constraint result models.
- `src/retrocast/models/candidates.py`: pre-score candidate records and candidate audit metadata.
- `src/retrocast/models/evaluation.py`: scored candidates, target evaluation, and evaluation results.

Validator internals may use dataclasses if that keeps local computation small,
but serialized artifacts should use Pydantic models.

## Boundary Models

### Failure Records

Candidate failures reuse RetroCast's typed error contract. A persisted failure is
the serializable shape returned by `RetroCastException.to_dict()`.

```python
from typing import Any, Literal

from pydantic import BaseModel, Field


JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
CheckStatus = Literal["pass", "fail", "unknown", "not_evaluated"]


class FailureRecord(BaseModel):
    code: str
    message: str
    context: dict[str, JsonValue] = Field(default_factory=dict)
    retryable: bool = False
```

### Scope Records

```python
class StockTerminationConstraint(BaseModel):
    type: Literal["stock_termination"] = "stock_termination"
    stock_name: str
    match_level: str


class RequiredLeafConstraint(BaseModel):
    type: Literal["required_leaf"] = "required_leaf"
    inchikey: str
    match_level: str


class MaxDepthConstraint(BaseModel):
    type: Literal["max_depth"] = "max_depth"
    max_depth: int


Constraint = StockTerminationConstraint | RequiredLeafConstraint | MaxDepthConstraint


class MetricScope(BaseModel):
    id: str
    constraints: list[Constraint]
```

### Validity and Constraint Results

Store detailed checks only on failure or in debug output. Passing checks may be a
compact status.

```python
class CheckResult(BaseModel):
    code: str
    status: CheckStatus
    message: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TierResult(BaseModel):
    tier: int
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)


class ConstraintResult(BaseModel):
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)


class ReactionValidity(BaseModel):
    reaction_index: int
    reaction_signature: tuple[frozenset[str], str] | None = None
    product_inchikey: str | None = None
    reactant_inchikeys: list[str] = Field(default_factory=list)
    tiers: dict[int, TierResult] = Field(default_factory=dict)


class RouteValidity(BaseModel):
    tiers: dict[int, TierResult] = Field(default_factory=dict)
    reactions: list[ReactionValidity] = Field(default_factory=list)
```

Do not persist per-candidate `solv` by default. It is derived:

```text
solv_i[scope] = validity.tiers[i].status == "pass"
                and constraint_results[scope].status == "pass"
```

Target/stat summaries may store computed Solv-i ranks and rates.

### Candidate Records

`CandidateRecord` is the pre-score unit. It represents one raw ranked model slot.

```python
class CandidateSource(BaseModel):
    key: str | None = None
    row_index: int | None = None
    record_id: str | None = None


class CandidateRecord(BaseModel):
    rank: int
    route: Route | None = None
    adapter_failure: FailureRecord | None = None
    source: CandidateSource = Field(default_factory=CandidateSource)
```

If the artifact is stored as `dict[target_id, list[CandidateRecord]]`, do not
repeat `target_id` inside every candidate. If a streaming JSONL artifact is used,
put `target_id` on the row envelope.

Rejected and unassigned are different states:

- Rejected: a raw slot could not be adapted to `Route`; it has `route=None` and `adapter_failure`.
- Unassigned: RetroCast could not attach a raw slot to a benchmark target.

### Candidate Audit Metadata

Score should not infer whether failed slots were preserved. Ingest writes
explicit metadata.

```python
class CandidateAuditMetadata(BaseModel):
    candidate_audit_version: str
    preserves_failed_candidates: bool
    candidate_denominator: Literal["complete", "partial", "route_only"]
    target_assignment: Literal["benchmark_target_key", "target_hint", "route_match", "unknown"]
    n_raw_entries_seen: int
    n_candidate_records_written: int
    n_routes_adapted: int
    n_adaptation_failures: int
    n_unassigned_candidates: int
    sampling_policy: str | None = None
```

Candidate-level validity metrics are reportable only when:

```text
preserves_failed_candidates == true
candidate_denominator == "complete"
n_unassigned_candidates == 0
```

### Scored Candidates

`ScoredCandidate` is the score-stage unit. It adds validity, constraint results,
and benchmark-reference annotations to a pre-score candidate.

```python
class ScoredCandidate(BaseModel):
    rank: int
    route: Route | None = None
    validity: RouteValidity
    constraint_results: dict[str, ConstraintResult] = Field(default_factory=dict)
    matches_acceptable: bool = False
    matched_acceptable_index: int | None = None
    adapter_failure: FailureRecord | None = None
```

Compact normal case:

```json
{
  "rank": 1,
  "validity": {
    "tiers": {
      "0": { "tier": 0, "status": "pass" },
      "1": { "tier": 1, "status": "pass" }
    },
    "reactions": []
  },
  "constraint_results": {
    "stock": { "status": "pass" },
    "stock-required-sm": {
      "status": "fail",
      "checks": [
        { "code": "constraint.required_leaf", "status": "fail" }
      ]
    }
  }
}
```

### Target Evaluation

Target-level summaries are derived from scored candidates.

```python
class TargetEvaluation(BaseModel):
    target_id: str
    candidates: list[ScoredCandidate] = Field(default_factory=list)

    tier_validity_ranks: dict[int, int | None] = Field(default_factory=dict)
    solv_ranks: dict[str, dict[int, int | None]] = Field(default_factory=dict)
    top_k_ranks: dict[str, int | None] = Field(default_factory=dict)

    wall_time: float | None = None
    cpu_time: float | None = None
```

`solv_ranks["stock"][1]` is the raw rank of the first `solv_1[stock]`
candidate. `top_k_ranks["stock"]` is the effective acceptable-route rank after
filtering to the `stock` scope.

Legacy aliases remain loadable during migration:

- `ScoredRoute.is_solved` means `constraint_results["stock"].status == "pass"`.
- `TargetEvaluation.is_solvable` means at least one candidate passes the `stock` scope.
- `ModelStatistics.solvability` means target-level stock-scope pass rate.

These names should be deprecated through `src/retrocast/_warnings.py` where
practical, without warning during artifact parsing.

### Evaluation Results

Store metric scopes once per scored artifact.

```python
class EvaluationResults(BaseModel):
    model_name: str
    benchmark_name: str
    metric_scopes: list[MetricScope] = Field(default_factory=list)
    results: dict[str, TargetEvaluation] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
```

## Workflow Boundaries

### Route Standardization Mode

Default ingest remains useful for users who only want canonical route payloads.

```bash
retrocast ingest MODEL BENCHMARK
```

Output:

```text
3-processed/<benchmark>/<model>/routes.json.gz
3-processed/<benchmark>/<model>/manifest.json
```

This mode supports:

- scope pass checks over saved routes
- `top_k[stock]`
- route-level diagnostics over saved routes

This mode does not support:

- adaptation success rate as a candidate denominator
- candidate tier-0 pass rate
- raw-rank `mrr_tier_i`
- raw-rank `mrr_solv_i[scope]`

### Candidate Audit Mode

Benchmark evaluation that reports candidate-level tier rates or raw-rank MRR
validity metrics uses rank-preserving candidates.

```bash
retrocast ingest MODEL BENCHMARK --preserve-failed-candidates
retrocast score MODEL BENCHMARK --metrics tier,solv,mrr,top-k
```

Score infers the required denominator from requested metrics:

- if requested metrics need candidate denominators and candidate audit is missing or partial, fail with `InputError`.
- if requested metrics can be computed from routes, route-only artifacts are allowed.

Output:

```text
3-processed/<benchmark>/<model>/candidates.json.gz
3-processed/<benchmark>/<model>/routes.json.gz
3-processed/<benchmark>/<model>/manifest.json
```

`candidates.json.gz` preserves raw rank slots before deduplication. Duplicate
model outputs still consume rank and should affect raw-rank diagnostics.

`routes.json.gz` is a compatibility view containing successful canonical routes.
It may be deduplicated and sampled, but those transformations must not mutate
`candidates.json.gz` unless the manifest marks the denominator as partial.

### Target Assignment

Failed candidates are useful only if they can be assigned to a benchmark target.

For target-keyed provider output, the enclosing target key supplies assignment.
For provider-wide output:

- successful routes can be assigned by canonical target SMILES, as today.
- failed candidates need `target_hint_id`, `target_hint_smiles`, or equivalent source context.
- failed candidates without target hints are unassigned and make candidate-level benchmark metrics partial.

Adapter contract for candidate audit mode:

- `iter_raw_entries()` should yield every rank slot the provider returned.
- if an adapter cannot enumerate malformed slots, it marks the candidate denominator as partial.

## Scoring Procedure

For each candidate:

1. Preserve the raw rank.
2. If `route is None`, record adapter failure and mark tier/scope checks according to the requested checks.
3. Evaluate scope constraints for valid routes.
4. Evaluate enabled tier validators.
5. Compute acceptable-route match independently from tier and Solv-i.

For each target:

1. Compute `tier_validity_ranks[i]` from raw ranks.
2. Compute `solv_ranks[scope_id][i]` from raw ranks.
3. Compute `top_k_ranks[scope_id]` using effective rank after filtering by that scope.

Do not compute default acceptable-route Top-K after Tier-i or Solv-i filtering.
That can exist as a diagnostic only if it is named explicitly.

## Statistics

Headline statistics:

- `tier_i`: target-level rate for at least one Tier-i-valid route.
- `solv_i[scope]`: target-level rate for at least one Solv-i route under the scope.
- `mrr_tier_i`: raw-rank MRR for first Tier-i-valid candidate.
- `mrr_solv_i[scope]`: raw-rank MRR for first Solv-i candidate under the scope.
- `top_k[scope]`: reference reconstruction after filtering by the scope.

Diagnostics:

- `adaptation_success_rate`
- `candidate_tier_i_pass_rate`
- `candidate_constraint_pass_rate[scope]`
- `saved_route_tier_i_pass_rate`
- failure counts by code
- per-reaction failure records

## Validator Architecture

Validators are optional and explicit.

```python
class ValidityValidator:
    tier: int
    name: str

    def evaluate_reaction(self, reaction: RouteReaction) -> TierResult: ...
```

Baseline install:

- Tier-0 molecule and route-record syntax checks.
- Metric-scope constraints such as stock termination.

Optional `solv1` dependency group:

- Mapped reaction validation.
- Template extraction or graph-edit validation.

Optional `solv1-mapping` dependency group:

- Automatic atom mapping for unmapped reactions.

Mapped-route users should not need an automatic mapper. Users who only need
route standardization should not need the tier-1 dependency stack.

## Developer Experience

The stored schema should have helper methods so users do not need to manually
walk nested JSON.

Route-level inspection:

```python
candidate.validity.passes_tier(0)
candidate.validity.failed_reactions(tier=0)
candidate.passes_scope("stock")
candidate.passes_solv(tier=0, scope_id="stock")
```

Target-level inspection:

```python
target.first_tier_valid_candidate(tier=0)
target.first_solv_candidate(tier=0, scope_id="stock")
target.first_acceptable_candidate(scope_id="stock")
```

Reaction-level inspection:

```python
for reaction in candidate.validity.failed_reactions(tier=0):
    print(reaction.reaction_index, reaction.failed_checks(tier=0))
```
