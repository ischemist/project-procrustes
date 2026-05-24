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

Ranking metrics require artifacts that preserve the model's ranked route stream.
Route-only artifacts can still support `top_k[stock]`, but they cannot support
honest diagnostics over rank slots that failed adaptation.

### Diagnostics

Diagnostics should explain failures without becoming headline success metrics:

- failure counts by stable code
- candidate audit counts
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

### Status Records

```python
from typing import Any, Literal

from pydantic import BaseModel, Field


CheckStatus = Literal["pass", "fail", "unknown", "not_evaluated"]


class FailureRecord(BaseModel):
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
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
    details: dict[str, Any] = Field(default_factory=dict)


class TierResult(BaseModel):
    tier: int
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)


class ConstraintResult(BaseModel):
    status: CheckStatus
    checks: list[CheckResult] = Field(default_factory=list)


class ReactionValidity(BaseModel):
    reaction_index: int
    tiers: dict[int, TierResult] = Field(default_factory=dict)


class RouteValidity(BaseModel):
    tiers: dict[int, TierResult] = Field(default_factory=dict)
    reactions: list[ReactionValidity] = Field(default_factory=list)
```

`ScoredCandidate.route` is the source of chemistry. Validity annotations should
not duplicate product/reactant SMILES or InChIKeys; `reaction_index` points back
to `list(route.iter_reactions())`.

Artifacts use readable tier labels and omit empty checks:

```json
"validity": {
  "tier 0": {
    "status": "pass"
  },
  "reactions": [
    {
      "reaction_index": 1,
      "validity": {
        "tier 0": {
          "status": "fail",
          "checks": [{"code": "tier0.empty_reactants"}]
        }
      }
    }
  ]
}
```

Do not persist per-candidate `solv` by default. It is derived:

```text
solv_i[scope] = validity["tier i"].status == "pass"
                and constraint_results[scope].status == "pass"
```

Target/stat summaries may store computed Solv-i ranks and rates.

### Candidate Records

`CandidateRecord` is the pre-score unit. It preserves one raw target-local rank
slot when candidate-preserving ingest is enabled.

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

Exactly one of `route` or `adapter_failure` must be present.

### Scored Candidates

`ScoredCandidate` is the score-stage unit. It adds validity, constraint results,
and benchmark-reference annotations to a ranked candidate. A candidate may have
no route only when candidate-preserving ingest recorded an adapter failure.

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

    first_valid_ranks: dict[str, int | None] = Field(default_factory=dict)
    first_solv_ranks: dict[str, dict[str, int | None]] = Field(default_factory=dict)
    first_reconstruction_ranks: dict[str, int | None] = Field(default_factory=dict)

    wall_time: float | None = None
    cpu_time: float | None = None
```

`first_solv_ranks["stock"]["tier 1"]` is the raw rank of the first
`solv_1[stock]` candidate. `first_reconstruction_ranks["stock"]` is the
effective acceptable-route rank after filtering to the `stock` scope.

New Solv-N artifacts should use the new names directly:

- `constraint_results["stock"].status == "pass"` means stock termination.
- `TargetEvaluation.has_stock_terminated_route` means at least one candidate passes the `stock` scope.
- `ModelStatistics.stock_termination` means target-level stock-scope pass rate.

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
- raw-rank MRR over slots that failed adaptation

### Candidate-Preserving Mode

Candidate-preserving adaptation records every target-local raw rank slot,
including adapter failures. `ingest` enables this by default because it is the
benchmarking wrapper; standalone `adapt` keeps route-only output unless the flag
is explicit.

```bash
retrocast adapt --input raw.json.gz --output candidates.json.gz \
  --input-kind target-keyed-provider-output --benchmark benchmark.json.gz \
  --preserve-failed-candidates

retrocast ingest --model MODEL --dataset BENCHMARK
```

Project ingest output:

```text
3-processed/<benchmark>/<model>/routes.json.gz
3-processed/<benchmark>/<model>/candidates.json.gz
3-processed/<benchmark>/<model>/manifest.json
```

`routes.json.gz` remains the canonical route-only artifact. `candidates.json.gz`
is the candidate-denominator artifact used when scoring needs failed rank slots.

Candidate-preserving adaptation currently requires target-keyed provider output so
failed slots can be assigned to benchmark targets explicitly. Flat provider
output remains route-only until an adapter exposes target-local candidate slots.

## Scoring Procedure

For each candidate:

1. Preserve the raw rank.
2. If the candidate has an adapter failure, record a failed Tier-0 result and do not evaluate scope constraints.
3. Evaluate Tier-i validity for route candidates.
4. Evaluate scope constraints for route candidates.
5. Evaluate enabled tier validators.
6. Compute acceptable-route match independently from tier and Solv-i.

For each target:

1. Compute `first_valid_ranks["tier i"]` from raw ranks.
2. Compute `first_solv_ranks[scope_id]["tier i"]` from raw ranks.
3. Compute `first_reconstruction_ranks[scope_id]` using effective rank after filtering by that scope.

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

- failure counts by code
- candidate audit counts
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

The stored schema should be inspectable without helper magic:

```python
candidate.validity.tiers[0].status
candidate.constraint_results["stock"].status
target.first_valid_ranks["tier 0"]
target.first_solv_ranks["stock"]["tier 0"]
```

Reaction-level failures are explicit records:

```python
for reaction in candidate.validity.reactions:
    failed = [check.code for check in reaction.tiers[0].checks if check.status == "fail"]
    if failed:
        print(reaction.reaction_index, failed)
```
