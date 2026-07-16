---
icon: lucide/cable
---

# Schema Design

This document captures the thinking process for the core data model and workflow design in RetroCast. It is a more elaborate version of the [concepts overview page](/concepts) and is the primary place to build a mental model of the codebase.

## Goals

RetroCast is designed to handle two broad use cases:

- casting arbitrary planner output into canonical `Route`s
- evaluating those outputs for one target or many targets

## Route

In RetroCast, a `Route` is an AND/OR tree of `Molecule` and `Reaction` nodes.

```text
Route -> Molecule -> Reaction -> Molecule -> Reaction -> ...
```

```mermaid
graph TD
    r["Route"] --> m0["Molecule (target)"]
    m0 --> rx0["Reaction"]
    rx0 --> m1["Molecule"]
    rx0 --> m2["Molecule"]
    m1 --> rx1["Reaction"]
    rx1 --> m3["Molecule"]
    rx1 --> m4["Molecule"]
```

Basic schema

=== "Python"

    ```python
    class Molecule(BaseModel):
        smiles: SmilesStr
        inchikey: InChIKeyStr
        product_of: Reaction | None = None
        annotations: dict[str, Any] = Field(default_factory=dict)


    class Reaction(BaseModel):
        reactants: list[Molecule]
        mapped_reaction_smiles: ReactionSmilesStr | None = None
        template: str | None = None
        reagents: list[SmilesStr] | None = None
        solvents: list[SmilesStr] | None = None
        annotations: dict[str, Any] = Field(default_factory=dict)


    class Route(BaseModel):
        target: Molecule
        annotations: dict[str, Any] = Field(default_factory=dict)
        schema_version: str = "2"
    ```

=== "Rust"

    ```rust
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Molecule {
        pub smiles: CanonicalSmiles,
        pub inchikey: InchiKey,
        pub product_of: Option<Box<Reaction>>,
        #[serde(default)]
        pub annotations: Annotations,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Reaction {
        pub reactants: Vec<Molecule>,
        pub mapped_reaction_smiles: Option<ReactionSmiles>,
        pub template: Option<String>,
        pub reagents: Option<Vec<CanonicalSmiles>>,
        pub solvents: Option<Vec<CanonicalSmiles>>,
        #[serde(default)]
        pub annotations: Annotations,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Route {
        pub target: Molecule,
        #[serde(default)]
        pub annotations: Annotations,
        #[serde(default = "schema_v2")]
        pub schema_version: SchemaVersion,
    }
    ```

The Rust constructors are the validation boundary. `CanonicalSmiles`, `InchiKey`, and `SchemaVersion` are validated newtypes rather than aliases for `String`. Serde deserialization calls the same validation path used by adapters, so loading an artifact cannot create a state that the public constructor would reject.

Identical molecules in different positions (e.g. same building block used in two branches) are different nodes; whence a Route is a tree, not just a DAG. Primarily because enforcing a 1 molecule = 1 node would introduce operational (serialization, signatures) complexity without any clear/obvious benefit beyond just marginally smaller disk usage.

### Route path

RetroCast uses deterministic paths to refer to molecules and reactions inside a `Route`. The full grammar lives in [Route Node IDs](../reference/route-node-ids.md); but here's a useful cheat sheet:

- `rc:m:/` root target molecule
- `rc:r:/` root reaction
- `rc:m:/0` first reactant under `rc:r:/`
- `rc:r:/0` reaction producing `rc:m:/0`
- `rc:m:/1/0` first child under `rc:m:/1`

These IDs are derived in memory and are not serialized. Internally, they are typed addresses, not strings that each caller reparses.

=== "Python"

    ```python
    @dataclass(frozen=True)
    class RoutePath:
        kind: Literal["m", "r"]
        indices: tuple[int, ...] = ()

        @classmethod
        def parse(cls, value: str) -> RoutePath: ...

        @classmethod
        def target(cls) -> RoutePath: ...

        @classmethod
        def root_reaction(cls) -> RoutePath: ...

        def id(self) -> str: ...
        def depth(self) -> int: ...
        def produced_by(self) -> RoutePath: ...
        def product(self) -> RoutePath: ...
        def reactant(self, index: int) -> RoutePath: ...
    ```

=== "Rust"

    ```rust
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum RoutePath {
        Molecule(Box<[usize]>),
        Reaction(Box<[usize]>),
    }

    impl RoutePath {
        pub fn parse(value: &str) -> Result<Self, RoutePathError>;
        pub fn target() -> Self;
        pub fn root_reaction() -> Self;
        pub fn id(&self) -> String;
        pub fn depth(&self) -> usize;
        pub fn produced_by(&self) -> Result<Self, RoutePathError>;
        pub fn product(&self) -> Result<Self, RoutePathError>;
        pub fn reactant(&self, index: usize) -> Result<Self, RoutePathError>;
    }
    ```

semantics:

- `RoutePath.target()` is `rc:m:/`
- `RoutePath.root_reaction()` is `rc:r:/`
- `RoutePath.parse("rc:m:/1/0").produced_by()` is `rc:r:/1/0`
- `RoutePath.parse("rc:r:/1/0").product()` is `rc:m:/1/0`
- `RoutePath.parse("rc:r:/1/0").reactant(2)` is `rc:m:/1/0/2`

For serialized IDs, Python exposes validated strings and Rust exposes newtypes:

=== "Python"

    ```python
    ReactionId = Annotated[str, AfterValidator(validate_reaction_id)]
    MoleculeId = Annotated[str, AfterValidator(validate_molecule_id)]
    ```

=== "Rust"

    ```rust
    #[derive(Clone, Debug, Deserialize, Eq, Hash, Serialize)]
    #[serde(try_from = "String", into = "String")]
    pub struct ReactionId(RoutePath);

    #[derive(Clone, Debug, Deserialize, Eq, Hash, Serialize)]
    #[serde(try_from = "String", into = "String")]
    pub struct MoleculeId(RoutePath);
    ```

An invalid kind is rejected when the ID enters the core. Downstream scoring code cannot accidentally put a molecule address into `ReactionValidity`.

## Route Signatures

`Route` signatures give us a canonical way to talk about route structure without carrying around the whole tree or comparing nested objects by hand. They are the basis for route comparison: full-route equality, reaction equality, prefix matching to depth `k`, and subtree containment. The core idea is [Merkle-like](https://en.wikipedia.org/wiki/Merkle_tree): the signature of a parent is built from its own identity plus the signatures of its children. Signatures are:

- order-invariant over reactant ordering
- preserve multiplicity when the same reactant appears more than once
- and can be parameterized by match level when needed.

### Molecule Identity

We use [InChiKeys](https://en.wikipedia.org/wiki/International_Chemical_Identifier) as molecular identity. RetroCast currently supports three levels of InChiKey specificity:

- `retrocast.chem.InChIKeyLevel.FULL` - full 27-char InChIKey
- `retrocast.chem.InChIKeyLevel.NO_STEREO` - 27-char InChIKey generated without stereochemical information
- `retrocast.chem.InChIKeyLevel.CONNECTIVITY` - first 14 chars, connectivity layer only

Most users should use the default `FULL` level, but sometimes a model developer might wish to disambiguate planner's failure to account for proper stereochemistry from more fundamental failures to find the right connectivity (wherefore he might use `NO_STEREO`). Or might want to ignore isotope/protonation differences (wherefore `CONNECTIVITY`).

```python
class Molecule(BaseModel):
    ...

    def key(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> str:
        return reduce_inchikey(self.inchikey, match_level)
    def signature(self, match_level: InChIKeyLevel = InChIKeyLevel.FULL) -> str:
        return stable_hash(self.key(match_level))
```

### Reaction Identity

At the most basic structural level, a reaction identity is defined by the structures of reactants and product. Defining `key` and `signature` method on a `Reaction` class is not possible without having a pointer to the `Reaction` product. There are two options:

- treat `Reaction` as a Route-specific occurrence object (with an explicit pointer to its product), but that requires writing custom serialization logic and ensuring loaded Reaction objects are always "hydrated" with proper parent references. Violates the spirit of [SRP](https://en.wikipedia.org/wiki/Single-responsibility_principle) and I'm a bit WebDev-brained not to think of the Data-View split analogy, so instead
- we create a `ReactionView` model that provides a required route-contextual representation of a `Reaction`.

=== "Python"

    ```python
    class ReactionView:
        route: Route
        path: RoutePath
        value: Reaction

        def product(self) -> MoleculeView: ...
        def reactants(self) -> list[MoleculeView]: ...
        def key(self, match_level=InChIKeyLevel.FULL) -> tuple: ...
        def signature(self, match_level=InChIKeyLevel.FULL) -> str: ...
    ```

=== "Rust"

    ```rust
    pub struct ReactionView<'route> {
        pub route: &'route Route,
        pub path: RoutePath,
        pub value: &'route Reaction,
    }

    impl ReactionView<'_> {
        pub fn product(&self) -> MoleculeView<'_>;
        pub fn reactants(&self) -> Vec<MoleculeView<'_>>;
        pub fn key(&self, level: InchiKeyLevel) -> ReactionKey;
        pub fn signature(&self, level: InchiKeyLevel) -> Signature;
    }
    ```

for consistency in api design and ease of subtree comparison, we also define a similar view model for `Molecule`.

=== "Python"

    ```python
    class MoleculeView:
        route: Route
        path: RoutePath
        value: Molecule

        def key(self, match_level=InChIKeyLevel.FULL) -> str: ...
        def produced_by(self) -> ReactionView | None: ...
        def subtree_key(self, match_level=InChIKeyLevel.FULL, *, depth=None) -> tuple: ...
        def subtree_signature(self, match_level=InChIKeyLevel.FULL, *, depth=None) -> str: ...
    ```

=== "Rust"

    ```rust
    pub struct MoleculeView<'route> {
        pub route: &'route Route,
        pub path: RoutePath,
        pub value: &'route Molecule,
    }

    impl MoleculeView<'_> {
        pub fn key(&self, level: InchiKeyLevel) -> MoleculeKey;
        pub fn produced_by(&self) -> Option<ReactionView<'_>>;
        pub fn subtree_key(&self, level: InchiKeyLevel, depth: Option<usize>) -> SubtreeKey;
        pub fn subtree_signature(&self, level: InchiKeyLevel, depth: Option<usize>) -> Signature;
    }
    ```

Rust views borrow the route. They cannot outlive it and they do not introduce parent pointers into the serialized tree.

### Route Identity

With the primitives above, full route equality is established by the subtree signature of the target with unlimited depth. i.e. `route.signature()` is an alias for `route.molecule_at("rc:m:/").subtree_signature()`. A generic exact subtree equality is `route.molecule_at(path).subtree_signature()`.

We often might be interested in asking how far along the plan (starting from the target) do any two Routes agree? Such prefix matching is simply a subtree signature of fixed depth `k` rooted at the target.

=== "Python"

    ```python
    route.key(match_level=InChIKeyLevel.FULL, depth=None) -> tuple
    route.signature(match_level=InChIKeyLevel.FULL, depth=None) -> str
    ```

=== "Rust"

    ```rust
    impl Route {
        pub fn key(&self, level: InchiKeyLevel, depth: Option<usize>) -> SubtreeKey;
        pub fn signature(&self, level: InchiKeyLevel, depth: Option<usize>) -> Signature;
    }
    ```

### Content Signatures

The structural `key` / `signature` methods answer the basic route question: same molecules, same reaction graph. They do not read mapped reaction SMILES, templates, reagents, solvents, condition labels, or annotations.

Sometimes we want a stricter comparison: same structure, plus selected reaction content. For that, routes and route-bound views expose `content_key` / `content_signature` methods. The caller chooses which reaction fields matter:

```python
route.content_signature(fields=("mapped_reaction_smiles",))
route.content_signature(fields=("template", "reagents", "solvents"))
```

Content signatures follow the same Merkle shape as structural signatures: molecule identity, reaction identity, selected reaction content, and unordered child signatures. They also support `match_level` and route-prefix `depth`.

### Route Embedding

Route embedding asks whether one route occurs inside another route.

`Route.signature()` and `MoleculeView.subtree_signature()` answer exact equality for a chosen root. Embedding is looser in two ways: the query target may match an internal molecule in the container route, and the container may continue below a query leaf when the audit allows leaf extension.

`retrocast.curation.embedding` uses route views, paths, molecule keys, reaction signatures, and subtree signatures to produce audit traces:

=== "Python"

    ```python
    class EmbeddingMatch:
        query_path: RoutePath
        container_path: RoutePath
        matched_reactions: int
        leaf_extensions: tuple[LeafExtension, ...]

    find_route_embeddings(query, container, *, allow_leaf_extension=True) -> tuple[EmbeddingMatch, ...]
    route_embeds_at(query_view, container_view, *, allow_leaf_extension=True) -> EmbeddingMatch | None
    ```

=== "Rust"

    ```rust
    pub struct EmbeddingMatch {
        pub query_path: RoutePath,
        pub container_path: RoutePath,
        pub matched_reactions: usize,
        pub leaf_extensions: Vec<LeafExtension>,
    }

    pub fn find_route_embeddings(
        query: &Route,
        container: &Route,
        options: &EmbeddingOptions,
    ) -> Vec<EmbeddingMatch>;

    pub fn route_embeds_at(
        query: MoleculeView<'_>,
        container: MoleculeView<'_>,
        options: &EmbeddingOptions,
    ) -> Option<EmbeddingMatch>;
    ```

The detailed matching rules live in [Route Embedding](../reference/route-embedding.md). The important schema-design point is small: exact subtree equality is a signature check; embedding is a route-to-route matching check.

## Ownership and data flow

The public Python API and the standalone command execute the same functions in `retrocast-core`. Python names below are views over Rust-owned values; they are not a second set of workflow models.

```mermaid
flowchart LR
    raw["Raw planner payload"] --> adapter["Adapter"]
    adapter --> candidates["CandidateSet"]
    task["Task"] --> collect["Collect by target"]
    candidates --> collect
    collect --> predictions["Predictions"]
    predictions --> scorer["Scorer"]
    task --> scorer
    registry["Checker registry"] --> scorer
    scorer --> evaluation["Evaluation"]
    evaluation --> analyzer["Analyzer"]
    analyzer --> report["AnalysisReport"]

    io["Artifact I/O"] <--> candidates
    io <--> task
    io <--> evaluation
    io <--> report
```

Each arrow is a concrete read/write boundary:

| Stage | Reads | Writes | Invariant established |
| --- | --- | --- | --- |
| Load | bytes and artifact kind | validated schema value | invalid serialized states stop here |
| Adapt | planner payload and adapter options | ranked `Candidate`s | every raw rank becomes exactly one route or failure |
| Collect | candidates and task target index | target-keyed predictions | every candidate belongs to a known target |
| Score | predictions, task, checker registry | `Evaluation` | requested checks and effective constraints are recorded |
| Analyze | evaluation and metric options | `AnalysisReport` | every metric records its denominator and reliability |
| Write | any artifact value | bytes plus manifest metadata | content hash covers the serialized artifact |

No workflow serializes an intermediate value to call the next workflow. Artifact I/O is a leaf operation at the edge of the graph. This is what lets a single `retrocast pipeline` invocation keep routes in memory from adaptation through analysis.

The core API reflects that ownership:

=== "Python"

    ```python
    candidates = retrocast.ingest(raw, task, adapter="aizynthfinder")
    evaluation = retrocast.score(candidates, task, registry=registry)
    report = retrocast.analyze(evaluation)
    ```

=== "Rust"

    ```rust
    let candidates = ingest(raw, &task, &adapter, &ingest_options)?;
    let evaluation = score(&candidates, &task, &registry, &score_options)?;
    let report = analyze(&evaluation, &analysis_options)?;
    ```

The values on both tabs have the same owner: `retrocast-core`.

## Workflows

Raw planner payloads can be `adapt`ed into `Route` objects, which constitute fully Tier-0 valid routes, or `Candidate` objects, which preserve failed slots for proper accounting of Solv-0. Canonical `Candidates` can be `collect`ed into predictions for each `Target`/`Task`. A collection of `Task`s is a `Benchmark`.

Direct `adapt`ing is useful for single-target pipelines. For benchmarking, one can use `ingest` which is `adapt`ing and `collect`ing into `Benchmark` predictions.

`Benchmark` predictions are keyed by target id. These predictions can be `score`d with Tier-N validity checks against `TaskConstraint` records, resulting in Solv-N values. Predictions can also be compared against `Target.acceptable_routes` to obtain Top-K reconstruction accuracy.

## 1. Adapt

By default, adapt tries to turn raw planner output into canonical `Route` objects and if fails, returns None. While it's a reasonable default for regular planning, it inflates the Tier-0 validity of the planner's predictions for strict benchmarking. As such, `adapt` can be configured to return `Candidate` objects which either hold a `Route` or a `FailureRecord` that specifies `target_{id,smiles,inchikey}` for proper accounting of failed predictions.

=== "Python"

    ```python
    class FailureRecord(BaseModel):
        code: ErrorCode
        message: str | None = None
        target_id: str | None = None
        target_smiles: SmilesStr | None = None
        target_inchikey: InChIKeyStr | None = None
        context: dict[str, Any] = Field(default_factory=dict)


    class Candidate(BaseModel):
        rank: int
        route: Route | None = None
        failure: FailureRecord | None = None
    ```

=== "Rust"

    ```rust
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct FailureRecord {
        pub code: ErrorCode,
        pub message: Option<String>,
        pub target_id: Option<TargetId>,
        pub target_smiles: Option<CanonicalSmiles>,
        pub target_inchikey: Option<InchiKey>,
        #[serde(default)]
        pub context: Annotations,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Candidate {
        pub rank: NonZeroUsize,
        #[serde(flatten)]
        pub outcome: CandidateOutcome,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum CandidateOutcome {
        Route { route: Route },
        Failure { failure: FailureRecord },
    }
    ```

The JSON shape stays compatible, but Rust represents the exclusive-or directly: a `Candidate` cannot contain both a route and a failure, or neither. Rank zero cannot be constructed.

### Adapt modes

By default, `adapt` returns a `FailureRecord` if even a single SMILES is invalid. Model developers might sometimes be interested in the validity of predictions up to the corrupted SMILES, and `AdaptMode` allows them to relax adapters to return `Route`/`Candidate` objects that contain the longest-possible valid prefix `Route` with the corrupted SMILES node and all its children pruned out.

=== "Python"

    ```python
    AdaptMode = Literal["strict", "prune"]
    ```

=== "Rust"

    ```rust
    #[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum AdaptMode {
        #[default]
        Strict,
        Prune,
    }
    ```

### Adapt API

=== "Python"

    ```python
    adapt_route(raw_route_payload, adapter, *, mode="strict") -> Route | None

    adapt_routes(
        raw_payload,
        adapter,
        *,
        mode="strict",
        max_routes=None,
    ) -> list[Route]

    adapt_candidates(
        raw_payload,
        adapter,
        *,
        mode="strict",
        max_candidates=None,
    ) -> list[Candidate]
    ```

=== "Rust"

    ```rust
    pub trait Adapter: Send + Sync {
        type RawRoute;

        fn entries(&self, payload: RawPayload) -> Result<Vec<RawRouteEntry<Self::RawRoute>>>;
        fn adapt_route(&self, raw: Self::RawRoute, mode: AdaptMode) -> Result<Route>;
    }

    pub fn adapt_routes<A: Adapter>(
        payload: RawPayload,
        adapter: &A,
        options: &AdaptRouteOptions,
    ) -> Result<Vec<Route>>;

    pub fn adapt_candidates<A: Adapter>(
        payload: RawPayload,
        adapter: &A,
        options: &AdaptCandidateOptions,
    ) -> Result<Vec<Candidate>>;
    ```

`max_routes` is a route-first convenience limit: failures are skipped and only successful `Route`s count toward the limit. `max_candidates` is the benchmark-safe limit: it processes the first N raw candidate slots and preserves failures, so Tier-0 validity and MRR remain honest.

Benchmark CLI ingestion uses the candidate-preserving path. Route-only adaptation remains available as a library convenience for ad hoc inspection.

## 2. Collect

Collection maps adapted outputs onto known targets.

=== "Python"

    ```python
    CollectedCandidates = dict[str, list[Candidate]]
    CollectedRoutes = dict[str, list[Route]]

    collect_candidates(candidates, task) -> CollectedCandidates
    collect_routes(routes, task) -> CollectedRoutes
    ```

=== "Rust"

    ```rust
    pub type CollectedCandidates = BTreeMap<TargetId, Vec<Candidate>>;
    pub type CollectedRoutes = BTreeMap<TargetId, Vec<Route>>;

    pub fn collect_candidates(
        candidates: impl IntoIterator<Item = Candidate>,
        task: &Task,
    ) -> Result<CollectedCandidates>;

    pub fn collect_routes(
        routes: impl IntoIterator<Item = Route>,
        task: &Task,
    ) -> Result<CollectedRoutes>;
    ```

collection rules:

- if `candidate.route` exists, place it by `route.target`
- otherwise place it by `candidate.failure.target_id` / `candidate.failure.target_inchikey`

where `Task` is defined through `Target`s and `TaskConstraint`s:

=== "Python"

    ```python
    class Target(BaseModel):
        id: str
        smiles: SmilesStr
        inchikey: InChIKeyStr
        acceptable_routes: list[Route] = Field(default_factory=list)
        annotations: dict[str, Any] = Field(default_factory=dict)


    class TaskConstraint(BaseModel):
        kind: str
        model_config = ConfigDict(extra="allow")


    class StockTerminationConstraint(TaskConstraint):
        kind: Literal["retrocast.stock_termination"] = "retrocast.stock_termination"
        stock: str


    class RequiredLeavesConstraint(TaskConstraint):
        kind: Literal["retrocast.required_leaves"] = "retrocast.required_leaves"
        smiles: list[SmilesStr]


    class RouteDepthConstraint(TaskConstraint):
        kind: Literal["retrocast.route_depth"] = "retrocast.route_depth"
        max_depth: int | Literal["short", "medium", "long"]


    class Task(BaseModel):
        name: str
        targets: dict[str, Target]
        default_constraints: list[TaskConstraint] = Field(default_factory=list)
        constraints: dict[str, list[TaskConstraint]] = Field(default_factory=dict)
        metric_label: str | None = None
        annotations: dict[str, Any] = Field(default_factory=dict)
        schema_version: str = "2"

        def effective_constraints(self, target_id: str) -> list[TaskConstraint]: ...


    class Benchmark(Task):
        description: str
    ```

=== "Rust"

    ```rust
    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Target {
        pub id: TargetId,
        pub smiles: CanonicalSmiles,
        pub inchikey: InchiKey,
        #[serde(default)]
        pub acceptable_routes: Vec<Route>,
        #[serde(default)]
        pub annotations: Annotations,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(tag = "kind")]
    pub enum TaskConstraint {
        #[serde(rename = "retrocast.stock_termination")]
        StockTermination { stock: StockId },
        #[serde(rename = "retrocast.required_leaves")]
        RequiredLeaves { smiles: Vec<CanonicalSmiles> },
        #[serde(rename = "retrocast.route_depth")]
        RouteDepth { max_depth: DepthConstraint },
        #[serde(untagged)]
        Extension(ExtensionConstraint),
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Task {
        pub name: String,
        pub targets: BTreeMap<TargetId, Target>,
        #[serde(default)]
        pub default_constraints: Vec<TaskConstraint>,
        #[serde(default)]
        pub constraints: BTreeMap<TargetId, Vec<TaskConstraint>>,
        pub metric_label: Option<String>,
        #[serde(default)]
        pub annotations: Annotations,
        #[serde(default = "schema_v2")]
        pub schema_version: SchemaVersion,
    }

    impl Task {
        pub fn effective_constraints(
            &self,
            target_id: &TargetId,
        ) -> Result<Vec<&TaskConstraint>>;
    }
    ```

The Rust enum validates built-in fields while `ExtensionConstraint` preserves a namespaced kind and JSON object for third-party constraints. `Task::new` and deserialization reject duplicate kinds within one scope and target-map keys that disagree with `Target.id`.

`route_depth` integers are inclusive maximum depths. Named depth constraints are ranges:

- `short`: depth 1-3
- `medium`: depth 4-6
- `long`: depth 7+

One benchmark is a `Task`. One daedalus query is also a `Task`, usually with one target.

Within one target, constraints are keyed by `kind`. A target-specific constraint with the same `kind` overrides the default constraint. RetroCast-owned constraints use the `retrocast.` namespace. Custom constraints should use a project namespace, e.g. `ariadne.reaction_count`.

`metric_label` controls the bracketed task name in Solv-N metrics, e.g. `Solv-0[buyables]`. If unset, the label is derived from effective target constraints in fixed order: stock label, `leaf`, `depth` (e.g. `buyables+depth` or `buyables+leaf`).

## 3. Ingest

Ingest is just the convenience alias for `adapt + collect`

=== "Python"

    ```python
    ingest(raw, adapter_name, task, *, max_candidates=None, workers=1) -> NativePredictions
    ingest_file(raw_path, adapter_name, task_path, *, max_candidates=None, workers=1) -> NativePredictions
    ```

=== "Rust"

    ```rust
    pub fn ingest_routes<A: Adapter>(
        raw: RawPayload,
        adapter: &A,
        task: &Task,
        options: &IngestRouteOptions,
    ) -> Result<CollectedRoutes>;

    pub fn ingest_candidates<A: Adapter>(
        raw: RawPayload,
        adapter: &A,
        task: &Task,
        options: &IngestCandidateOptions,
    ) -> Result<CollectedCandidates>;
    ```

`max_candidates` is applied per target during benchmark ingestion. It is intentionally first-N by raw planner rank; random candidate sampling is not part of benchmark ingestion because planner rank is part of the measured behavior.

### In-process ownership

Artifacts use schema-v2 JSON, but an in-process pipeline does not use JSON as an internal transport. Both front ends keep the Rust values returned by each stage:

=== "Python"

    ```python
    predictions = retrocast.ingest(raw, "aizynthfinder", task)
    evaluation = retrocast.score(predictions, task, stocks)
    report = retrocast.analyze(evaluation)
    ```

=== "Rust"

    ```rust
    let predictions = ingest(raw, adapter, &task, mode, limit, workers)?;
    let evaluation = score(&predictions, &task, &stocks, level, matching, None, workers)?;
    let report = analyze(&evaluation, &ks, &prefix_depths, n_boot, seed, workers)?;
    ```

The Python variables are opaque Rust handles. `score` consumes `NativePredictions` and moves its graph into `NativeEvaluation`; the old predictions handle then rejects access. There is no Python DTO or compatibility implementation. `.to_dict()`, `.json()`, and `.write()` create explicit snapshots for inspection or persistence before that move.

## 4. Score

The ultimate method for scoring any synthesis plan is by passing it through the [Solv-N filter](/dev/rationale/solv-n-evaluation), which is defined as:

```text
Solv-i[task] = Tier-i validity + satisfaction of task constraints.
```

Tier-N validity is stored as `TierResult`. `RouteValidity` stores the validity of a `Route` as a whole and of all individual `Reactions` that compose it.

=== "Python"

    ```python
    class CheckStatus(StrEnum):
        PASS = "pass"
        FAIL = "fail"
        NOT_EVALUATED = "not_evaluated"


    class Tier(IntEnum):
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3


    class CheckResult(BaseModel):
        code: str
        status: CheckStatus = CheckStatus.FAIL
        message: str | None = None
        details: dict[str, Any] = Field(default_factory=dict)


    class TierResult(BaseModel):
        status: CheckStatus
        checks: list[CheckResult] = Field(default_factory=list)


    class ReactionValidity(BaseModel):
        reaction_id: ReactionId
        tiers: dict[Tier, TierResult] = Field(default_factory=dict)


    class RouteValidity(BaseModel):
        tiers: dict[Tier, TierResult] = Field(default_factory=dict)
        reactions: list[ReactionValidity] = Field(default_factory=list)
    ```

=== "Rust"

    ```rust
    #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum CheckStatus {
        Pass,
        Fail,
        NotEvaluated,
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
    #[repr(u8)]
    pub enum Tier {
        Zero = 0,
        One = 1,
        Two = 2,
        Three = 3,
    }

    pub struct CheckResult {
        pub code: CheckCode,
        pub status: CheckStatus,
        pub message: Option<String>,
        pub details: Annotations,
    }

    pub struct TierResult {
        pub status: CheckStatus,
        pub checks: Vec<CheckResult>,
    }

    pub struct ReactionValidity {
        pub reaction_id: ReactionId,
        pub tiers: BTreeMap<Tier, TierResult>,
    }

    pub struct RouteValidity {
        pub tiers: BTreeMap<Tier, TierResult>,
        pub reactions: Vec<ReactionValidity>,
    }
    ```

Task constraint satisfaction is stored separately:

=== "Python"

    ```python
    class ConstraintResult(BaseModel):
        status: CheckStatus
        checks: list[CheckResult] = Field(default_factory=list)
    ```

=== "Rust"

    ```rust
    pub struct ConstraintResult {
        pub status: CheckStatus,
        pub checks: Vec<CheckResult>,
    }
    ```

A `Candidate` that undergoes scoring becomes `ScoredCandidate`.

=== "Python"

    ```python
    class ScoredCandidate(BaseModel):
        rank: int
        route: Route | None = None
        failure: FailureRecord | None = None
        validity: RouteValidity = Field(default_factory=RouteValidity)
        constraints: ConstraintResult
        matches_acceptable: bool = False
        matched_acceptable_index: int | None = None

        def satisfies_validity(self, tier: Tier | int) -> bool: ...
        def satisfies_task(self) -> bool: ...
        def satisfies_solv(self, tier: Tier | int) -> bool: ...
    ```

=== "Rust"

    ```rust
    pub struct ScoredCandidate {
        pub rank: NonZeroUsize,
        #[serde(flatten)]
        pub outcome: CandidateOutcome,
        pub validity: RouteValidity,
        pub constraints: ConstraintResult,
        pub matches_acceptable: bool,
        pub matched_acceptable_index: Option<usize>,
    }

    impl ScoredCandidate {
        pub fn satisfies_validity(&self, tier: Tier) -> bool;
        pub fn satisfies_task(&self) -> bool;
        pub fn satisfies_solv(&self, tier: Tier) -> bool;
    }
    ```

`ScoredCandidate`s are collected into `TargetResult`, which in turn is collected into `Evaluation`.

=== "Python"

    ```python
    class TargetResult(BaseModel):
        target: Target
        effective_constraints: list[TaskConstraint]
        candidates: list[ScoredCandidate] = Field(default_factory=list)
        wall_time: float | None = None
        cpu_time: float | None = None


    class Evaluation(BaseModel):
        task: Task
        tiers: list[Tier] = Field(default_factory=list)
        metric_label: str
        acceptable_match_level: InChIKeyLevel = InChIKeyLevel.FULL
        acceptable_route_match: AcceptableRouteMatch = AcceptableRouteMatch.EXACT
        targets: dict[str, TargetResult] = Field(default_factory=dict)
        schema_version: str = "2"
    ```

=== "Rust"

    ```rust
    pub struct TargetResult {
        pub target: Target,
        pub effective_constraints: Vec<TaskConstraint>,
        pub candidates: Vec<ScoredCandidate>,
        pub wall_time: Option<f64>,
        pub cpu_time: Option<f64>,
    }

    pub struct Evaluation {
        pub task: Task,
        pub tiers: Vec<Tier>,
        pub metric_label: String,
        pub acceptable_match_level: InchiKeyLevel,
        pub acceptable_route_match: AcceptableRouteMatch,
        pub targets: BTreeMap<TargetId, TargetResult>,
        pub schema_version: SchemaVersion,
    }
    ```

`TargetResult.effective_constraints` is a snapshot, not a cache. Analysis reads what scoring actually enforced even if a task definition is later edited.

!!! note "An intentional violation of single-responsibility principle"

    In principle, Tier-0 validity should be assessed at the `score` workflow stage. The cleanest design would then be if `adapt` always returned an equivalent of `Candidate`s (or a `Route` was extended to hold the information of `Candidate`), but that would result in subpar UX for every use case outside of benchmarking. As a result, we intentionally allow for a slight leakage of responsibility between `adapt` and `score` (i.e., `adapt` without ``--preserve-failed-candidates` returns Tier-0 valid `Route`s.)

### Score API

Because we're far from having a universal Tier-2 validity checker, it is natural to expect multiple solutions emerging from different research groups. To enable modularity of scoring, we define a `TierChecker` protocol.

=== "Python"

    ```python
    class TierChecker(Protocol):
        tier: Tier
        name: str

        def check_route(self, route: Route) -> RouteValidity: ...
    ```

=== "Rust"

    ```rust
    pub trait TierChecker: Send + Sync {
        fn tier(&self) -> Tier;
        fn name(&self) -> &str;
        fn check_route(&self, route: &Route) -> Result<RouteValidity>;
    }
    ```

Similarly, we define a protocol for task constraint satisfaction:

=== "Python"

    ```python
    class TaskConstraintChecker(Protocol):
        kind: str

        def check_route(
            self,
            route: Route,
            constraint: TaskConstraint,
        ) -> list[CheckResult]: ...
    ```

=== "Rust"

    ```rust
    pub trait ConstraintChecker: Send + Sync {
        fn kind(&self) -> &str;
        fn check_route(
            &self,
            route: &Route,
            constraint: &TaskConstraint,
        ) -> Result<Vec<CheckResult>>;
    }
    ```

This results in the following API:

=== "Python"

    ```python
    def score_candidate(candidate, *, target, constraints, registry, options=None) -> ScoredCandidate: ...

    def score_target(candidates, *, target, constraints, registry, options=None) -> TargetResult: ...

    def score(predictions, task, *, registry, options=None) -> Evaluation: ...
    ```

=== "Rust"

    ```rust
    pub fn score_candidate(
        candidate: Candidate,
        target: &Target,
        constraints: &[TaskConstraint],
        registry: &CheckerRegistry,
        options: &ScoreOptions,
    ) -> Result<ScoredCandidate>;

    pub fn score_target(
        candidates: Vec<Candidate>,
        target: &Target,
        constraints: &[TaskConstraint],
        registry: &CheckerRegistry,
        options: &ScoreOptions,
    ) -> Result<TargetResult>;

    pub fn score(
        predictions: CollectedCandidates,
        task: &Task,
        registry: &CheckerRegistry,
        options: &ScoreOptions,
    ) -> Result<Evaluation>;
    ```

The registry is validated before target work begins. A missing checker is one configuration error for the evaluation, not a separate failed route check for every candidate.

### How task constraint checkers work?

There are two components. `Task.effective_constraints(target_id)` holds information on what constraint should be satisfied. That constraint is passed to `.check_route` within the score workflow. For example:

```py
constraints = benchmark.effective_constraints("target-1")
# [
#   StockTerminationConstraint(stock="buyables-stock"),
#   RequiredLeavesConstraint(smiles=["CCO"]),
# ]

for constraint in constraints:
    checker = registry[constraint.kind]
    checks.extend(checker.check_route(route, constraint))
```

For example, for a regular retrosynthesis benchmark (task constraint is stock termination):

For a stock-only task:

```py
benchmark = Task(
    name="mkt-cnv-160",
    targets=targets,
    default_constraints=[
        StockTerminationConstraint(stock="buyables-stock"),
    ],
)

score(
    predictions,
    task=benchmark,
    constraint_checkers=[
        StockTerminationChecker(
            stocks={"buyables-stock": buyables}, # buyables is Set[InChIKeyStr]
            match_level=InChIKeyLevel.FULL,
        ),
    ],
)
```

For a bidirectional search problem (stock-plus-required-leaf task):

```py
benchmark = Task(
    name="mkt-cnv-160-leaf",
    targets=targets,
    default_constraints=[
        StockTerminationConstraint(stock="buyables-stock"),
    ],
    constraints={
        "target-1": [
            RequiredLeavesConstraint(smiles=["CCO"]),
        ],
    },
)
```

and scoring needs checkers for both kinds:

```py
score(
    predictions,
    task=benchmark,
    constraint_checkers=[
        StockTerminationChecker(
            stocks={"buyables-stock": buyables},
            match_level=InChIKeyLevel.FULL,
        ),
        RequiredLeavesChecker(match_level=InChIKeyLevel.FULL),
    ],
)
```

This architecture allows for definition of custom `TaskConstraint` for a given `Task` and so long as you define a complementary `TaskConstraintChecker`, you can use the general `score` workflow.

Notably, the `score` workflow enforces that requirement. If you try to pass a `Task` with constraint of kind `ariadne.reaction_count`, but did not specify a corresponding kind checker, you'd get a runtime error. RetroCast treats unverifiable constraints as unsolved by default.

## 5. Analyze

`analyze` derives benchmark-style metrics from `Evaluation`.

=== "Python"

    ```python
    class ReliabilityFlag(BaseModel):
        code: Literal["OK", "LOW_N", "EXTREME_P"]
        message: str


    class MetricSummary(BaseModel):
        value: float
        count: int
        ci_low: float | None = None
        ci_high: float | None = None
        reliability: ReliabilityFlag | None = None


    class AnalysisReport(BaseModel):
        metrics: dict[str, MetricSummary] = Field(default_factory=dict)
        by_stratum: dict[str, dict[str, MetricSummary]] = Field(default_factory=dict)
        bootstrap_resamples: int
        runtime: RuntimeSummary | None = None
        schema_version: str = "2"


    analyze(evaluation, *, ks=(1, 5, 10, 50), prefix_depths=(1, 2, 3)) -> AnalysisReport
    ```

=== "Rust"

    ```rust
    pub struct ReliabilityFlag {
        pub code: ReliabilityCode,
        pub message: String,
    }

    pub struct MetricSummary {
        pub value: f64,
        pub count: usize,
        pub ci_low: Option<f64>,
        pub ci_high: Option<f64>,
        pub reliability: Option<ReliabilityFlag>,
    }

    pub struct AnalysisReport {
        pub metrics: BTreeMap<MetricName, MetricSummary>,
        pub by_stratum: BTreeMap<Stratum, BTreeMap<MetricName, MetricSummary>>,
        pub bootstrap_resamples: usize,
        pub runtime: Option<RuntimeSummary>,
        pub schema_version: SchemaVersion,
    }

    pub fn analyze(
        evaluation: &Evaluation,
        options: &AnalysisOptions,
    ) -> Result<AnalysisReport>;
    ```

Analysis reads only the evaluation artifact. It does not reopen planner output, stocks, or task files. `MetricSummary.count` is the metric denominator, and a confidence interval without its count is therefore not a valid summary.

Stage-manifest statistics are derived by the same core rather than recomputed by a front end:

=== "Python"

    ```python
    statistics = evaluation_statistics(evaluation)
    # evaluation can still retain its opaque Rust handle
    ```

=== "Rust"

    ```rust
    let statistics = evaluation_statistics(&evaluation);
    ```

This includes candidate and failure counts, per-target candidate distributions, recorded wall/CPU summaries, and target-level Solv-N counts. These values are provenance data, so Python and the standalone executable must not have separate rounding or denominator rules.

Top-K reconstruction metrics are emitted only for targets with acceptable_routes; if no target has acceptable_routes, reconstruction metrics are omitted. Reconstruction diagnostics use `Evaluation.acceptable_match_level`, so route, root-reaction, and prefix comparisons stay on the same molecular identity basis. `Evaluation.acceptable_route_match` records whether headline acceptable-route reconstruction used target-rooted prefix matching or exact full-route identity. Its model default is `EXACT` so legacy artifacts without the field are interpreted according to their original scoring semantics; new scoring writes the selected mode explicitly.

## Artifact I/O and provenance

Schema values do not know where they came from. Provenance is recorded beside an artifact rather than injected into every route and metric.

=== "Python"

    ```python
    class FileInfo(BaseModel):
        label: str
        path: str
        sha256: str
        content_sha256: str | None = None


    class Manifest(BaseModel):
        action: str
        source: list[FileInfo] = Field(default_factory=list)
        output: list[FileInfo] = Field(default_factory=list)
        parameters: dict[str, Any] = Field(default_factory=dict)
        statistics: dict[str, Any] = Field(default_factory=dict)
        schema_version: str = "2"
    ```

=== "Rust"

    ```rust
    pub struct FileInfo {
        pub label: String,
        pub path: Utf8PathBuf,
        pub sha256: Sha256Digest,
        pub content_sha256: Option<Sha256Digest>,
    }

    pub struct Manifest {
        pub action: ActionName,
        pub source: Vec<FileInfo>,
        pub output: Vec<FileInfo>,
        pub parameters: Annotations,
        pub statistics: Annotations,
        pub schema_version: SchemaVersion,
    }
    ```

The physical hash covers the stored bytes. The optional content hash covers the decompressed canonical JSON representation, so equivalent `.json` and `.json.gz` artifacts can be related without pretending their files are identical.

Readers determine the artifact kind from the requested operation or an explicit kind, not by trying every schema until one happens to validate. Writers use one canonical JSON policy for field names, enum values, map ordering, and omitted optionals. Gzip metadata is fixed so writing the same value twice produces the same bytes.

Verification walks manifest edges and checks both physical files and the logical content-hash chain. It never repairs files as a side effect. The CLI and Python API return the same structured `VerificationReport`; only the front-end presentation differs.
