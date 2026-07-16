---
icon: lucide/git-branch
---

# Adaptation

Adaptation turns planner-specific output into schema-2 `Candidate`s. An adapter understands one raw format; the workflow decides how to preserve ranks, failures, and benchmark targets.

## Choose A Workflow

| Goal | Python | Rust | Result |
| --- | --- | --- | --- |
| Adapt one planner payload | `retrocast.adapt(...)` | `adapt_candidates_with_workers(...)` | ranked candidates |
| Adapt and collect in memory | `retrocast.ingest(...)` | `adapt::ingest(...)` | predictions grouped by target |
| Read, adapt, and collect a file | `retrocast.ingest_file(...)` | `adapt::ingest_file(...)` | predictions grouped by target |

Use `adapt` to inspect a payload. Use `ingest` for evaluation because it maps every candidate onto a task target and returns the value consumed by scoring.

## Terms

`raw_payload` is the planner artifact passed to an adapter.

`RawRouteEntry` is the envelope produced when the adapter traverses that artifact. It contains one raw route record and provenance such as source order and target hints.

`Route` is the canonical chemistry tree produced by a successful cast.

`Candidate` stores a one-based planner rank and exactly one of `route` or `failure`. A failure is a result, not a missing list element.

## Adapt A Payload

=== "Python 0.8.x"

    ```python
    import retrocast

    candidates = retrocast.adapt(
        raw_payload,
        "paroutes",
        mode="strict",
        max_candidates=50,
        workers=12,
    )

    for candidate in candidates:
        if route := candidate.get("route"):
            print(candidate["rank"], route["target"]["smiles"])
        else:
            print(candidate["rank"], candidate["failure"]["code"])
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::{
        adapters::{adapt_candidates_with_workers, built_in},
        route::AdaptMode,
    };

    let adapter = built_in("paroutes").expect("built-in adapter");
    let candidates = adapt_candidates_with_workers(
        raw_payload,
        adapter.as_ref(),
        AdaptMode::Strict,
        None,
        None,
        Some(50),
        12,
    )?;

    for candidate in candidates {
        match (candidate.route, candidate.failure) {
            (Some(route), None) => println!("{} {}", candidate.rank, route.target.smiles),
            (None, Some(failure)) => println!("{} {}", candidate.rank, failure.code),
            _ => unreachable!("Candidate validates exactly one outcome"),
        }
    }
    ```

=== "Python 0.7.1"

    ```python
    from retrocast import adapt_candidates, get_adapter

    adapter = get_adapter("paroutes")
    candidates = adapt_candidates(
        raw_payload,
        adapter,
        mode="strict",
        max_candidates=50,
    )

    for candidate in candidates:
        if candidate.route is not None:
            print(candidate.rank, candidate.route.target.smiles)
        else:
            print(candidate.rank, candidate.failure.code)
    ```

`max_candidates` means the first N raw prediction slots. Failed slots consume a rank and remain visible. This is required for honest Solv-0 and MRR accounting.

## Supply A Target Hint

Some raw formats need the expected target to validate or interpret a route. The target uses the schema-2 `Target` shape.

=== "Python 0.8.x"

    ```python
    candidates = retrocast.adapt(
        raw_route_record,
        "synplanner",
        target=target_dict,
        source_key="target-001",
    )
    ```

=== "Rust 0.8.x"

    ```rust
    let candidates = adapt_candidates_with_workers(
        raw_route_record,
        adapter.as_ref(),
        AdaptMode::Strict,
        Some(&target),
        Some("target-001"),
        None,
        1,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast import adapt_candidates, get_adapter

    adapter = get_adapter("synplanner")
    candidates = adapt_candidates(
        raw_route_record,
        adapter,
        target=target,
        source_key="target-001",
    )
    ```

If the adapted root does not match the supplied target, the slot becomes an `adapter.target_mismatch` failure.

## Ingest For A Task

Ingest combines adaptation and collection. Successful candidates are collected by route target identity. Failed candidates use the target hints preserved in their `FailureRecord`.

=== "Python 0.8.x"

    ```python
    predictions = retrocast.ingest(
        raw_payload,
        "aizynthfinder",
        task,
        max_candidates=50,
        workers=12,
    )
    ```

=== "Rust 0.8.x"

    ```rust
    use retrocast_core::adapt::ingest;

    let predictions = ingest(
        raw_payload,
        adapter.as_ref(),
        &task,
        AdaptMode::Strict,
        Some(50),
        12,
    )?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast import get_adapter, ingest_candidates

    adapter = get_adapter("aizynthfinder")
    predictions = ingest_candidates(
        raw_payload,
        adapter,
        task,
        max_candidates=50,
    )
    ```

Python 0.8.x returns `NativePredictions`; Rust and Python 0.7.1 return target-id maps of ranked candidates, using Rust types and Pydantic models respectively.

## Stream A Large Artifact

The file entry point opens JSON or JSON gzip in Rust. For multi-target map payloads, it decodes and adapts target-by-target instead of materializing the complete raw graph in Python.

=== "Python 0.8.x"

    ```python
    predictions = retrocast.ingest_file(
        "results.json.gz",
        "aizynthfinder",
        "benchmark.json.gz",
        workers=12,
    )
    predictions.write("candidates.json.gz")
    ```

=== "Rust 0.8.x"

    ```rust
    let predictions = ingest_file(
        raw_path,
        adapter.as_ref(),
        &task,
        AdaptMode::Strict,
        None,
        12,
    )?;
    retrocast_core::io::write_json(output_path, &predictions)?;
    ```

=== "Python 0.7.1"

    ```python
    from retrocast import get_adapter, ingest_candidates
    from retrocast.io import load_benchmark, load_json_gz, save_collected_candidates

    raw_payload = load_json_gz("results.json.gz")
    task = load_benchmark("benchmark.json.gz")
    adapter = get_adapter("aizynthfinder")
    predictions = ingest_candidates(raw_payload, adapter, task)
    save_collected_candidates(predictions, "candidates.json.gz")
    ```

## Adapt Modes

`strict` rejects a raw route with invalid chemistry, cycles, or an impossible route structure.

`prune` allows an adapter to return the longest valid target-rooted prefix when an invalid branch can be removed unambiguously. It still fails if pruning removes the target or leaves a reaction without reactants.

=== "Python 0.8.x"

    ```python
    candidates = retrocast.adapt(raw_payload, "paroutes", mode="prune")
    ```

=== "Rust 0.8.x"

    ```rust
    let mode = AdaptMode::Prune;
    ```

=== "Python 0.7.1"

    ```python
    candidates = adapt_candidates(raw_payload, adapter, mode="prune")
    ```

## Available Adapters

Built-in adapter identifiers are lowercase and stable:

```text
aizynthfinder  askcos          directmultistep  dreamretroer
molbuilder     multistepttl    paroutes          retrochimera
retrostar      synllama        synplanner        syntheseus
ursa
```

The standalone CLI prints the registry with `retrocast list-adapters`. See [Writing a Custom Adapter](../../dev/reference/adapters.md) for the Rust adapter contract and raw-shape patterns.
