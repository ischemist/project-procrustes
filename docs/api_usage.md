# RetroCast API Usage Guide

This guide shows how to use RetroCast as a Python library to adapt retrosynthesis routes from different models into a unified format.

## Installation

```bash
pip install retrocast
# or if installing from source
pip install -e .
```

## Quick Start

### Adapting a Single Route

The simplest way to adapt a route is using `adapt_single_route`. This function works with both route-centric and target-centric adapters:

#### Route-Centric Adapters (DMS, AiZynth, SynPlanner)

For route-centric models, pass a single route object from the model's output:

```python
from retrocast import adapt_single_route, TargetInput

# Define your target molecule
target = TargetInput(
    id="aspirin",
    smiles="CC(=O)Oc1ccccc1C(=O)O"
)

# Your model's raw prediction (example: DMS format)
# This is ONE route from the model's output list
raw_route = {
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "children": [
        {"smiles": "Oc1ccccc1C(=O)O", "children": []},
        {"smiles": "CC(=O)Cl", "children": []}
    ]
}

# Adapt it to the unified format
route = adapt_single_route(raw_route, target, adapter_name="dms")

if route:
    print(f"✓ Route successfully adapted!")
    print(f"  Depth: {route.depth} steps")
    print(f"  Starting materials: {len(route.leaves)}")
    print(f"  Route signature: {route.get_signature()[:16]}...")

    # Access route details
    for leaf in route.leaves:
        print(f"  - {leaf.smiles}")
else:
    print("✗ Route adaptation failed")
```

#### Target-Centric Adapters (RetroChimera, ASKCOS)

For target-centric models, pass the complete target data dict (containing metadata and routes):

```python
from retrocast import adapt_single_route, TargetInput

target = TargetInput(id="mol1", smiles="CCO")

# RetroChimera format: complete target data with nested routes
retrochimera_data = {
    "smiles": "CCO",
    "result": {
        "outputs": [{
            "routes": [
                {
                    "reactions": [...],
                    "num_steps": 2,
                    "step_probability_min": 0.85
                }
            ]
        }]
    }
}

# The function automatically detects the format and handles it correctly
route = adapt_single_route(retrochimera_data, target, adapter_name="retrochimera")
```

### Adapting Multiple Routes

Use `adapt_routes` to process multiple routes at once:

```python
from retrocast import adapt_routes, TargetInput

target = TargetInput(
    id="ibuprofen",
    smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O"
)

# Multiple routes from your model
raw_routes = [route1, route2, route3, ...]  # Your model's output

# Adapt all routes
routes = adapt_routes(raw_routes, target, "aizynth")

print(f"Adapted {len(routes)} routes successfully")

# Optionally limit the number of routes
top_10_routes = adapt_routes(raw_routes, target, "aizynth", max_routes=10)
```

## Working with Different Adapters

RetroCast supports multiple retrosynthesis model formats. See which adapters are available:

```python
from retrocast import ADAPTER_MAP

print("Available adapters:")
for name in ADAPTER_MAP.keys():
    print(f"  - {name}")
```

Current adapters include:
- `aizynth` - AiZynthFinder
- `askcos` - ASKCOS
- `dms` - DirectMultiStep
- `dreamretro` - DreamRetro
- `multistepttl` - MultiStepTTL
- `paroutes` - PARoutes
- `retrochimera` - RetroChimera
- `retrostar` - RetroStar
- `synplanner` - SynPlanner
- `syntheseus` - Syntheseus
- `synllama` - SynLLaMA

### Using a Specific Adapter

```python
from retrocast import get_adapter

# Get an adapter instance
adapter = get_adapter("dms")

# Use the adapter directly
target = TargetInput(id="mol1", smiles="CCO")
raw_data = [...]  # Your model's output

for route in adapter.cast(raw_data, target):
    print(f"Route {route.rank}: depth={route.depth}")
```

## Post-Processing Routes

RetroCast provides utilities for filtering and deduplicating routes:

### Deduplication

Remove duplicate routes based on their structural signature:

```python
from retrocast import adapt_routes, deduplicate_routes, TargetInput

target = TargetInput(id="test", smiles="CCO")
raw_routes = [...]  # May contain duplicates

# Adapt all routes
routes = adapt_routes(raw_routes, target, "dms")

# Remove duplicates
unique_routes = deduplicate_routes(routes)

print(f"Found {len(routes)} routes, {len(unique_routes)} unique")
```

### Sampling Strategies

Select a subset of routes using different strategies:

```python
from retrocast import (
    adapt_routes,
    deduplicate_routes,
    sample_top_k,
    sample_random_k,
    sample_k_by_depth,
    TargetInput
)

target = TargetInput(id="test", smiles="CCO")
routes = adapt_routes([...], target, "dms")
unique_routes = deduplicate_routes(routes)

# Take top K routes (by rank)
top_5 = sample_top_k(unique_routes, k=5)

# Take random K routes
random_10 = sample_random_k(unique_routes, k=10)

# Take diverse routes by depth (round-robin from each depth level)
diverse_20 = sample_k_by_depth(unique_routes, max_total=20)
```

## Exploring Route Structure

The `Route` object provides access to the complete retrosynthetic tree:

```python
from retrocast import adapt_single_route, TargetInput

target = TargetInput(id="mol", smiles="CCO")
route = adapt_single_route(raw_route, target, "dms")

# Route properties
print(f"Rank: {route.rank}")
print(f"Depth: {route.depth}")  # Number of reaction steps
print(f"Signature: {route.get_signature()}")

# Access the target molecule
print(f"Target SMILES: {route.target.smiles}")
print(f"Target InChIKey: {route.target.inchikey}")

# Get all starting materials
for leaf in route.leaves:
    print(f"Starting material: {leaf.smiles}")

# Check if it's a leaf (no synthesis step)
if route.target.is_leaf:
    print("This is a direct purchase - target is already available!")

# Access the first reaction step (if not a leaf)
if not route.target.is_leaf and route.target.synthesis_step:
    step = route.target.synthesis_step
    print(f"First reaction has {len(step.reactants)} reactants")

    # Access template, mapped SMILES, etc.
    if step.template:
        print(f"Template: {step.template}")
    if step.mapped_smiles:
        print(f"Mapped reaction: {step.mapped_smiles}")
    if step.reagents:
        print(f"Reagents: {step.reagents}")
```

## Complete Example Workflow

Here's a complete example showing a typical workflow:

```python
from retrocast import (
    adapt_routes,
    deduplicate_routes,
    sample_top_k,
    TargetInput,
    ADAPTER_MAP
)

# 1. Define your target
target = TargetInput(
    id="my_molecule",
    smiles="CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
)

# 2. Load your model's predictions
# (This would come from your model's output file)
raw_routes = [...]  # List of routes in your model's format

# 3. Adapt to unified format
print(f"Adapting routes using 'dms' adapter...")
routes = adapt_routes(raw_routes, target, "dms")
print(f"  ✓ Adapted {len(routes)} routes")

# 4. Remove duplicates
unique_routes = deduplicate_routes(routes)
print(f"  ✓ {len(unique_routes)} unique routes")

# 5. Select top-k
top_10 = sample_top_k(unique_routes, k=10)
print(f"  ✓ Selected top 10 routes")

# 6. Analyze results
for i, route in enumerate(top_10, start=1):
    print(f"\nRoute {i}:")
    print(f"  Rank: {route.rank}")
    print(f"  Depth: {route.depth} steps")
    print(f"  Starting materials: {len(route.leaves)}")
    print(f"  Materials:")
    for leaf in route.leaves:
        print(f"    - {leaf.smiles}")

# 7. Export to JSON (if needed)
import json
routes_dict = [route.model_dump() for route in top_10]
with open("output_routes.json", "w") as f:
    json.dump(routes_dict, f, indent=2)
```

## Error Handling

The API handles errors gracefully:

```python
from retrocast import adapt_single_route, TargetInput
from retrocast.exceptions import RetroCastException

target = TargetInput(id="test", smiles="CCO")

# Invalid data returns None instead of raising
route = adapt_single_route({"invalid": "data"}, target, "dms")
if route is None:
    print("Route adaptation failed - check your data format")

# Invalid adapter name raises an exception
try:
    route = adapt_single_route(data, target, "nonexistent")
except RetroCastException as e:
    print(f"Error: {e}")
```

## Advanced: Direct Adapter Usage

For advanced use cases, you can work directly with adapter instances:

```python
from retrocast import get_adapter, TargetInput

adapter = get_adapter("dms")
target = TargetInput(id="test", smiles="CCO")

# The adapt method is a generator - yields routes one at a time
for route in adapter.cast(raw_data, target):
    # Process route immediately without loading all into memory
    if route.depth <= 3:
        print(f"Found short route with depth {route.depth}")
        # Store or process this route
        break  # Can stop early if you found what you need
```

## Type Hints and IDE Support

RetroCast is fully typed for excellent IDE support:

```python
from retrocast import Route, Molecule, ReactionStep, TargetInput

def analyze_route(route: Route) -> dict[str, int]:
    """Example function with type hints."""
    return {
        "depth": route.depth,
        "num_leaves": len(route.leaves),
        "rank": route.rank
    }

# Your IDE will provide autocomplete and type checking
route: Route = adapt_single_route(data, target, "dms")
if route:
    stats = analyze_route(route)
```

## Next Steps

- See the [Format Specification](format_spec.md) for details on the unified route format
- See the [Adapter Documentation](adapters.md) for model-specific details
- Check the test files in `tests/test_api.py` for more examples
