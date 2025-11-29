"""
Check for duplicate routes per target in a routes.json.gz file.

Uses the Route.signature computed field to identify duplicates.

uv run scripts/dev/dup-check.py
"""

from pathlib import Path

from retrocast.io.data import load_routes


def check_duplicates(routes_path: Path) -> None:
    """
    Load routes and check if all routes for each target are unique.

    Args:
        routes_path: Path to routes.json.gz file
    """
    print(f"Loading routes from {routes_path}...")
    routes_dict = load_routes(routes_path)

    total_targets = len(routes_dict)
    total_routes = sum(len(routes) for routes in routes_dict.values())
    print(f"Loaded {total_routes:,} routes for {total_targets:,} targets\n")

    # Check for duplicates per target
    targets_with_duplicates = []
    total_duplicates = 0

    for target_id, routes in routes_dict.items():
        if len(routes) <= 1:
            # Cannot have duplicates with 0 or 1 route
            continue

        # Collect signatures
        signatures = {}
        duplicates = []

        for i, route in enumerate(routes):
            sig = route.signature
            if sig in signatures:
                # Found a duplicate
                duplicates.append((i, route.rank, signatures[sig]))
            else:
                signatures[sig] = (i, route.rank)

        if duplicates:
            targets_with_duplicates.append(target_id)
            total_duplicates += len(duplicates)

            print(f"Target: {target_id}")
            print(f"  Total routes: {len(routes)}")
            print(f"  Unique signatures: {len(signatures)}")
            print(f"  Duplicates found: {len(duplicates)}")

            for dup_idx, dup_rank, (orig_idx, orig_rank) in duplicates:
                print(f"    Route #{dup_idx} (rank={dup_rank}) is a duplicate of Route #{orig_idx} (rank={orig_rank})")
            print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total targets: {total_targets:,}")
    print(f"Total routes: {total_routes:,}")
    print(f"Targets with duplicates: {len(targets_with_duplicates):,}")
    print(f"Total duplicate routes: {total_duplicates:,}")

    if targets_with_duplicates:
        print("\nTargets with duplicates:")
        for target_id in targets_with_duplicates:
            print(f"  - {target_id}")
    else:
        print("\nNo duplicates found! All routes are unique per target.")


if __name__ == "__main__":
    routes_path = Path("data/3-processed/mkt-cnv-160/dms-explorer-xl/routes.json.gz")

    if not routes_path.exists():
        print(f"Error: Routes file not found at {routes_path}")
        exit(1)

    check_duplicates(routes_path)
