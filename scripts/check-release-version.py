#!/usr/bin/env python3
"""Fail a release before building when its tag and Rust packages disagree."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from pathlib import Path

RETROCAST_PACKAGES = frozenset(
    {
        "retrocast-cli",
        "retrocast-core",
        "retrocast-python",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="",
        help="Release tag to validate, such as v0.8.3. Empty checks package consistency only.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    manifest = repository_root / "packages" / "retrocast-rs" / "Cargo.toml"
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    workspace_members = set(metadata["workspace_members"])
    package_versions = {
        package["name"]: package["version"]
        for package in metadata["packages"]
        if package["id"] in workspace_members and package["name"] in RETROCAST_PACKAGES
    }

    missing_packages = RETROCAST_PACKAGES - package_versions.keys()
    if missing_packages:
        names = ", ".join(sorted(missing_packages))
        raise SystemExit(f"missing RetroCast workspace packages from cargo metadata: {names}")

    versions = set(package_versions.values())
    if len(versions) != 1:
        details = ", ".join(f"{name}={version}" for name, version in sorted(package_versions.items()))
        raise SystemExit(f"RetroCast workspace package versions disagree: {details}")

    version = versions.pop()
    lockfile = manifest.with_name("Cargo.lock")
    lock_data = tomllib.loads(lockfile.read_text())
    locked_versions = {
        package["name"]: package["version"] for package in lock_data["package"] if package["name"] in RETROCAST_PACKAGES
    }
    if locked_versions != package_versions:
        expected = ", ".join(f"{name}={value}" for name, value in sorted(package_versions.items()))
        actual = ", ".join(f"{name}={value}" for name, value in sorted(locked_versions.items()))
        raise SystemExit(f"Cargo.lock package versions disagree (expected {expected}; found {actual})")

    if args.tag and args.tag != f"v{version}":
        raise SystemExit(f"release tag {args.tag!r} does not match RetroCast workspace version {version!r}")

    print(f"RetroCast workspace packages consistently report {version}")
    if args.tag:
        print(f"Release tag {args.tag} matches workspace version {version}")


if __name__ == "__main__":
    main()
