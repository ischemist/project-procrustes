from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath

PR_LOC_MARKER = "<!-- retrocast-pr-loc-report -->"


@dataclass(frozen=True)
class LineChange:
    path: str
    additions: int
    deletions: int

    @property
    def changed(self) -> int:
        return self.additions + self.deletions

    @property
    def net(self) -> int:
        return self.additions - self.deletions


@dataclass(frozen=True)
class LocBucket:
    label: str
    paths: tuple[str, ...]
    additions: int
    deletions: int

    @property
    def changed(self) -> int:
        return self.additions + self.deletions

    @property
    def net(self) -> int:
        return self.additions - self.deletions


def main() -> None:
    parser = argparse.ArgumentParser(description="report pr line changes by retrocast module bucket")
    parser.add_argument("--base", required=True, help="base git ref")
    parser.add_argument("--head", default="HEAD", help="head git ref")
    args = parser.parse_args()

    changes = _git_numstat(args.base, args.head)
    print(_markdown_report(changes, args.base, args.head))


def _git_numstat(base: str, head: str) -> list[LineChange]:
    result = subprocess.run(
        ["git", "diff", "--numstat", f"{base}...{head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    changes: list[LineChange] = []
    for line in result.stdout.splitlines():
        additions, deletions, path = line.split("\t", 2)
        if additions == "-" or deletions == "-":
            continue
        changes.append(LineChange(path=path, additions=int(additions), deletions=int(deletions)))
    return changes


def _markdown_report(changes: list[LineChange], base: str, head: str) -> str:
    buckets = [
        _bucket("total", changes),
        _bucket("source total", [change for change in changes if _is_source_path(change.path)]),
        _bucket("core", [change for change in changes if _is_core_path(change.path)]),
        _bucket("io/curation", [change for change in changes if _is_io_curation_path(change.path)]),
        _bucket("cli/visualization", [change for change in changes if _is_cli_visualization_path(change.path)]),
        _bucket("docs", [change for change in changes if _is_docs_path(change.path)]),
        _bucket(
            "non-source",
            [change for change in changes if not _is_source_path(change.path) and not _is_docs_path(change.path)],
        ),
    ]

    lines = [
        PR_LOC_MARKER,
        "### pr loc report",
        "",
        f"`{base}...{head}`",
        "",
        "| bucket | files | + | - | changed | net |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket in buckets:
        lines.append(
            f"| {bucket.label} | {len(bucket.paths)} | {bucket.additions} | {bucket.deletions} | "
            f"{bucket.changed} | {_format_signed(bucket.net)} |"
        )
    lines.extend(
        [
            "",
            "bucket definitions:",
            "- `core`: `src/retrocast/**` excluding `cli`, `visualization`, `io`, and `curation`.",
            "- `io/curation`: `src/retrocast/io/**` and `src/retrocast/curation/**`.",
            "- `cli/visualization`: `src/retrocast/cli/**` and `src/retrocast/visualization/**`.",
            "- `docs`: `docs/**` and `prompts/**`.",
        ]
    )
    return "\n".join(lines)


def _bucket(label: str, changes: list[LineChange]) -> LocBucket:
    return LocBucket(
        label=label,
        paths=tuple(sorted({change.path for change in changes})),
        additions=sum(change.additions for change in changes),
        deletions=sum(change.deletions for change in changes),
    )


def _is_source_path(path: str) -> bool:
    return PurePosixPath(path).parts[:2] == ("src", "retrocast")


def _is_docs_path(path: str) -> bool:
    return PurePosixPath(path).parts[:1] in {("docs",), ("prompts",)}


def _is_core_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return parts[:2] == ("src", "retrocast") and (
        len(parts) < 3 or parts[2] not in {"cli", "visualization", "io", "curation"}
    )


def _is_io_curation_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return parts[:3] in {("src", "retrocast", "io"), ("src", "retrocast", "curation")}


def _is_cli_visualization_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return parts[:3] in {("src", "retrocast", "cli"), ("src", "retrocast", "visualization")}


def _format_signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


if __name__ == "__main__":
    main()
