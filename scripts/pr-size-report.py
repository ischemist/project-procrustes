from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from rich.console import Console
from rich.table import Table

MARKER = "<!-- retrocast-pr-loc-report -->"
BUCKETS = ("total", "source total", "core", "io/curation", "cli/visualization", "docs", "tests", "ci/scripts")


def main() -> None:
    parser = argparse.ArgumentParser(description="report pr size by module bucket")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--measure", choices=("git", "cloc"), default="git")
    parser.add_argument("--format", choices=("markdown", "rich"), default="markdown")
    args = parser.parse_args()

    rows = _git_rows(args.base, args.head) if args.measure == "git" else _cloc_rows(args.base, args.head)
    if args.format == "rich":
        Console().print(_rich_table(rows, args.measure, args.base, args.head))
    else:
        print(_markdown(rows, args.measure, args.base, args.head))


def _git_rows(base: str, head: str) -> list[dict[str, int | str]]:
    by_bucket = _empty_buckets()
    output = subprocess.check_output(["git", "diff", "--numstat", f"{base}...{head}"], text=True)
    for line in output.splitlines():
        added, deleted, raw_path = line.split("\t", 2)
        if added == "-" or deleted == "-":
            continue
        path = PurePosixPath(raw_path)
        for bucket in _buckets_for_path(path):
            by_bucket[bucket]["files"].add(path)
            by_bucket[bucket]["added"] += int(added)
            by_bucket[bucket]["deleted"] += int(deleted)
    return [
        {
            "bucket": bucket,
            "files": len(values["files"]),
            "+": values["added"],
            "-": values["deleted"],
            "changed": values["added"] + values["deleted"],
            "net": values["added"] - values["deleted"],
        }
        for bucket, values in by_bucket.items()
    ]


def _cloc_rows(base: str, head: str) -> list[dict[str, int | str]]:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        base_tree = _archive(base, tmp / "base")
        head_tree = _archive(head, tmp / "head")
        base_files = _files_by_bucket(_tree_files(base_tree))
        head_files = _files_by_bucket(_tree_files(head_tree))
        rows: list[dict[str, int | str]] = []
        for bucket in BUCKETS:
            base_count = _cloc_count(base_tree, base_files[bucket])
            head_count = _cloc_count(head_tree, head_files[bucket])
            rows.append(
                {
                    "bucket": bucket,
                    "base files": base_count["files"],
                    "head files": head_count["files"],
                    "base code": base_count["code"],
                    "head code": head_count["code"],
                    "code delta": head_count["code"] - base_count["code"],
                }
            )
        return rows


def _archive(ref: str, destination: Path) -> Path:
    destination.mkdir()
    archive = subprocess.run(["git", "archive", ref], check=True, capture_output=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter="data")
    return destination


def _tree_files(root: Path) -> list[PurePosixPath]:
    ignored = {"__pycache__", ".git"}
    return sorted(
        PurePosixPath(path.relative_to(root).as_posix())
        for path in root.rglob("*")
        if path.is_file() and not (set(path.relative_to(root).parts) & ignored)
    )


def _cloc_count(root: Path, paths: list[PurePosixPath]) -> dict[str, int]:
    if not paths:
        return {"files": 0, "code": 0}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as file_list:
        for path in paths:
            file_list.write(str(root / path) + "\n")
        file_list.flush()
        output = subprocess.check_output(["cloc", "--json", "--quiet", f"--list-file={file_list.name}"], text=True)
    total = json.loads(output)["SUM"]
    return {"files": int(total["nFiles"]), "code": int(total["code"])}


def _empty_buckets() -> dict[str, dict[str, int | set[PurePosixPath]]]:
    return {bucket: {"files": set(), "added": 0, "deleted": 0} for bucket in BUCKETS}


def _files_by_bucket(paths: list[PurePosixPath]) -> dict[str, list[PurePosixPath]]:
    buckets = {bucket: [] for bucket in BUCKETS}
    for path in paths:
        for bucket in _buckets_for_path(path):
            buckets[bucket].append(path)
    return buckets


def _buckets_for_path(path: PurePosixPath) -> tuple[str, ...]:
    bucket = _bucket_for_path(path)
    if bucket in {"core", "io/curation", "cli/visualization"}:
        return ("total", "source total", bucket)
    if bucket is not None:
        return ("total", bucket)
    return ("total",)


def _bucket_for_path(path: PurePosixPath) -> str | None:
    if path.parts[:3] in {("src", "retrocast", "io"), ("src", "retrocast", "curation")}:
        return "io/curation"
    if path.parts[:3] in {("src", "retrocast", "cli"), ("src", "retrocast", "visualization")}:
        return "cli/visualization"
    if path.parts[:2] == ("src", "retrocast"):
        return "core"
    if path.parts[:1] in {("docs",), ("prompts",)}:
        return "docs"
    if path.parts[:1] == ("tests",):
        return "tests"
    if path.parts[:1] in {(".github",), ("scripts",)}:
        return "ci/scripts"
    return None


def _markdown(rows: list[dict[str, int | str]], measure: str, base: str, head: str) -> str:
    columns = list(rows[0])
    title = "loc" if measure == "git" else "cloc"
    lines = [MARKER, f"### pr {title} report", "", f"`{base}...{head}`", "", _markdown_header(columns)]
    for row in rows:
        lines.append(
            "| " + " | ".join(_cell(row[column], signed=_is_delta_column(column)) for column in columns) + " |"
        )
    return "\n".join(lines)


def _markdown_header(columns: list[str]) -> str:
    return (
        "| "
        + " | ".join(columns)
        + " |\n| "
        + " | ".join("---" if i == 0 else "---:" for i in range(len(columns)))
        + " |"
    )


def _rich_table(rows: list[dict[str, int | str]], measure: str, base: str, head: str) -> Table:
    columns = list(rows[0])
    title = "loc" if measure == "git" else "cloc"
    table = Table(title=f"pr {title} report: {base}...{head}")
    for index, column in enumerate(columns):
        table.add_column(column, justify="left" if index == 0 else "right")
    for row in rows:
        table.add_row(*(_cell(row[column], signed=_is_delta_column(column)) for column in columns))
    return table


def _cell(value: int | str, *, signed: bool) -> str:
    if signed and isinstance(value, int):
        return f"+{value}" if value > 0 else str(value)
    return str(value)


def _is_delta_column(column: str) -> bool:
    return column in {"+", "net", "code delta"}


if __name__ == "__main__":
    main()
