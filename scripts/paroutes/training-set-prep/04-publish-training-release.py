from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

from retrocast.io.provenance import calculate_file_hash
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RELEASES_DIR = BASE_DIR / "data" / "retrocast" / "releases" / "paroutes-training-sets"
DEFAULT_DATASET = "paroutes"
REMOTE_ROOT_ENV_VAR = "RETROCAST_TRAINING_SET_REMOTE_ROOT"
DOCS_ROOT = "https://retrocast.ischemist.com"
RELATED_DOC_LINKS = [
    ("docs home", DOCS_ROOT + "/"),
    ("quick start", DOCS_ROOT + "/quick-start/"),
    ("python library guide", DOCS_ROOT + "/guides/library/"),
    ("training sets guide", DOCS_ROOT + "/guides/training-sets/"),
]
TRAINING_INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>retrocast {dataset} training sets</title>
  <style>
    :root {{
      color-scheme: light dark;
      --page-background: #f4f4f4;
      --surface-background: #ffffff;
      --text-color: #111111;
      --muted-text-color: #666666;
      --meta-text-color: #888888;
      --border-color: #dddddd;
      --header-background: #333333;
      --header-text-color: #ffffff;
      --row-hover-background: #f5f5f5;
      --link-color: #005fcc;
      --visited-link-color: #5d2ca0;
    }}

    @media (prefers-color-scheme: dark) {{
      :root {{
        --page-background: #101418;
        --surface-background: #161b22;
        --text-color: #e6edf3;
        --muted-text-color: #9da7b3;
        --meta-text-color: #8b949e;
        --border-color: #30363d;
        --header-background: #21262d;
        --header-text-color: #f0f6fc;
        --row-hover-background: #1f2630;
        --link-color: #7ab7ff;
        --visited-link-color: #c297ff;
      }}
    }}

    body {{
      font-family: monospace;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      background: var(--page-background);
      color: var(--text-color);
    }}
    h1 {{ border-bottom: 2px solid var(--header-background); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface-background); }}
    th, td {{ padding: 8px; border: 1px solid var(--border-color); text-align: left; vertical-align: top; }}
    th {{ background-color: var(--header-background); color: var(--header-text-color); }}
    tr:hover {{ background-color: var(--row-hover-background); }}
    a {{ color: var(--link-color); }}
    a:visited {{ color: var(--visited-link-color); }}
    .hash {{ font-size: 0.8em; color: var(--muted-text-color); }}
    .meta {{ font-size: 0.8em; color: var(--meta-text-color); margin-bottom: 20px; }}
    ul {{ margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>{dataset} training sets</h1>
  <div class="meta">
    generated: {timestamp}<br>
    latest release: <a href="{latest_json_href}">{release}</a><br>
    release checksums: <a href="{release_checksums_href}">{release}/SHA256SUMS</a><br>
    shell helper: <a href="/retrocast/get-training-set.sh">get-training-set.sh</a>
  </div>

  <h2>related docs</h2>
  <ul>
    {doc_links}
  </ul>

  <h2>published files for {release}</h2>
  <table>
    <tr><th>file</th><th>size</th><th>sha256</th></tr>
    {rows}
  </table>
</body>
</html>
"""


ReleaseFileEntry = tuple[Path, str, int]


def main() -> None:
    configure_script_logging()
    parser = argparse.ArgumentParser(description="stage or upload a hosted paroutes training-set release.")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=find_latest_release_dir(DEFAULT_RELEASES_DIR),
        help="local release root to publish",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset slug under training-sets/")
    parser.add_argument(
        "--remote-root",
        default=os.getenv(REMOTE_ROOT_ENV_VAR),
        help="remote retrocast root, e.g. icgroup:/var/www/files.ischemist.com/retrocast",
    )
    parser.add_argument("--staging-dir", type=Path, default=None, help="optional persistent local staging dir")
    parser.add_argument("--upload", action="store_true", help="rsync the staged release to the remote host")
    args = parser.parse_args()

    with managed_staging_dir(args.staging_dir) as staging_dir:
        staged_root = stage_training_release(
            release_root=args.release_root,
            dataset=args.dataset,
            staging_dir=staging_dir,
        )
        logger.info("staged hosted training-set payload at %s", staged_root)

        if args.upload:
            if not args.remote_root:
                parser.error(
                    f"--upload requires --remote-root or ${REMOTE_ROOT_ENV_VAR}, "
                    "e.g. icgroup:/var/www/files.ischemist.com/retrocast"
                )
            upload_training_release(staged_root=staged_root, dataset=args.dataset, remote_root=args.remote_root)
            upload_shell_script(remote_root=args.remote_root)


def find_latest_release_dir(releases_dir: Path) -> Path:
    candidates = sorted(path for path in releases_dir.iterdir() if path.is_dir() and path.name.startswith("v"))
    if not candidates:
        raise FileNotFoundError(f"no release directories found under {releases_dir}")
    return candidates[-1]


@contextmanager
def managed_staging_dir(staging_dir: Path | None) -> Iterator[Path]:
    if staging_dir is not None:
        staging_dir.mkdir(parents=True, exist_ok=True)
        yield staging_dir
        return

    with tempfile.TemporaryDirectory(prefix="retrocast-training-set-publish-") as tmp_dir:
        yield Path(tmp_dir)


def stage_training_release(*, release_root: Path, dataset: str, staging_dir: Path) -> Path:
    hosted_root = staging_dir / "training-sets" / dataset
    release_dir = hosted_root / release_root.name
    artifact_dirs = find_release_artifact_dirs(release_root)

    if release_dir.exists():
        shutil.rmtree(release_dir)

    for artifact_dir in artifact_dirs:
        staged_artifact_dir = release_dir / artifact_dir.name
        copy_artifact_dir(source=artifact_dir, destination=staged_artifact_dir)
    release_files = collect_release_file_entries(release_dir)
    write_release_sha256sums(release_dir, release_files)

    latest_path = hosted_root / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps({"dataset": dataset, "latest_release": release_root.name}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dataset_index(
        hosted_root=hosted_root, dataset=dataset, release=release_root.name, release_files=release_files
    )
    return hosted_root


def find_release_artifact_dirs(release_root: Path) -> list[Path]:
    artifact_dirs = sorted(
        path for path in release_root.iterdir() if path.is_dir() and (path / "manifest.json").exists()
    )
    if not artifact_dirs:
        raise FileNotFoundError(f"no artifact directories with manifest.json found under {release_root}")
    return artifact_dirs


def copy_artifact_dir(*, source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.name.startswith(".") or path.name == "SHA256SUMS":
            continue
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def collect_release_file_entries(release_dir: Path) -> list[ReleaseFileEntry]:
    files = sorted(path for path in release_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    return [(path.relative_to(release_dir), calculate_file_hash(path), path.stat().st_size) for path in files]


def write_release_sha256sums(release_dir: Path, files: list[ReleaseFileEntry]) -> None:
    sha256sums = "".join(f"{sha256}  {relative_path.as_posix()}\n" for relative_path, sha256, _ in files)
    (release_dir / "SHA256SUMS").write_text(sha256sums, encoding="utf-8")


def write_dataset_index(
    *, hosted_root: Path, dataset: str, release: str, release_files: list[ReleaseFileEntry]
) -> None:
    rows = "".join(
        f"<tr><td><a href='{build_release_href(release, relative_path)}'>{escape(release + '/' + relative_path.as_posix())}</a></td>"
        f"<td>{format_size(size_bytes)}</td><td class='hash'>{sha256}</td></tr>"
        for relative_path, sha256, size_bytes in release_files
    )
    doc_links = "".join(f"<li><a href='{href}'>{label}</a></li>" for label, href in RELATED_DOC_LINKS)
    html = TRAINING_INDEX_TEMPLATE.format(
        dataset=escape(dataset),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        latest_json_href="latest.json",
        release=escape(release),
        release_checksums_href=build_release_href(release, Path("SHA256SUMS")),
        doc_links=doc_links,
        rows=rows,
    )
    (hosted_root / "index.html").write_text(html, encoding="utf-8")


def build_release_href(release: str, relative_path: Path) -> str:
    parts = [quote(release, safe="")]
    parts.extend(quote(part, safe=".") for part in relative_path.parts)
    return "/".join(parts)


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.2f} MB"
    return f"{size_bytes / 1024:.2f} KB"


def upload_training_release(*, staged_root: Path, dataset: str, remote_root: str) -> None:
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        str(staged_root) + "/",
        f"{remote_root}/training-sets/{dataset}/",
    ]
    logger.info("uploading staged training-set payload to %s/training-sets/%s/", remote_root, dataset)
    subprocess.run(cmd, check=True)


def upload_shell_script(*, remote_root: str) -> None:
    shell_script = BASE_DIR / "get-training-set.sh"
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        str(shell_script),
        f"{remote_root}/get-training-set.sh",
    ]
    logger.info("uploading %s to %s/get-training-set.sh", shell_script, remote_root)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
