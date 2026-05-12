from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from retrocast.io.provenance import calculate_file_hash
from retrocast.utils.logging import configure_script_logging, logger

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RELEASES_DIR = BASE_DIR / "data" / "retrocast" / "releases" / "paroutes-training-sets"
DEFAULT_DATASET = "paroutes"
REMOTE_ROOT_ENV_VAR = "RETROCAST_TRAINING_SET_REMOTE_ROOT"


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
        write_sha256sums(staged_artifact_dir)

    latest_path = hosted_root / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps({"dataset": dataset, "latest_release": release_root.name}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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


def write_sha256sums(artifact_dir: Path) -> None:
    files = sorted(path for path in artifact_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    sha256sums = "".join(f"{calculate_file_hash(path)}  {path.name}\n" for path in files)
    (artifact_dir / "SHA256SUMS").write_text(sha256sums, encoding="utf-8")


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
