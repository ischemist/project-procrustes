from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "paroutes"
    / "training-set-prep"
    / "04-publish-training-release.py"
)
SPEC = importlib.util.spec_from_file_location("publish_training_release", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"failed to load publish script module from {MODULE_PATH}")
publish_training_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_training_release)


@pytest.mark.integration
class TestPublishTrainingRelease:
    def test_stage_training_release_keeps_dataset_directory(self, tmp_path) -> None:
        release_root = tmp_path / "releases" / "v2026-05-12"
        artifact_dir = release_root / "route-holdout-n1-n5"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (artifact_dir / "training.jsonl.gz").write_text("payload", encoding="utf-8")

        staged_root = publish_training_release.stage_training_release(
            release_root=release_root,
            dataset="paroutes",
            staging_dir=tmp_path / "staging",
        )

        assert staged_root == tmp_path / "staging" / "training-sets" / "paroutes"
        assert (staged_root / "latest.json").exists()
        assert (staged_root / "index.html").exists()
        assert (staged_root / "v2026-05-12" / "SHA256SUMS").exists()
        assert not (staged_root / "v2026-05-12" / "route-holdout-n1-n5" / "SHA256SUMS").exists()
        assert "route-holdout-n1-n5/training.jsonl.gz" in (staged_root / "v2026-05-12" / "SHA256SUMS").read_text(
            encoding="utf-8"
        )

    def test_upload_training_release_targets_dataset_directory(self, monkeypatch) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], check: bool) -> None:
            assert check is True
            calls.append(cmd)

        monkeypatch.setattr(publish_training_release.subprocess, "run", fake_run)

        publish_training_release.upload_training_release(
            staged_root=Path("/tmp/staging/training-sets/paroutes"),
            dataset="paroutes",
            remote_root="icgroup:/var/www/files.ischemist.com/retrocast",
        )

        assert calls == [
            [
                "rsync",
                "-avz",
                "--progress",
                "/tmp/staging/training-sets/paroutes/",
                "icgroup:/var/www/files.ischemist.com/retrocast/training-sets/paroutes/",
            ]
        ]
