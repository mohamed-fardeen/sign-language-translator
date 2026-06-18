"""Integration tests for dataset downloaders.

These tests mock ``subprocess.run`` so they never invoke the Kaggle CLI
for real; they verify that the correct command is built, that
extraction and validation are wired up, and that all three datasets
behave identically.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from signlang.data.ingestion.downloaders import (
    ASLCitizenDownloader,
    MSASLDownloader,
    WLASLDownloader,
    get_downloader,
)


def _make_fake_run(files_by_kaggle_id: dict[str, list[str]]):
    """Return a side_effect that "downloads" the requested dataset by
    creating the expected files under the requested target dir.
    """

    def side_effect(cmd, **kwargs):
        d_idx = cmd.index("-d")
        kaggle_id = cmd[d_idx + 1]
        p_idx = cmd.index("-p")
        target = Path(cmd[p_idx + 1])
        target.mkdir(parents=True, exist_ok=True)
        for rel in files_by_kaggle_id.get(kaggle_id, []):
            out = target / rel
            if rel.endswith("/"):
                out.mkdir(parents=True, exist_ok=True)
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"x" * 16)
        return MagicMock(returncode=0, stdout="", stderr="")

    return side_effect


WLASL_FILES = ["WLASL_v0.3.json", "videos/", "videos/00001.mp4"]
MSASL_FILES = ["MSASL_train.json", "videos/", "videos/00001.mp4"]
CITIZEN_FILES = ["ASL_Citizen.json", "videos/", "videos/00001.mp4"]


@pytest.fixture
def fake_kaggle_run():
    return _make_fake_run(
        {
            "risangbaskoro/wlasl-processed": WLASL_FILES,
            "soongpal/ms-asl": MSASL_FILES,
            "grassknoted/asl-citizen": CITIZEN_FILES,
        }
    )


def _assert_kaggle_cmd(cmd: list[str], kaggle_id: str, target: Path) -> None:
    assert cmd[0] == "python" or cmd[0].endswith(("python", "python.exe"))
    assert "-m" in cmd
    assert "kaggle" in cmd
    assert "datasets" in cmd
    assert "download" in cmd
    assert kaggle_id in cmd
    assert str(target) in cmd
    assert "-d" in cmd
    assert "-p" in cmd


def test_wlasl_downloader_invokes_kaggle_cli(tmp_path, fake_kaggle_run):
    dl = WLASLDownloader(
        name="wlasl",
        root=tmp_path,
        kaggle_id="risangbaskoro/wlasl-processed",
    )
    with patch("subprocess.run", side_effect=fake_kaggle_run) as mock_run:
        result = dl.download()
    assert result == tmp_path
    assert (tmp_path / "WLASL_v0.3.json").exists()
    assert (tmp_path / "videos").is_dir()
    args, _ = mock_run.call_args
    _assert_kaggle_cmd(args[0], "risangbaskoro/wlasl-processed", tmp_path)


def test_msasl_downloader_invokes_kaggle_cli(tmp_path, fake_kaggle_run):
    dl = MSASLDownloader(name="msasl", root=tmp_path, kaggle_id="soongpal/ms-asl")
    with patch("subprocess.run", side_effect=fake_kaggle_run) as mock_run:
        dl.download()
    assert (tmp_path / "MSASL_train.json").exists()
    args, _ = mock_run.call_args
    _assert_kaggle_cmd(args[0], "soongpal/ms-asl", tmp_path)


def test_asl_citizen_downloader_invokes_kaggle_cli(tmp_path, fake_kaggle_run):
    dl = ASLCitizenDownloader(
        name="asl_citizen", root=tmp_path, kaggle_id="grassknoted/asl-citizen"
    )
    with patch("subprocess.run", side_effect=fake_kaggle_run) as mock_run:
        dl.download()
    assert (tmp_path / "ASL_Citizen.json").exists()
    args, _ = mock_run.call_args
    _assert_kaggle_cmd(args[0], "grassknoted/asl-citizen", tmp_path)


def test_download_skips_when_already_complete(tmp_path):
    (tmp_path / "WLASL_v0.3.json").write_text("{}")
    (tmp_path / "videos").mkdir()
    dl = WLASLDownloader(
        name="wlasl", root=tmp_path, kaggle_id="risangbaskoro/wlasl-processed"
    )
    with patch("subprocess.run") as mock_run:
        dl.download()
    mock_run.assert_not_called()


def test_download_force_redownloads(tmp_path, fake_kaggle_run):
    (tmp_path / "WLASL_v0.3.json").write_text("{}")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "old.mp4").write_bytes(b"x" * 16)
    dl = WLASLDownloader(
        name="wlasl", root=tmp_path, kaggle_id="risangbaskoro/wlasl-processed"
    )
    with patch("subprocess.run", side_effect=fake_kaggle_run) as mock_run:
        dl.download(force=True)
    assert mock_run.call_count == 1


def test_download_fails_when_kaggle_returns_error(tmp_path):
    dl = WLASLDownloader(
        name="wlasl", root=tmp_path, kaggle_id="bad/slug"
    )
    err = subprocess.CalledProcessError(returncode=1, cmd=["kaggle"], stderr="403 Forbidden")
    with (
        patch("subprocess.run", side_effect=err),
        pytest.raises(RuntimeError, match="kaggle datasets download failed"),
    ):
        dl.download()


def test_download_fails_when_kaggle_cli_missing(tmp_path):
    dl = WLASLDownloader(name="wlasl", root=tmp_path, kaggle_id="x/y")
    with (
        patch("subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(RuntimeError, match="Kaggle CLI not found"),
    ):
        dl.download()


def test_download_fails_when_expected_files_missing(tmp_path):
    def side_effect(cmd, **kwargs):
        Path(cmd[cmd.index("-p") + 1]).mkdir(parents=True, exist_ok=True)
        return MagicMock(returncode=0, stdout="", stderr="")

    dl = WLASLDownloader(
        name="wlasl", root=tmp_path, kaggle_id="risangbaskoro/wlasl-processed"
    )
    with (
        patch("subprocess.run", side_effect=side_effect),
        pytest.raises(RuntimeError, match="expected files are missing"),
    ):
        dl.download()


def test_download_fails_on_corrupt_zip(tmp_path):
    (tmp_path / "WLASL_v0.3.json").parent.mkdir(parents=True, exist_ok=True)
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"PK\x03\x04not-a-real-zip")

    def side_effect(cmd, **kwargs):
        return MagicMock(returncode=0, stdout="", stderr="")

    dl = WLASLDownloader(
        name="wlasl", root=tmp_path, kaggle_id="risangbaskoro/wlasl-processed"
    )
    with patch("subprocess.run", side_effect=side_effect), pytest.raises(RuntimeError):
        dl.download()


def test_get_downloader_factory():
    assert isinstance(get_downloader("wlasl", Path("/tmp"), "x/y"), WLASLDownloader)
    assert isinstance(get_downloader("msasl", Path("/tmp"), "x/y"), MSASLDownloader)
    assert isinstance(get_downloader("asl_citizen", Path("/tmp"), "x/y"), ASLCitizenDownloader)
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_downloader("nope", Path("/tmp"), "x/y")


def test_wlasl_kaggle_id_matches_canonical_slug():
    """Regression: the WLASL slug was previously `risangbudi07/wlasl-processed`
    which 404s. Pin the canonical id here so it cannot drift again.
    """
    from signlang.config import load_config, to_dict

    cfg = to_dict(load_config(["data=wlasl"]))["data"]
    assert cfg["kaggle_id"] == "risangbaskoro/wlasl-processed"
