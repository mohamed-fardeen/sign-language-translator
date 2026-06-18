from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

from signlang.utils.logging import get_logger

log = get_logger(__name__)


class DatasetDownloader(ABC):
    """Base class for dataset downloaders backed by the Kaggle CLI.

    Requires the ``kaggle`` package and credentials configured either at
    ``~/.kaggle/kaggle.json`` or via the ``KAGGLE_USERNAME`` and
    ``KAGGLE_KEY`` environment variables.
    """

    def __init__(self, name: str, root: str | Path, kaggle_id: str) -> None:
        self.name = name
        self.root = Path(root)
        self.kaggle_id = kaggle_id

    @abstractmethod
    def expected_files(self) -> list[Path]:
        ...

    def is_complete(self) -> bool:
        def _ok(p: Path) -> bool:
            if not p.exists():
                return False
            if p.is_file():
                return p.stat().st_size > 0
            return p.is_dir()

        return all(_ok(p) for p in self.expected_files())

    def _run_kaggle(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            self.kaggle_id,
            "-p",
            str(self.root),
        ]
        log.info("dataset.download.start", name=self.name, kaggle_id=self.kaggle_id, dest=str(self.root))
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Kaggle CLI not found. Install with `pip install kaggle` and "
                "configure credentials at https://www.kaggle.com/settings."
            ) from exc
        except subprocess.CalledProcessError as exc:
            log.error("dataset.download.failed", name=self.name, stderr=exc.stderr)
            raise RuntimeError(
                f"kaggle datasets download failed for {self.name}: {exc.stderr.strip()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"kaggle datasets download timed out for {self.name}"
            ) from exc

    def _extract_zip(self) -> None:
        for zip_path in sorted(self.root.glob("*.zip")):
            log.info("dataset.extract", name=self.name, zip=str(zip_path))
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    bad = zf.testzip()
                    if bad is not None:
                        raise RuntimeError(
                            f"Corrupt entry in {zip_path}: {bad}"
                        )
                    zf.extractall(self.root)
            except zipfile.BadZipFile as exc:
                raise RuntimeError(f"Not a valid zip file: {zip_path}") from exc
            zip_path.unlink()

    def _validate(self) -> None:
        def _missing(p: Path) -> str | None:
            if not p.exists():
                return str(p)
            if p.is_file() and p.stat().st_size == 0:
                return str(p)
            if p.is_dir() and not any(p.iterdir()):
                return str(p)
            return None

        bad = [m for m in (_missing(p) for p in self.expected_files()) if m is not None]
        if bad:
            raise RuntimeError(
                f"Download of {self.name} finished but expected files are missing or empty: {bad}"
            )

    def download(self, force: bool = False) -> Path:
        if self.is_complete() and not force:
            log.info("dataset.skip", name=self.name, root=str(self.root))
            return self.root
        self.root.mkdir(parents=True, exist_ok=True)
        self._run_kaggle()
        self._extract_zip()
        self._validate()
        log.info("dataset.download.done", name=self.name, root=str(self.root))
        return self.root


class WLASLDownloader(DatasetDownloader):
    def expected_files(self) -> list[Path]:
        return [self.root / "WLASL_v0.3.json", self.root / "videos"]


class MSASLDownloader(DatasetDownloader):
    def expected_files(self) -> list[Path]:
        return [self.root / "MSASL_train.json", self.root / "videos"]


class ASLCitizenDownloader(DatasetDownloader):
    def expected_files(self) -> list[Path]:
        return [self.root / "ASL_Citizen.json", self.root / "videos"]


def get_downloader(name: str, root: str | Path, kaggle_id: str) -> DatasetDownloader:
    registry = {
        "wlasl": WLASLDownloader,
        "msasl": MSASLDownloader,
        "asl_citizen": ASLCitizenDownloader,
    }
    if name not in registry:
        raise ValueError(f"Unknown dataset: {name}")
    return registry[name](name=name, root=root, kaggle_id=kaggle_id)


def copy_local(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)
