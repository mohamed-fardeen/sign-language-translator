from __future__ import annotations

import shutil
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.error import URLError

from signlang.utils.logging import get_logger

log = get_logger(__name__)


class DatasetDownloader(ABC):
    def __init__(self, name: str, root: str | Path, url: str) -> None:
        self.name = name
        self.root = Path(root)
        self.url = url

    @abstractmethod
    def expected_files(self) -> list[Path]:
        ...

    def is_complete(self) -> bool:
        return all(p.exists() and p.stat().st_size > 0 for p in self.expected_files())

    def download(self, force: bool = False) -> Path:
        if self.is_complete() and not force:
            log.info("dataset.skip", name=self.name, root=str(self.root))
            return self.root
        self.root.mkdir(parents=True, exist_ok=True)
        out = self.root / f"{self.name}.zip"
        log.info("dataset.download.start", name=self.name, url=self.url, dest=str(out))
        try:
            urllib.request.urlretrieve(self.url, out)
        except URLError as exc:
            log.error("dataset.download.failed", name=self.name, error=str(exc))
            raise
        with zipfile.ZipFile(out) as zf:
            zf.extractall(self.root)
        out.unlink(missing_ok=True)
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


def get_downloader(name: str, root: str | Path, url: str) -> DatasetDownloader:
    registry = {
        "wlasl": WLASLDownloader,
        "msasl": MSASLDownloader,
        "asl_citizen": ASLCitizenDownloader,
    }
    if name not in registry:
        raise ValueError(f"Unknown dataset: {name}")
    return registry[name](name=name, root=root, url=url)


def safe_extract(zip_path: Path, dest: Path, max_bytes: int = 50 * 1024**3) -> None:
    if zip_path.stat().st_size > max_bytes:
        raise ValueError(f"Refusing to extract archive larger than {max_bytes} bytes")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > max_bytes:
            raise ValueError(f"Refusing to extract archive contents larger than {max_bytes} bytes")
        zf.extractall(dest)


def copy_local(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)