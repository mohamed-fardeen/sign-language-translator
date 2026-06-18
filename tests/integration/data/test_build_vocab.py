"""Integration tests for the build_vocab script."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from scripts import build_vocab


@pytest.fixture(autouse=True)
def _use_stdlib_structlog(monkeypatch: pytest.MonkeyPatch):
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )
    monkeypatch.setattr(build_vocab, "configure_logging", lambda *a, **k: None)
    yield


def _write_manifest(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _run(monkeypatch: pytest.MonkeyPatch, manifest: Path, out_dir: Path, vocab_size: int | None = None) -> None:
    args = [
        "build_vocab",
        "--manifest", str(manifest),
        "--out-dir", str(out_dir),
    ]
    if vocab_size is not None:
        args.extend(["--vocab-size", str(vocab_size)])
    monkeypatch.setattr(sys, "argv", args)
    build_vocab.main()


def test_build_vocab_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {"clip": f"v_{i}.npz", "label": 0, "gloss": g, "video": f"v_{i}.mp4"}
            for i, g in enumerate(["apple", "banana", "cherry", "apple", "banana", "date"])
        ],
    )
    out_dir = tmp_path / "features"
    _run(monkeypatch, manifest, out_dir)

    vocab_path = out_dir.parent.parent / "vocab" / "vocab.json"
    assert vocab_path.exists()
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert vocab["vocab_size"] == 4
    assert vocab["blank_id"] == 0
    assert "0" not in vocab["id_to_gloss"], "id 0 must not be assigned to any gloss"
    # Sorted alphabetical
    assert vocab["id_to_gloss"] == {
        "1": "apple",
        "2": "banana",
        "3": "cherry",
        "4": "date",
    }
    assert vocab["gloss_to_id"] == {
        "apple": 1,
        "banana": 2,
        "cherry": 3,
        "date": 4,
    }
    # Splits were written
    assert (out_dir / "train.json").exists()
    assert (out_dir / "val.json").exists()
    assert (out_dir / "test.json").exists()


def test_build_vocab_fails_on_single_gloss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {"clip": f"v_{i}.npz", "label": 0, "gloss": "only", "video": f"v_{i}.mp4"}
            for i in range(3)
        ],
    )
    out_dir = tmp_path / "features"
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run(monkeypatch, manifest, out_dir)
    assert exc_info.value.code == 1
    assert "vocab.too_small" in caplog.text


def test_build_vocab_fails_on_empty_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [])
    out_dir = tmp_path / "features"
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run(monkeypatch, manifest, out_dir)
    assert exc_info.value.code == 1
    assert "vocab.empty_manifest" in caplog.text


def test_id_to_gloss_does_not_contain_phantom_zero_gloss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the original bug: vocab_size=1, id_to_gloss={'1': 0}."""
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {"clip": f"v_{i}.npz", "label": 0, "gloss": g, "video": f"v_{i}.mp4"}
            for i, g in enumerate(["zebra", "apple", "mango", "apple", "mango", "zebra"])
        ],
    )
    out_dir = tmp_path / "features"
    _run(monkeypatch, manifest, out_dir)

    vocab_path = out_dir.parent.parent / "vocab" / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert vocab["vocab_size"] == 3
    assert "1" in vocab["id_to_gloss"]
    assert vocab["id_to_gloss"]["1"] == "apple"  # alphabetically first
    # The original bug: id_to_gloss={"1": 0} is now fixed
    assert vocab["id_to_gloss"].get("1") != 0


def test_build_vocab_caps_vocab_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --vocab-size is smaller than the number of unique glosses,
    only the top-N by frequency are kept.
    """
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {"clip": f"v_{i}.npz", "label": 0, "gloss": g, "video": f"v_{i}.mp4"}
            for i, g in enumerate(
                ["apple"] * 10 + ["banana"] * 5 + ["cherry"] * 3 + ["date"] * 1
            )
        ],
    )
    out_dir = tmp_path / "features"
    _run(monkeypatch, manifest, out_dir, vocab_size=2)

    vocab_path = out_dir.parent.parent / "vocab" / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert vocab["vocab_size"] == 2
    assert set(vocab["id_to_gloss"].values()) == {"apple", "banana"}


def test_build_vocab_relabels_manifest_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After build_vocab, the train/val/test manifests use the new ids
    (which are 1..N alphabetical, possibly different from the input
    label)."""
    records = []
    for gloss, count in [("apple", 6), ("mango", 4), ("zebra", 5)]:
        for j in range(count):
            records.append(
                {
                    "clip": f"{gloss}_{j}.npz",
                    "label": 999,
                    "gloss": gloss,
                    "video": f"{gloss}_{j}.mp4",
                }
            )
    manifest = _write_manifest(tmp_path / "manifest.jsonl", records)
    out_dir = tmp_path / "features"
    _run(monkeypatch, manifest, out_dir)

    by_clip: dict[str, dict] = {}
    for split in ("train.json", "val.json", "test.json"):
        data = json.loads((out_dir / split).read_text(encoding="utf-8"))
        for r in data["records"]:
            by_clip[r["clip"]] = r

    # apple=1, mango=2, zebra=3 (alphabetical, 1-based)
    assert by_clip["apple_0.npz"]["label"] == 1
    assert by_clip["mango_0.npz"]["label"] == 2
    assert by_clip["zebra_0.npz"]["label"] == 3
    # Old placeholder labels (999) are gone everywhere.
    assert all(r["label"] != 999 for r in by_clip.values())
