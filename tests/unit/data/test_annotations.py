from __future__ import annotations

import json
from pathlib import Path

import pytest

from signlang.data.ingestion.annotations import (
    build_gloss_to_id,
    build_id_to_gloss,
    load_wlasl_annotations,
)


def _write_annotations(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_load_wlasl_annotations_basic(tmp_path: Path) -> None:
    p = _write_annotations(
        tmp_path / "a.json",
        [
            {"gloss": "book", "instances": [{"video_id": "69241"}, {"video_id": "65225"}]},
            {"gloss": "shoes", "instances": [{"video_id": "70000"}]},
        ],
    )
    out = load_wlasl_annotations(p)
    assert out == {"69241": "book", "65225": "book", "70000": "shoes"}


def test_load_wlasl_annotations_preserves_first_seen_on_collision(tmp_path: Path) -> None:
    p = _write_annotations(
        tmp_path / "a.json",
        [
            {"gloss": "book", "instances": [{"video_id": "100"}]},
            {"gloss": "shoes", "instances": [{"video_id": "100"}]},
        ],
    )
    out = load_wlasl_annotations(p)
    assert out == {"100": "book"}


def test_load_wlasl_annotations_skips_entries_without_required_fields(tmp_path: Path) -> None:
    p = _write_annotations(
        tmp_path / "a.json",
        [
            {"instances": [{"video_id": "1"}]},
            {"gloss": "x", "instances": [{}]},
            {"gloss": "x", "instances": [{"video_id": "2"}]},
            "not a dict",
            {"gloss": "y", "instances": []},
        ],
    )
    out = load_wlasl_annotations(p)
    assert out == {"2": "x"}


def test_load_wlasl_annotations_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_wlasl_annotations(tmp_path / "nope.json")


def test_load_wlasl_annotations_wrong_shape(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON list"):
        load_wlasl_annotations(p)


def test_build_gloss_to_id_is_sorted_and_deterministic() -> None:
    a = build_gloss_to_id({"zebra", "apple", "mango"})
    b = build_gloss_to_id(reversed(["zebra", "apple", "mango"]))
    assert a == b == {"apple": 1, "mango": 2, "zebra": 3}
    assert all(v >= 1 for v in a.values())


def test_build_gloss_to_id_never_assigns_zero() -> None:
    out = build_gloss_to_id({"only_one_gloss"})
    assert 0 not in out.values()
    assert out == {"only_one_gloss": 1}


def test_build_gloss_to_id_dedupes_input() -> None:
    out = build_gloss_to_id(["a", "b", "a", "c", "b", "a"])
    assert out == {"a": 1, "b": 2, "c": 3}


def test_build_id_to_gloss_inverse() -> None:
    g2i = {"apple": 1, "zebra": 3}
    i2g = build_id_to_gloss(g2i)
    assert i2g == {"1": "apple", "3": "zebra"}
    for gloss, i in g2i.items():
        assert i2g[str(i)] == gloss
