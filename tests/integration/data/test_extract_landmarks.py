"""Integration tests for the parallel extract_landmarks script.

All tests call ``extract_landmarks.main()`` in-process and replace
``ProcessPoolExecutor`` with a class double so no real processes spawn.
"""
from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from scripts import extract_landmarks

from signlang.data.landmarks.holistic import process_video


def _make_video(path: Path, n_frames: int = 4) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create test video: {path}")
    for i in range(n_frames):
        frame = np.full((64, 64, 3), fill_value=(i * 30) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_annotations(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    interim_dir: Path,
    out_dir: Path,
    extra_argv: list[str] | None = None,
    annotations_file: Path | None = None,
    pool_cls=None,
    cpu_count: int | None = None,
) -> None:
    args: list[str] = ["extract_landmarks", "--interim-dir", str(interim_dir), "--out-dir", str(out_dir)]
    if annotations_file is not None:
        args.extend(["--annotations", str(annotations_file)])
    if extra_argv:
        args.extend(extra_argv)
    monkeypatch.setattr(sys, "argv", args)
    if cpu_count is not None:
        monkeypatch.setattr("os.cpu_count", lambda: cpu_count)
    if pool_cls is not None:
        monkeypatch.setattr(extract_landmarks, "ProcessPoolExecutor", pool_cls)
    extract_landmarks.main()


def _make_fake_pool_class(state: dict, results: list[tuple[bool, str | None]] | None = None):
    class _FakePool:
        def __init__(self, max_workers: int) -> None:
            state["max_workers"] = max_workers
            state["exited"] = False
            self._results = results or []
            self._idx = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            state["exited"] = True
            return False

        def submit(self, fn, *args, **kwargs) -> Future:
            state.setdefault("submissions", []).append(("submit", args))
            fut: Future = Future()
            if self._idx < len(self._results):
                res = self._results[self._idx]
            else:
                res = (True, None)
            self._idx += 1
            fut.set_result(res)
            return fut

    return _FakePool


def _make_never_call_pool_class(state: dict):
    class _NeverPool:
        def __init__(self, max_workers: int) -> None:
            state["called"] = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, *a, **k):
            raise AssertionError("submit should not be called")

    return _NeverPool


def _make_real_pool_class():
    class _RealPool:
        def __init__(self, max_workers: int) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *args, **kwargs) -> Future:
            fut: Future = Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:
                fut.set_exception(exc)
            return fut

    return _RealPool


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
    monkeypatch.setattr(extract_landmarks, "configure_logging", lambda *a, **k: None)
    yield


@pytest.fixture
def fake_videos(tmp_path: Path) -> tuple[Path, Path]:
    interim = tmp_path / "interim"
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _make_video(interim / f"v_{i}.mp4")
    return interim, out


@pytest.fixture
def annotations_file(tmp_path: Path) -> Path:
    return _write_annotations(
        tmp_path / "wlasl_min.json",
        [
            {"gloss": "hello", "instances": [{"video_id": "v_0"}]},
            {"gloss": "thanks", "instances": [{"video_id": "v_1"}]},
            {"gloss": "yes", "instances": [{"video_id": "v_2"}]},
        ],
    )


# --- unit tests ---


def test_default_workers_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [(9, 6), (4, 3), (2, 1), (1, 1), (None, 1), (100, 6)]
    for cpu, expected in cases:
        monkeypatch.setattr("os.cpu_count", lambda c=cpu: c)
        assert extract_landmarks._default_workers() == expected, (
            f"cpu_count={cpu!r} expected {expected}"
        )


def test_process_video_writes_compatible_npz(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _make_video(video)
    out = tmp_path / "v.npz"
    ok, err = process_video(video, out, target_fps=10)
    assert ok, f"process_video failed: {err}"
    with np.load(out) as data:
        keys = set(data.files)
        assert keys == {"pose", "lh", "rh", "mask"}
        assert data["pose"].shape[1] == 99
        assert data["lh"].shape[1] == 63
        assert data["rh"].shape[1] == 63
        assert data["pose"].dtype == np.float32
        assert data["lh"].dtype == np.float32
        assert data["rh"].dtype == np.float32
        assert data["mask"].dtype == bool
        assert data["mask"].all()


def test_process_video_returns_error_on_missing_file(tmp_path: Path) -> None:
    ok, err = process_video(tmp_path / "nope.mp4", tmp_path / "out.npz")
    assert ok is False
    assert err is not None


# --- integration tests: pool + workers ---


def test_uses_process_pool_executor(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        extra_argv=["--workers", "4"],
        pool_cls=pool_cls,
    )
    assert state["max_workers"] == 4
    assert state["exited"] is True
    submitted = [s for tag, s in state["submissions"] if tag == "submit"]
    assert len(submitted) == 3


def test_workers_override_via_cli(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        extra_argv=["--workers", "2"],
        pool_cls=pool_cls,
    )
    assert state["max_workers"] == 2


def test_workers_default_uses_cpu_count_formula(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        pool_cls=pool_cls,
        cpu_count=8,
    )
    assert state["max_workers"] == 6


def test_skips_existing_npz_files(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    pre = out_dir / "v_0.npz"
    pre.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pre,
        pose=np.zeros((2, 99), dtype=np.float32),
        lh=np.zeros((2, 63), dtype=np.float32),
        rh=np.zeros((2, 63), dtype=np.float32),
        mask=np.ones(2, dtype=bool),
    )
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        extra_argv=["--workers", "2"],
        pool_cls=pool_cls,
    )
    submitted = [s for tag, s in state["submissions"] if tag == "submit"]
    submitted_names = {Path(str(args[0])).name for args in submitted}
    assert "v_0.mp4" not in submitted_names
    assert submitted_names == {"v_1.mp4", "v_2.mp4"}


def test_max_videos_caps_submissions(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        extra_argv=["--max-videos", "2", "--workers", "2"],
        pool_cls=pool_cls,
    )
    submitted = [s for tag, s in state["submissions"] if tag == "submit"]
    assert len(submitted) == 2


def test_manifest_records_sorted_by_path(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=annotations_file,
            pool_cls=pool_cls,
        )
    manifest = out_dir.parent / "manifest.jsonl"
    assert manifest.exists()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    paths_in_manifest = [r["clip"] for r in records]
    assert paths_in_manifest == sorted(paths_in_manifest)


def test_progress_log_contains_rate_and_eta(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    monkeypatch.setattr(extract_landmarks, "PROGRESS_EVERY_N", 1)
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=annotations_file,
            pool_cls=pool_cls,
        )
    text = caplog.text
    assert "landmarks.progress" in text
    assert "rate_per_min" in text
    assert "eta_seconds" in text
    assert "landmarks.done" in text


def test_failed_future_does_not_crash_pool(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(
        state,
        results=[
            (False, "synthetic error"),
            (True, None),
            (False, "another error"),
        ],
    )
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=annotations_file,
            pool_cls=pool_cls,
        )
    text = caplog.text
    assert "landmarks.skip" in text
    assert "synthetic error" in text
    assert "another error" in text
    assert "'videos_failed': 2" in text
    assert "'videos_extracted': 1" in text


def test_no_pending_work_no_pool_started(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    for i in range(3):
        pre = out_dir / f"v_{i}.npz"
        pre.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pre,
            pose=np.zeros((2, 99), dtype=np.float32),
            lh=np.zeros((2, 63), dtype=np.float32),
            rh=np.zeros((2, 63), dtype=np.float32),
            mask=np.ones(2, dtype=bool),
        )
    state: dict = {}
    pool_cls = _make_never_call_pool_class(state)
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=annotations_file,
            pool_cls=pool_cls,
        )
    assert not state.get("called", False)
    assert "'skipped_existing': 3" in caplog.text


def test_npz_output_format_compatible_real_extraction(
    tmp_path: Path,
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim = tmp_path / "interim"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_video(interim / "v_0.mp4", n_frames=4)
    pool_cls = _make_real_pool_class()
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        extra_argv=["--workers", "1"],
        pool_cls=pool_cls,
    )
    npz = out_dir / "v_0.npz"
    assert npz.exists()
    with np.load(npz) as data:
        assert set(data.files) == {"pose", "lh", "rh", "mask"}
        T = data["pose"].shape[0]
        assert data["pose"].shape == (T, 99)
        assert data["lh"].shape == (T, 63)
        assert data["rh"].shape == (T, 63)
        assert data["mask"].shape == (T,)
        assert data["pose"].dtype == np.float32
        assert data["lh"].dtype == np.float32
        assert data["rh"].dtype == np.float32
        assert data["mask"].dtype == bool
        assert data["mask"].all()


# --- integration tests: label resolution (the bug fix) ---


def test_extract_fails_when_annotations_arg_missing(
    fake_videos: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    with pytest.raises(SystemExit):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=None,
        )


def test_extract_fails_when_annotations_file_missing(
    fake_videos: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=tmp_path / "nope.json",
            pool_cls=pool_cls,
        )
    assert exc_info.value.code == 1
    assert "annotations.load_failed" in caplog.text


def test_extract_assigns_correct_glosses_from_annotations(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        pool_cls=pool_cls,
    )
    manifest = out_dir.parent / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    by_clip = {r["clip"]: r for r in records}
    assert len(records) == 3
    # Alphabetical order: hello=1, thanks=2, yes=3
    assert by_clip["v_0.npz"]["gloss"] == "hello"
    assert by_clip["v_0.npz"]["label"] == 1
    assert by_clip["v_1.npz"]["gloss"] == "thanks"
    assert by_clip["v_1.npz"]["label"] == 2
    assert by_clip["v_2.npz"]["gloss"] == "yes"
    assert by_clip["v_2.npz"]["label"] == 3


def test_extract_does_not_assign_label_zero_silently(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    _run_main(
        monkeypatch,
        interim_dir=interim,
        out_dir=out_dir,
        annotations_file=annotations_file,
        pool_cls=pool_cls,
    )
    manifest = out_dir.parent / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    assert records, "manifest should not be empty"
    distinct_labels = {r["label"] for r in records}
    assert len(distinct_labels) > 1, f"labels collapsed to {distinct_labels}"
    assert 0 not in distinct_labels, "no record should have label=0"


def test_extract_skips_unannotated_videos_with_warning(
    fake_videos: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    # Annotate only v_0 and v_1; v_2 has no entry.
    ann = _write_annotations(
        tmp_path / "wlasl_partial.json",
        [
            {"gloss": "hello", "instances": [{"video_id": "v_0"}]},
            {"gloss": "thanks", "instances": [{"video_id": "v_1"}]},
        ],
    )
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=ann,
            pool_cls=pool_cls,
        )
    manifest = out_dir.parent / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    assert {r["clip"] for r in records} == {"v_0.npz", "v_1.npz"}
    assert "'videos_skipped_no_annotation': 1" in caplog.text
    assert "no annotation" in caplog.text
    # Pool only got 2 submissions, not 3.
    submitted = [s for tag, s in state["submissions"] if tag == "submit"]
    assert len(submitted) == 2


def test_extract_warns_on_high_skip_pct(
    fake_videos: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    # Annotate only 1 of 3 (33% skip > 10%).
    ann = _write_annotations(
        tmp_path / "wlasl_partial.json",
        [{"gloss": "hello", "instances": [{"video_id": "v_0"}]}],
    )
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=ann,
            pool_cls=pool_cls,
        )
    assert "landmarks.high_skip_pct" in caplog.text
    assert "'skip_pct': 66.67" in caplog.text or "'skip_pct': 66.66" in caplog.text


def test_extract_warns_on_low_label_diversity_does_not_exit(
    fake_videos: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    # All 3 videos have the same gloss.
    ann = _write_annotations(
        tmp_path / "wlasl_single.json",
        [
            {"gloss": "book", "instances": [{"video_id": "v_0"}]},
            {"gloss": "book", "instances": [{"video_id": "v_1"}]},
            {"gloss": "book", "instances": [{"video_id": "v_2"}]},
        ],
    )
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=ann,
            pool_cls=pool_cls,
        )
    assert "landmarks.low_label_diversity" in caplog.text
    # Manifest is still written (extraction succeeds, vocab build is the
    # one that fails on a single-class corpus).
    manifest = out_dir.parent / "manifest.jsonl"
    assert manifest.exists()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 3


def test_extract_done_log_reports_counts(
    fake_videos: tuple[Path, Path],
    annotations_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    interim, out_dir = fake_videos
    state: dict = {}
    pool_cls = _make_fake_pool_class(state)
    with caplog.at_level(logging.INFO):
        _run_main(
            monkeypatch,
            interim_dir=interim,
            out_dir=out_dir,
            annotations_file=annotations_file,
            pool_cls=pool_cls,
        )
    text = caplog.text
    assert "'videos_total': 3" in text
    assert "'videos_extracted': 3" in text
    assert "'unique_glosses': 3" in text
    assert "'unique_labels': 3" in text
    assert "'skip_pct': 0.0" in text
