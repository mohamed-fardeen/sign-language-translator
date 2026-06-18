from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def load_wlasl_annotations(json_path: str | Path) -> dict[str, str]:
    """Read a WLASL annotations file and return a ``{video_id: gloss}`` map.

    The WLASL format is a JSON list of entries::

        [{"gloss": "book", "instances": [{"video_id": "69241", ...}, ...]}, ...]

    Duplicate ``video_id`` values collapse to the first-seen gloss so
    the mapping is well-defined and reproducible.
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"WLASL annotations file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {p}, got {type(data).__name__}"
        )
    out: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        gloss = entry.get("gloss")
        instances = entry.get("instances") or []
        if gloss is None:
            continue
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            vid = inst.get("video_id")
            if vid is None:
                continue
            key = str(vid)
            if key not in out:
                out[key] = str(gloss)
    return out


def build_gloss_to_id(glosses: Iterable[str]) -> dict[str, int]:
    """Deterministic ``gloss -> 1..N`` mapping, sorted alphabetically.

    Id ``0`` is reserved for the CTC blank and is never assigned to
    any gloss.
    """
    seen = sorted({str(g) for g in glosses})
    return {g: i + 1 for i, g in enumerate(seen)}


def build_id_to_gloss(gloss_to_id: dict[str, int]) -> dict[str, str]:
    """Inverse of :func:`build_gloss_to_id` with stringified int keys
    (suitable for JSON serialisation).
    """
    return {str(i): g for g, i in gloss_to_id.items()}
