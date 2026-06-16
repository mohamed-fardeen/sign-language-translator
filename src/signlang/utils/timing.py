from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def timer(name: str = "block") -> Generator[dict, None, None]:
    state: dict = {"name": name, "elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed_ms"] = (time.perf_counter() - start) * 1000.0