from __future__ import annotations

from signlang.evaluation.metrics import cer, class_distribution, topk_accuracy
from signlang.evaluation.reports import classification_report


def test_topk_accuracy() -> None:
    assert topk_accuracy([1, 2, 3], [1, 2, 4], k=1) == 2 / 3


def test_cer_perfect() -> None:
    assert cer([[1, 2, 3]], [[1, 2, 3]]) == 0.0


def test_cer_with_errors() -> None:
    assert cer([[1, 2, 3]], [[1, 2, 4]]) == 1 / 3


def test_class_distribution() -> None:
    d = class_distribution([1, 1, 2, 3, 3, 3])
    assert d == {1: 2, 2: 1, 3: 3}


def test_classification_report_keys() -> None:
    rep = classification_report([1, 1, 2, 3], [1, 2, 2, 3], label_names={1: "a", 2: "b", 3: "c"})
    assert rep["accuracy"] == 0.75
    assert {"precision", "recall", "f1"} <= set(rep["per_class"][0].keys())
