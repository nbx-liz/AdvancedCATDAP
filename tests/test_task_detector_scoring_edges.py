import numpy as np
import pandas as pd

from advanced_catdap.components.task_detector import TaskDetector
from advanced_catdap.components.scoring import Scorer


def test_task_detector_auto_string_target_is_classification():
    det = TaskDetector(task_type="auto")
    s = pd.Series(["a", "b", "a"])
    assert det.detect(s) == "classification"


def test_task_detector_calc_aic_returns_inf_on_small_n():
    det = TaskDetector(task_type="auto")
    score = det._calc_aic_score(term_val=1.0, k=3, n=4, is_regression=True)
    assert np.isinf(score)


def test_scorer_cls_bincount_returns_inf_when_memory_guard_trips():
    scorer = Scorer(max_classification_bytes=1)
    target_int = np.array([0, 1, 0, 1], dtype=np.int64)
    indices = np.array([0, 1, 0, 1], dtype=np.int64)
    score, k = scorer.calc_score_cls_bincount_idx(
        target_int=target_int,
        n_classes=2,
        indices=indices,
        minlength=10,
        check_memory=True,
    )
    assert np.isinf(score)
    assert k == 0


def test_scorer_cls_partial_returns_inf_when_memory_guard_trips():
    scorer = Scorer(max_classification_bytes=1)
    score = scorer.calc_score_classification_partial(
        codes_valid=np.array([0, 1], dtype=np.int64),
        target_int_valid=np.array([0, 1], dtype=np.int64),
        n_classes=2,
        r=10,
        stats_missing={"n": 0, "loglik_part": 0.0},
        n_total=2,
    )
    assert np.isinf(score)
