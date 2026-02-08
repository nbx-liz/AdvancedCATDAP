import numpy as np
import pandas as pd

from advanced_catdap.components.discretizer import DiscretizationStrategy, Discretizer
from advanced_catdap.components.scoring import Scorer


def _make_discretizer(**kwargs):
    return Discretizer(task_type="classification", scorer=Scorer(), **kwargs)


def test_group_categorical_ordered_category_order_keeps_contiguous_block():
    d = _make_discretizer(min_cat_fraction=0.2, max_categories=10)
    raw = pd.Series(["low"] * 8 + ["mid"] + ["high"] * 8 + ["other"] * 3)
    target = np.array([0, 1] * 10)
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)

    codes, _score, _r, method, rule = d._group_categorical(
        raw_series=raw,
        target=target,
        target_sq=target_sq,
        target_int=target_int,
        n_classes=2,
        sample_indices=np.arange(len(raw)),
        baseline_score=100.0,
        ordered=True,
        category_order=["low", "mid", "high"],
    )

    assert method.startswith("category(")
    # "mid" is below frequency threshold, but should be pulled in to keep a contiguous ordered block.
    assert "mid" in rule["value_map"]
    assert len(codes) == len(raw)


def test_group_categorical_nominal_order_is_freq_then_label():
    d = _make_discretizer(min_cat_fraction=0.0, max_categories=10)
    raw = pd.Series(["C"] * 5 + ["B"] * 3 + ["A"] * 3)
    target = np.array([0, 1] * 5 + [0])
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)

    _codes, _score, _r, _method, rule = d._group_categorical(
        raw_series=raw,
        target=target,
        target_sq=target_sq,
        target_int=target_int,
        n_classes=2,
        sample_indices=np.arange(len(raw)),
        baseline_score=100.0,
        ordered=False,
        category_order=None,
    )

    # freq: C(5) first, then A/B tie(3) sorted by label -> A then B
    assert list(rule["value_map"].keys()) == ["C", "A", "B"]


def test_transform_single_feature_integer_levels_handles_other_and_missing():
    d = _make_discretizer()
    rule = {
        "type": "integer_levels",
        "value_map": {1: 0, 2: 1},
        "other_code": 2,
        "missing_code": 3,
    }
    series = pd.Series([1, 2, 5, None])
    codes = d._transform_single_feature(series, rule)
    assert codes.tolist() == [0, 1, 2, 3]


def test_get_bin_label_integer_levels_fallback_label():
    d = _make_discretizer()
    rule = {
        "type": "integer_levels",
        "value_map": {1: 0},
        "other_code": 2,
        "missing_code": 3,
    }
    assert d._get_bin_label(99, rule) == "Level_99"


def test_get_feature_details_recomputes_codes_when_cached_length_mismatch():
    d = _make_discretizer()
    d.transform_rules_ = {
        "f": {
            "type": "category",
            "value_map": {"x": 0, "y": 1},
            "other_code": 2,
            "missing_code": 3,
        }
    }
    # Intentionally wrong cached size to force recalc branch.
    d.processed_codes_ = {"f": np.array([0, 1])}

    df = pd.DataFrame({"f": ["x", "y", "z", None]})
    target = np.array([1, 0, 1, 0])

    details = d.get_feature_details(df, target)
    assert not details.empty
    assert details["Count"].sum() == 4


def test_transform_single_feature_tree_uses_leaf_positions():
    d = _make_discretizer()

    class _FakeTree:
        def apply(self, x):
            return np.array([20 if v[0] > 0 else 10 for v in x], dtype=np.int64)

    rule = {
        "type": "tree",
        "model": _FakeTree(),
        "leaves": np.array([10, 20], dtype=np.int64),
        "missing_code": 2,
    }
    series = pd.Series([-1.0, 5.0, None])
    codes = d._transform_single_feature(series, rule)
    assert codes.tolist() == [0, 1, 2]


def test_transform_returns_empty_dataframe_when_rules_absent():
    d = _make_discretizer()
    out = d.transform(pd.DataFrame({"a": [1, 2, 3]}))
    assert list(out.columns) == []
    assert list(out.index) == [0, 1, 2]


def test_transform_cut_rule_executes_searchsorted_branch():
    d = _make_discretizer()
    d.transform_rules_ = {"a": {"type": "cut", "bins": np.array([-np.inf, 0.0, 10.0, np.inf]), "missing_code": 3}}
    out = d.transform(pd.DataFrame({"a": [-1.0, 5.0, 15.0, None]}))
    assert out["a"].tolist() == [0, 1, 2, 3]


def test_should_prefix_labels_styles():
    d = _make_discretizer(label_prefix_style="none")
    assert d._should_prefix_labels("tree") is False
    d.label_prefix_style = "all_bins"
    assert d._should_prefix_labels("category") is True
    d.label_prefix_style = "unknown_style"
    assert d._should_prefix_labels("tree") is False


def test_build_sort_rank_map_handles_nan_min_value():
    d = _make_discretizer()
    rank_map = d._build_sort_rank_map(
        code_order=[0, 1, 2],
        rule={"type": "tree", "missing_code": 2},
        code_min_map={0: np.nan, 1: 1.0},
    )
    # code 1 should be first because code 0 has NaN (treated as +inf)
    assert rank_map[1] == 1
    assert rank_map[0] == 2
    assert rank_map[2] == 3


def test_build_bin_label_and_key_fallback_without_rank_map():
    d = _make_discretizer()
    rule = {"type": "tree", "missing_code": 4, "other_code": 3}
    k0, _ = d._build_bin_label_and_key(0, rule, total_bins=5, rank_map=None)
    ko, _ = d._build_bin_label_and_key(3, rule, total_bins=5, rank_map=None)
    km, _ = d._build_bin_label_and_key(4, rule, total_bins=5, rank_map=None)
    assert k0 == "01"
    assert ko > k0
    assert km > ko


def test_get_feature_details_uses_transform_when_no_cached_codes():
    d = _make_discretizer()
    d.transform_rules_ = {
        "f": {
            "type": "category",
            "value_map": {"x": 0},
            "other_code": 1,
            "missing_code": 2,
        }
    }
    d.processed_codes_ = {}
    df = pd.DataFrame({"f": ["x", "z", None]})
    target = np.array([1, 0, 1])
    details = d.get_feature_details(df, target)
    assert details["Count"].sum() == 3


def test_group_categorical_ordered_with_nonmatching_order_falls_back_to_sort():
    d = _make_discretizer(min_cat_fraction=0.0, max_categories=10)
    raw = pd.Series(["c", "a", "b", "a", "b", "c"])
    target = np.array([0, 1, 0, 1, 0, 1])
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)

    _codes, _score, _r, _method, rule = d._group_categorical(
        raw_series=raw,
        target=target,
        target_sq=target_sq,
        target_int=target_int,
        n_classes=2,
        sample_indices=np.arange(len(raw)),
        baseline_score=100.0,
        ordered=True,
        category_order=["x", "y"],  # no overlap
    )
    assert set(rule["value_map"].keys()) == {"a", "b", "c"}


def test_group_categorical_ordered_flag_without_order_uses_index_sort():
    d = _make_discretizer(min_cat_fraction=0.0, max_categories=10)
    raw = pd.Series(["c", "a", "b", "a", "b", "c"])
    target = np.array([0, 1, 0, 1, 0, 1])
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)

    _codes, _score, _r, _method, rule = d._group_categorical(
        raw_series=raw,
        target=target,
        target_sq=target_sq,
        target_int=target_int,
        n_classes=2,
        sample_indices=np.arange(len(raw)),
        baseline_score=100.0,
        ordered=True,
        category_order=None,
    )
    # first-seen order in input is c, a, b
    assert list(rule["value_map"].keys()) == ["c", "a", "b"]


def test_group_categorical_with_ordered_categorical_dtype_branch():
    d = _make_discretizer(min_cat_fraction=0.0, max_categories=10)
    raw = pd.Series(
        pd.Categorical(
            ["mid", "high", "low", "mid", "high", "low"],
            categories=["low", "mid", "high"],
            ordered=True,
        )
    )
    target = np.array([0, 1, 0, 1, 0, 1])
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)

    _codes, _score, _r, _method, rule = d._group_categorical(
        raw_series=raw,
        target=target,
        target_sq=target_sq,
        target_int=target_int,
        n_classes=2,
        sample_indices=np.arange(len(raw)),
        baseline_score=100.0,
        ordered=True,
        category_order=None,
    )
    assert set(rule["value_map"].keys()) == {"low", "mid", "high"}
    assert rule.get("ordered") is True


def test_get_bin_label_category_multi_and_fallback_and_integer_other_and_tree_fallback():
    d = _make_discretizer()
    category_rule = {
        "type": "category",
        "value_map": {"x1": 0, "x2": 0, "x3": 0, "x4": 0},
        "other_code": 9,
        "missing_code": 10,
    }
    label_multi = d._get_bin_label(0, category_rule)
    assert label_multi.endswith("...")
    assert d._get_bin_label(123, category_rule).startswith("Cat_")

    int_rule = {"type": "integer_levels", "value_map": {1: 0}, "other_code": 7, "missing_code": 8}
    assert d._get_bin_label(7, int_rule) == "Other"

    tree_rule = {"type": "tree", "missing_code": 9}
    assert d._get_bin_label(1, tree_rule).startswith("TreeLeaf_")

    unknown_rule = {"type": "mystery", "missing_code": 9}
    assert d._get_bin_label(1, unknown_rule).startswith("Bin_")


def test_transform_single_feature_unknown_rule_returns_missing_code():
    d = _make_discretizer()
    rule = {"type": "unknown", "missing_code": 4}
    codes = d._transform_single_feature(pd.Series([1, 2, None]), rule)
    assert codes.tolist() == [4, 4, 4]


def test_discretization_strategy_base_pass_branch():
    class _BasePassStrategy(DiscretizationStrategy):
        def discretize(self, vals_sample, n_bins, min_samples, task_type, y_sample=None):
            return super().discretize(vals_sample, n_bins, min_samples, task_type, y_sample)

    s = _BasePassStrategy()
    assert s.discretize(np.array([1.0, 2.0]), 2, 1, "classification", None) is None


def test_discretize_numeric_uses_small_valid_sample_branch():
    d = _make_discretizer()
    values = np.array([1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)
    sample_mask = np.ones(len(values), dtype=bool)
    sample_idx = np.arange(len(values))

    class _OkStrategy:
        def discretize(self, vals_sample, n_bins, min_samples, task_type, y_sample=None):
            if n_bins != 2:
                return None, 1, None
            n = len(vals_sample)
            codes = np.zeros(n, dtype=int)
            codes[n // 2 :] = 1
            return codes, 2, {"type": "cut", "bins": np.array([-np.inf, 5.0, np.inf]), "missing_code": 2}

    d.strategies = {"ok": _OkStrategy()}
    out_codes, _score, r, method, rule = d._discretize_numeric(
        values,
        target,
        target_sq,
        target_int,
        n_classes=2,
        max_bins=2,
        min_samples=1,
        sample_indices=sample_idx,
        sample_mask_full=sample_mask,
        baseline_score=10.0,
        feature_name="num",
    )
    assert out_codes is not None
    assert r >= 2
    assert method == "ok_2(2)"
    assert rule["type"] == "cut"


def test_discretize_numeric_unexpected_exception_path_returns_no_candidates():
    d = _make_discretizer()
    values = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([0, 1, 0, 1])
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)
    sample_mask = np.ones(len(values), dtype=bool)
    sample_idx = np.arange(len(values))

    class _BoomStrategy:
        def discretize(self, vals_sample, n_bins, min_samples, task_type, y_sample=None):
            raise RuntimeError("boom")

    d.strategies = {"boom": _BoomStrategy()}
    out_codes, score, r, method, rule = d._discretize_numeric(
        values,
        target,
        target_sq,
        target_int,
        n_classes=2,
        max_bins=3,
        min_samples=1,
        sample_indices=sample_idx,
        sample_mask_full=sample_mask,
        baseline_score=10.0,
        feature_name="num",
    )
    assert out_codes is None
    assert score == 10.0
    assert r == 1
    assert method == "no_candidates"
    assert rule is None


def test_group_integer_levels_all_nan_branch():
    d = _make_discretizer()
    values = np.array([np.nan, np.nan, np.nan], dtype=float)
    target = np.array([0, 1, 0], dtype=int)
    target_sq = target.astype(float) ** 2
    target_int = target.astype(int)
    codes, score, r, method, rule = d._group_integer_levels(
        values, target, target_sq, target_int, n_classes=2, baseline_score=5.0
    )
    assert codes is None
    assert score == 5.0
    assert r == 1
    assert method == "all_nan"
    assert rule is None


def test_transform_single_feature_cut_and_qcut_helper_branch():
    d = _make_discretizer()
    rule_cut = {"type": "cut", "bins": np.array([-np.inf, 0.0, 10.0, np.inf]), "missing_code": 3}
    rule_qcut = {"type": "qcut", "bins": np.array([-np.inf, 5.0, np.inf]), "missing_code": 2}
    s = pd.Series([-1.0, 5.0, 15.0, None])
    out_cut = d._transform_single_feature(s, rule_cut)
    out_qcut = d._transform_single_feature(s, rule_qcut)
    assert out_cut.tolist() == [0, 1, 2, 3]
    assert out_qcut.tolist() == [0, 1, 1, 2]
