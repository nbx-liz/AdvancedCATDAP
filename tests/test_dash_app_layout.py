import plotly.graph_objects as go
from dash.development.base_component import Component

from advanced_catdap.frontend import dash_app as dash_mod


def _walk_components(node):
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _walk_components(child)
        return
    if isinstance(node, Component):
        yield node
        children = getattr(node, "children", None)
        if children is not None:
            yield from _walk_components(children)


def test_sidebar_has_expected_controls():
    sidebar = dash_mod.create_sidebar_content()
    ids = {getattr(c, "id", None) for c in _walk_components(sidebar)}
    assert "upload-data" in ids
    assert "upload-status" in ids
    assert "report-filename-input" in ids
    assert "btn-export-html" in ids


def test_render_dashboard_tab_empty_result():
    view = dash_mod.render_dashboard_tab(result=None, meta=None)
    assert "Please upload data and run analysis." in str(view)


def _full_result_payload():
    return {
        "mode": "classification",
        "baseline_score": 1000.0,
        "feature_importances": [
            {"Feature": "A", "Delta_Score": 120.0, "Score": 880.0},
            {"Feature": "B", "Delta_Score": 80.0, "Score": 920.0},
            {"Feature": "C", "Delta_Score": 30.0, "Score": 970.0},
        ],
        "interaction_importances": [
            {"Feature_1": "A", "Feature_2": "B", "Gain": 15.0},
            {"Feature_1": "B", "Feature_2": "C", "Gain": 8.0},
        ],
        "feature_details": {
            "A": {
                "bin_labels": ["L", "H"],
                "bin_counts": [10, 20],
                "bin_means": [0.1, 0.8],
            },
            "B": {
                "bin_labels": ["X", "Y"],
                "bin_counts": [15, 15],
                "bin_means": [0.2, 0.6],
            },
        },
        "interaction_details": {
            "A|B": {
                "feature_1": "A",
                "feature_2": "B",
                "bin_labels_1": ["L", "H"],
                "bin_labels_2": ["X", "Y"],
                "counts": [[3, 7], [12, 8]],
                "means": [[0.1, 0.2], [0.7, 0.9]],
            }
        },
    }


def test_render_dashboard_tab_with_data():
    result = _full_result_payload()
    meta = {"n_columns": 4}
    params = {"task_type": "classification", "use_aicc": True}
    view = dash_mod.render_dashboard_tab(result, meta, params)
    assert "Top Features by Impact" in str(view)
    assert "Interaction Network" in str(view)


def test_render_dashboard_tab_selected_features_uses_transform_rules():
    result = _full_result_payload()
    result["transform_rules"] = {"Churn": {"bins": [0, 1]}, "Target_Spend": {"bins": [0, 1]}}
    view = dash_mod.render_dashboard_tab(result, {"n_columns": 13}, {"task_type": "auto"})
    assert "2 / 12 features" in str(view)


def test_render_dashboard_tab_selected_features_falls_back_to_feature_importances():
    result = _full_result_payload()
    result.pop("transform_rules", None)
    view = dash_mod.render_dashboard_tab(result, {"n_columns": 4}, {"task_type": "auto"})
    assert "3 / 3 features" in str(view)


def test_render_dashboard_tab_shows_interaction_empty_reason():
    result = _full_result_payload()
    result["interaction_importances"] = []
    result["interaction_details"] = {}
    params = {"task_type": "auto", "use_aicc": True}
    view = dash_mod.render_dashboard_tab(result, {"n_columns": 4}, params)
    text = str(view)
    assert "Interaction Network is not available" in text
    assert "No interaction pair passed the gain threshold." in text
    assert "Auto task detection selected: Classification." in text
    assert "interaction_importances and interaction_details are empty." in text


def test_render_dashboard_tab_handles_infinity_baseline():
    result = _full_result_payload()
    result["baseline_score"] = "Infinity"
    result["feature_importances"] = [
        {"feature": "Age", "score": "Infinity", "delta_score": None, "actual_bins": 3, "method": "tree_2(3)"}
    ]
    view = dash_mod.render_dashboard_tab(result, {"n_columns": 4}, {"task_type": "auto"})
    as_text = str(view)
    assert "AIC Comparison" in as_text
    assert "Delta N/A" in as_text


def test_render_deepdive_tab_modes():
    result = _full_result_payload()
    deep_top = dash_mod.render_deepdive_tab(
        result=result,
        selected_mode="top5",
        selected_feature=None,
        theme="dark",
        meta={"n_columns": 4},
        target_col="target",
        selected_interaction_pair="A|B",
    )
    deep_select = dash_mod.render_deepdive_tab(
        result=result,
        selected_mode="select",
        selected_feature="A",
        theme="dark",
        meta={"n_columns": 4},
        target_col="target",
        selected_interaction_pair="A|B",
    )
    assert "Feature Analysis" in str(deep_top)
    assert "Bivariate Interaction Analysis" in str(deep_select)


def test_render_content_both_tabs():
    result = _full_result_payload()
    dashboard = dash_mod.render_content(
        "tab-dashboard",
        result,
        {"mode": "top5", "feature": None, "interaction": "A|B"},
        "dark",
        {"n_columns": 4},
        {"target_col": "target", "task_type": "auto"},
    )
    deepdive = dash_mod.render_content(
        "tab-deepdive",
        result,
        {"mode": "select", "feature": "A", "interaction": "A|B"},
        "dark",
        {"n_columns": 4},
        {"target_col": "target", "task_type": "auto"},
    )
    assert dashboard is not None
    assert deepdive is not None


def test_apply_chart_style_sets_template():
    fig = go.Figure()
    styled = dash_mod.apply_chart_style(fig)
    assert styled.layout.template is not None
    assert styled.layout.font.family == "Segoe UI"
