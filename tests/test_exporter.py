import io
import re
import pandas as pd
from advanced_catdap.service.exporter import ResultExporter

def test_html_export_interactive():
    result = {
        'mode': 'CLASSIFICATION',
        'baseline_score': 500.0,
        'feature_importances': [
            {'Feature': 'ColA', 'Delta_Score': 100.0, 'Score': 400.0},
            {'Feature': 'ColB', 'Delta_Score': 50.0, 'Score': 450.0}
        ],
        'interaction_importances': [
             {'Feature_1': 'ColA', 'Feature_2': 'ColB', 'Gain': 10.0}
        ],
        'feature_details': {
            'ColA': {
                'bin_counts': [10, 20],
                'bin_means': [0.5, 0.6],
                'bin_labels': ['Low', 'High']
            }
        },
        'interaction_details': {
            'ColA - ColB': {
                'feature_1': 'ColA', 'feature_2': 'ColB',
                'means': [[0.1, 0.2], [0.3, 0.4]],
                'counts': [[10, 20], [30, 40]],
                'bin_labels_1': ['Lo', 'Hi'], 'bin_labels_2': ['A', 'B']
            }
        }
    }
    meta = {'dataset_id': 'test_data', 'n_rows': 100, 'n_columns': 5}
    
    html_io = ResultExporter.generate_html_report(result, meta)
    assert isinstance(html_io, io.BytesIO)
    
    content = html_io.getvalue().decode('utf-8')
    assert "<!DOCTYPE html>" in content
    assert "AdvancedCATDAP Report" in content
    assert "ColA" in content
    assert "plotly" in content 
    assert "function showFeature(featName)" in content 
    assert "function toggleTheme()" in content # Theme toggle check
    assert "inttable_" in content # Interaction table check
    assert "ColA - ColB" in content


def test_normalize_feature_importances_with_mixed_columns():
    fi_data = [
        {"feature": "A", "delta_score": "4226.0", "score": "63517.0"},
        {"Feature": "B", "Delta_Score": 100.5, "Score": 67600},
        {"Feature": "C", "Delta_Score": "not-a-number", "Score": 1},
    ]
    df = ResultExporter.normalize_feature_importances(fi_data)
    assert list(df.columns) == ["Feature", "Delta_Score", "Score"]
    assert set(df["Feature"].tolist()) == {"A", "B"}
    assert df["Delta_Score"].dtype.kind in ("f", "i")


def test_html_export_uses_safe_dom_ids():
    result = {
        "mode": "CLASSIFICATION",
        "baseline_score": 100.0,
        "feature_importances": [{"Feature": "With Space", "Delta_Score": 10.0, "Score": 90.0}],
        "interaction_importances": [{"Feature_1": "A|B", "Feature_2": "C/D", "Gain": 1.0}],
        "feature_details": {
            "With Space": {"bin_counts": [1], "bin_means": [0.1], "bin_labels": ["x"]}
        },
        "interaction_details": {
            "A|B": {
                "feature_1": "A|B",
                "feature_2": "C/D",
                "means": [[0.1]],
                "counts": [[1]],
                "bin_labels_1": ["x"],
                "bin_labels_2": ["y"],
            }
        },
    }
    html_io = ResultExporter.generate_html_report(result, meta={})
    content = html_io.getvalue().decode("utf-8")
    assert 'id="chart_With Space"' not in content
    assert "chart_feat_" in content


def test_build_interaction_matrix_sums_gain():
    df = ResultExporter.normalize_interaction_importances([
        {"Feature_1": "A", "Feature_2": "B", "Gain": 2.0},
        {"feature_1": "A", "feature_2": "B", "gain": 3.0},
        {"Feature_1": "C", "Feature_2": "B", "Gain": 1.5},
    ])
    mat = ResultExporter.build_interaction_matrix(df)
    assert not mat.empty
    assert mat.loc["B", "A"] == 5.0
    assert mat.loc["B", "C"] == 1.5


def test_build_interaction_matrix_from_details_fallback():
    details = {
        "A|B": {
            "feature_1": "A",
            "feature_2": "B",
            "counts": [[10, 20], [5, 0]],
            "means": [[0.1, 0.2], [0.4, 0.0]],
        }
    }
    mat = ResultExporter.build_interaction_matrix_from_details(details)
    assert not mat.empty
    assert mat.loc["B", "A"] > 0


def test_normalize_interaction_importances_pair_score_fallback():
    df = ResultExporter.normalize_interaction_importances([
        {"feature_1": "A", "feature_2": "B", "pair_score": 7.5},
    ])
    assert not df.empty
    assert list(df.columns) == ["Feature_1", "Feature_2", "Gain"]
    assert df.iloc[0]["Feature_1"] == "A"
    assert df.iloc[0]["Feature_2"] == "B"
    assert float(df.iloc[0]["Gain"]) == 7.5


def test_normalize_interaction_importances_prefers_gain_over_pair_score():
    df = ResultExporter.normalize_interaction_importances([
        {"Feature_1": "A", "Feature_2": "B", "gain": 2.0, "pair_score": 9.0},
    ])
    assert not df.empty
    assert float(df.iloc[0]["Gain"]) == 2.0


def test_html_export_pair_score_only_renders_global_heatmap():
    result = {
        "mode": "CLASSIFICATION",
        "baseline_score": 10.0,
        "feature_importances": [{"Feature": "A", "Delta_Score": 1.0, "Score": 9.0}],
        "interaction_importances": [{"feature_1": "A", "feature_2": "B", "pair_score": 3.0}],
        "feature_details": {},
        "interaction_details": {},
    }
    html_io = ResultExporter.generate_html_report(result, meta={})
    content = html_io.getvalue().decode("utf-8")
    assert "Global Interaction Network" in content
    assert '"type":"heatmap"' in content
    assert "Gain=%{z:.6g}" in content


def test_html_export_global_heatmap_uses_plain_json_z_not_bdata():
    result = {
        "mode": "CLASSIFICATION",
        "baseline_score": 10.0,
        "feature_importances": [{"Feature": "A", "Delta_Score": 1.0, "Score": 9.0}],
        "interaction_importances": [
            {"feature_1": "A", "feature_2": "B", "pair_score": 3.0},
            {"feature_1": "A", "feature_2": "C", "pair_score": 1.0},
        ],
        "feature_details": {},
        "interaction_details": {},
    }
    html_io = ResultExporter.generate_html_report(result, meta={})
    content = html_io.getvalue().decode("utf-8")
    title_idx = content.find("Global Interaction Network")
    assert title_idx >= 0
    newplot_idx = content.rfind("Plotly.newPlot(", 0, title_idx)
    assert newplot_idx >= 0
    section = content[newplot_idx:title_idx + 2000]
    assert '"type":"heatmap"' in section
    assert '"bdata"' not in section


def test_html_export_interaction_missing_metric_shows_message():
    result = {
        "mode": "CLASSIFICATION",
        "baseline_score": 10.0,
        "feature_importances": [{"Feature": "A", "Delta_Score": 1.0, "Score": 9.0}],
        "interaction_importances": [{"feature_1": "A", "feature_2": "B"}],
        "feature_details": {},
        "interaction_details": {},
    }
    html_io = ResultExporter.generate_html_report(result, meta={})
    content = html_io.getvalue().decode("utf-8")
    assert "No interaction data available" in content
    assert "Pair_Score not found" in content


def test_html_export_interaction_details_without_counts_does_not_crash():
    result = {
        "mode": "CLASSIFICATION",
        "baseline_score": 10.0,
        "feature_importances": [{"Feature": "A", "Delta_Score": 1.0, "Score": 9.0}],
        "interaction_importances": [],
        "feature_details": {},
        "interaction_details": {
            "A|B": {
                "feature_1": "A",
                "feature_2": "B",
                "means": [[0.1, 0.2], [0.3, 0.4]],
                "bin_labels_1": ["L", "H"],
                "bin_labels_2": ["L", "H"],
            }
        },
    }
    html_io = ResultExporter.generate_html_report(result, meta={})
    content = html_io.getvalue().decode("utf-8")
    assert "Interaction: A vs B" in content
    assert "Sample Count Matrix" in content


def test_resolve_column_case_insensitive_and_missing():
    df = pd.DataFrame({"PAIR_SCORE": [1.0], "gain": [2.0]})
    assert ResultExporter._resolve_column(df, ["pair_score"]) == "PAIR_SCORE"
    assert ResultExporter._resolve_column(df, ["missing_col"]) is None


def test_format_compact_branches():
    assert ResultExporter._format_compact("x") == "x"
    assert ResultExporter._format_compact(2_000_000) == "2M"
    assert ResultExporter._format_compact(2_000) == "2k"
    assert ResultExporter._format_compact(123.456) == "123.5"
    assert ResultExporter._format_compact(12.3456) == "12.3"
    assert ResultExporter._format_compact(0.123456) == "0.1235"


def test_normalize_importances_empty_inputs():
    assert ResultExporter.normalize_feature_importances([]).empty
    assert ResultExporter.normalize_interaction_importances([]).empty


def test_html_export_feature_labels_from_edges_and_fallback_bins():
    result = {
        "mode": "CLASSIFICATION",
        "baseline_score": 10.0,
        "feature_importances": [{"Feature": "A", "Delta_Score": 1.0, "Score": 9.0}],
        "interaction_importances": [],
        "feature_details": {
            "A": {"bin_counts": [1, 2], "bin_means": [0.1, 0.2], "bin_edges": [0.0, 1.0, 2.0]},
            "B": {"bin_counts": [3, 4], "bin_means": [0.3, 0.4]},
        },
        "interaction_details": {},
    }
    html_io = ResultExporter.generate_html_report(result, meta={})
    content = html_io.getvalue().decode("utf-8")
    assert "[0.00, 1.00)" in content
    assert "Bin 0" in content
