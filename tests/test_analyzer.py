import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from advanced_catdap.service.analyzer import AnalyzerService
from advanced_catdap.service.schema import AnalysisParams


class _StubDiscretizer:
    def _transform_single_feature(self, series, _rule):
        codes, _ = pd.factorize(series.astype(str), sort=False)
        return codes

    def get_axis_metadata(self, _raw_series, codes, _rule):
        uniq = sorted(set(int(v) for v in codes))
        labels = [f"{i+1:02d}_{u}" for i, u in enumerate(uniq)]
        sort_keys = [f"{i+1:02d}" for i in range(len(uniq))]
        return uniq, labels, sort_keys


class _FixedOrderStubDiscretizer:
    def _transform_single_feature(self, series, _rule):
        # Produce 5 bins regardless of raw values
        return (pd.Series(range(len(series))) % 5).to_numpy()

    def get_axis_metadata(self, _raw_series, _codes, _rule):
        code_order = [0, 1, 2, 3, 4]
        sort_keys = ["05", "03", "04", "01", "02"]
        labels = [
            "05_[74.000, 84.000]",
            "03_[29.000, 64.000]",
            "04_[65.000, 73.000]",
            "01_[18.000, 22.000]",
            "02_[23.000, 28.000]",
        ]
        return code_order, labels, sort_keys

def test_analyzer_flow():
    # Mock the Core AdvancedCATDAP to avoid heavy computation
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance
        
        # Mock attributes
        mock_instance.mode = "classification"
        mock_instance.baseline_score = 100.0
        mock_instance.transform_rules_ = {
            "col1": {"type": "category"},
            "col2": {"type": "category"},
        }
        mock_instance.discretizer = _StubDiscretizer()
        mock_instance.feature_importances_ = pd.DataFrame([{
            "Feature": "col1", "Score": 50, "Delta_Score": 10, "Actual_Bins": 2, "Method": "tree"
        }])
        mock_instance.interaction_importances_ = None
        mock_instance.feature_details_ = pd.DataFrame({
            "Feature": ["col1"],
            "Bin_Idx": [0],
            "Bin_Label": ["Bin0"],
            "Count": [10],
            "Target_Mean": [0.1]
        })
        
        service = AnalyzerService()
        df = pd.DataFrame({"target": [0, 1], "col1": [1, 2]})
        params = AnalysisParams(target_col="target")
        
        result = service.run_analysis(df, params)
        
        assert result.mode == "classification"
        assert result.task_type == "classification"
        assert len(result.feature_importances) == 1
        assert result.feature_importances[0].feature == "col1"
        assert "col1" in result.feature_details


def test_analyzer_error_handling():
    """Test analyzer error path (lines 49-51)."""
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance
        mock_instance.fit.side_effect = ValueError("Test error")
        
        service = AnalyzerService()
        df = pd.DataFrame({"target": [0, 1], "col1": [1, 2]})
        params = AnalysisParams(target_col="target")
        
        with pytest.raises(ValueError, match="Test error"):
            service.run_analysis(df, params)


def test_analyzer_with_interaction_importances():
    """Test analyzer with interaction importances (lines 79-87, 110-161)."""
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance
        
        mock_instance.mode = "classification"
        mock_instance.baseline_score = 100.0
        mock_instance.transform_rules_ = {
            "col1": {"type": "category"},
            "col2": {"type": "category"},
        }
        mock_instance.discretizer = _StubDiscretizer()
        mock_instance.feature_importances_ = pd.DataFrame([
            {"Feature": "col1", "Score": 50, "Delta_Score": 10, "Actual_Bins": 2, "Method": "tree"},
            {"Feature": "col2", "Score": 40, "Delta_Score": 5, "Actual_Bins": 3, "Method": "tree"}
        ])
        mock_instance.interaction_importances_ = pd.DataFrame([{
            "Feature_1": "col1", "Feature_2": "col2", "Pair_Score": 80, "Gain": 15
        }])
        mock_instance.feature_details_ = pd.DataFrame({
            "Feature": ["col1", "col2"],
            "Bin_Idx": [0, 0],
            "Bin_Label": ["Bin0", "Bin0"],
            "Count": [10, 10],
            "Target_Mean": [0.1, 0.2]
        })
        
        service = AnalyzerService()
        df = pd.DataFrame({
            "target": [0, 1, 0, 1] * 10,
            "col1": list(range(40)),
            "col2": list(range(40))
        })
        params = AnalysisParams(target_col="target")
        
        result = service.run_analysis(df, params)
        
        assert len(result.interaction_importances) == 1
        assert result.interaction_importances[0].feature_1 == "col1"
        assert result.interaction_importances[0].feature_2 == "col2"
        assert "col1|col2" in result.interaction_details


def test_analyzer_interaction_with_categorical():
    """Test interaction details with categorical columns."""
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance
        
        mock_instance.mode = "classification"
        mock_instance.baseline_score = 100.0
        mock_instance.transform_rules_ = {
            "cat1": {"type": "category"},
            "cat2": {"type": "category"},
        }
        mock_instance.discretizer = _StubDiscretizer()
        mock_instance.feature_importances_ = pd.DataFrame([
            {"Feature": "cat1", "Score": 50, "Delta_Score": 10, "Actual_Bins": 2, "Method": "category"},
            {"Feature": "cat2", "Score": 40, "Delta_Score": 5, "Actual_Bins": 3, "Method": "category"}
        ])
        mock_instance.interaction_importances_ = pd.DataFrame([{
            "Feature_1": "cat1", "Feature_2": "cat2", "Pair_Score": 80, "Gain": 15
        }])
        mock_instance.feature_details_ = None
        
        service = AnalyzerService()
        df = pd.DataFrame({
            "target": [0, 1, 0, 1] * 5,
            "cat1": ["A", "B"] * 10,
            "cat2": ["X", "Y", "Z", "X", "Y"] * 4
        })
        params = AnalysisParams(target_col="target")
        
        result = service.run_analysis(df, params)
        
        assert len(result.interaction_importances) == 1
        assert "cat1|cat2" in result.interaction_details


def test_analyzer_rejects_id_like_string_target():
    service = AnalyzerService()
    df = pd.DataFrame(
        {
            "CustomerID": [f"CUST_{i:06d}" for i in range(150)],
            "feature_num": list(range(150)),
        }
    )
    params = AnalysisParams(target_col="CustomerID")

    with pytest.raises(ValueError, match="appears ID-like"):
        service.run_analysis(df, params)


def test_analyzer_allows_non_id_like_string_target():
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance
        mock_instance.mode = "classification"
        mock_instance.baseline_score = 10.0
        mock_instance.transform_rules_ = {
            "cat1": {"type": "category"},
            "cat2": {"type": "category"},
        }
        mock_instance.discretizer = _StubDiscretizer()
        mock_instance.feature_importances_ = pd.DataFrame(columns=["Feature", "Score", "Delta_Score", "Actual_Bins", "Method"])
        mock_instance.interaction_importances_ = None
        mock_instance.feature_details_ = None

        service = AnalyzerService()
        df = pd.DataFrame(
            {
                "target": ["Yes", "No"] * 80,
                "feature_num": list(range(160)),
            }
        )
        params = AnalysisParams(target_col="target")

        result = service.run_analysis(df, params)
        assert result.mode == "classification"


def test_analyzer_string_classification_interaction_uses_class_purity():
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance

        mock_instance.mode = "classification"
        mock_instance.baseline_score = 100.0
        mock_instance.transform_rules_ = {
            "cat1": {"type": "category"},
            "cat2": {"type": "category"},
        }
        mock_instance.discretizer = _StubDiscretizer()
        mock_instance.feature_importances_ = pd.DataFrame([
            {"Feature": "cat1", "Score": 50, "Delta_Score": 10, "Actual_Bins": 2, "Method": "category"},
            {"Feature": "cat2", "Score": 40, "Delta_Score": 5, "Actual_Bins": 3, "Method": "category"},
        ])
        mock_instance.interaction_importances_ = pd.DataFrame([
            {"Feature_1": "cat1", "Feature_2": "cat2", "Pair_Score": 80, "Gain": 15}
        ])
        mock_instance.feature_details_ = None

        service = AnalyzerService()
        df = pd.DataFrame(
            {
                "target": ["Male", "Female", "Female", "Male", "Female", "Male"] * 10,
                "cat1": ["A", "B", "A", "B", "A", "B"] * 10,
                "cat2": ["X", "X", "Y", "Y", "Z", "Z"] * 10,
            }
        )
        params = AnalysisParams(target_col="target")

        result = service.run_analysis(df, params)

        det = result.interaction_details["cat1|cat2"]
        assert det.metric_name == "Class Purity"
        assert det.dominant_labels is not None
        assert det.bin_sort_keys_1 is not None
        assert det.bin_sort_keys_2 is not None
        assert det.bin_display_labels_1 is not None
        assert det.bin_display_labels_2 is not None
        for row in det.means:
            for v in row:
                assert 0.0 <= float(v) <= 1.0


def test_analyzer_preserves_discretizer_axis_sort_keys():
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance
        mock_instance.mode = "classification"
        mock_instance.baseline_score = 100.0
        mock_instance.transform_rules_ = {"cat1": {"type": "category"}, "cat2": {"type": "category"}}
        mock_instance.discretizer = _FixedOrderStubDiscretizer()
        mock_instance.feature_importances_ = pd.DataFrame(
            [
                {"Feature": "cat1", "Score": 50, "Delta_Score": 10, "Actual_Bins": 2, "Method": "category"},
                {"Feature": "cat2", "Score": 40, "Delta_Score": 5, "Actual_Bins": 3, "Method": "category"},
            ]
        )
        mock_instance.interaction_importances_ = pd.DataFrame(
            [{"Feature_1": "cat1", "Feature_2": "cat2", "Pair_Score": 80, "Gain": 15}]
        )
        mock_instance.feature_details_ = None

        service = AnalyzerService()
        df = pd.DataFrame(
            {
                "target": ["Male", "Female", "Female", "Male", "Female", "Male"] * 10,
                "cat1": ["A", "B", "A", "B", "A", "B"] * 10,
                "cat2": ["X", "X", "Y", "Y", "Z", "Z"] * 10,
            }
        )
        result = service.run_analysis(df, AnalysisParams(target_col="target"))
        det = result.interaction_details["cat1|cat2"]
        assert det.bin_sort_keys_1 == ["05", "03", "04", "01", "02"]
        assert det.bin_sort_keys_2 == ["05", "03", "04", "01", "02"]
        assert det.bin_display_labels_1[3].startswith("01_")

