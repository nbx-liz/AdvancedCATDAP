import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from advanced_catdap.service.analyzer import AnalyzerService
from advanced_catdap.service.schema import AnalysisParams

def test_analyzer_flow():
    # Mock the Core AdvancedCATDAP to avoid heavy computation
    with patch("advanced_catdap.service.analyzer.AdvancedCATDAP") as mock_core_cls:
        mock_instance = MagicMock()
        mock_core_cls.return_value = mock_instance
        
        # Mock attributes
        mock_instance.mode = "classification"
        mock_instance.baseline_score = 100.0
        mock_instance.transform_rules_ = {}
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
        mock_instance.transform_rules_ = {}
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
        mock_instance.transform_rules_ = {}
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
        mock_instance.transform_rules_ = {}
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

