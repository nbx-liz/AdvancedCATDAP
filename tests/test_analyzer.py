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
        assert len(result.feature_importances) == 1
        assert result.feature_importances[0].feature == "col1"
        assert "col1" in result.feature_details
