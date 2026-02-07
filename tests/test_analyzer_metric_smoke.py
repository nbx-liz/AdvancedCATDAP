import numpy as np
import pandas as pd

from advanced_catdap.service.analyzer import AnalyzerService
from advanced_catdap.service.schema import AnalysisParams


def test_analyzer_metric_smoke_strong_feature_ranked_higher():
    """Smoke test migrated from scripts/debug_metric.py."""
    np.random.seed(42)
    n = 1000
    target = np.random.randint(0, 2, n)
    df = pd.DataFrame(
        {
            "target": target,
            "weak_feat": np.random.randn(n),
            "strong_feat": target * 10 + np.random.randn(n),
        }
    )

    service = AnalyzerService()
    params = AnalysisParams(target_col="target", task_type="classification", top_k=5)
    result = service.run_analysis(df, params)

    fi_dicts = [fi.model_dump(by_alias=True) for fi in result.feature_importances]
    df_fi = pd.DataFrame(fi_dicts)
    assert not df_fi.empty

    col_map = {c.lower(): c for c in df_fi.columns}
    feature_col = col_map.get("feature", "Feature")
    delta_col = col_map.get("delta_score", "Delta_Score")
    assert feature_col in df_fi.columns
    assert delta_col in df_fi.columns

    score_map = dict(zip(df_fi[feature_col], df_fi[delta_col]))
    assert "strong_feat" in score_map
    assert "weak_feat" in score_map
    assert float(score_map["strong_feat"]) > float(score_map["weak_feat"])
