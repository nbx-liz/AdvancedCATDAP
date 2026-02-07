import pandas as pd
import numpy as np
import logging
from advanced_catdap.service.analyzer import AnalyzerService
from advanced_catdap.service.schema import AnalysisParams

# Setup logging
logging.basicConfig(level=logging.INFO)

def run_debug():
    # 1. Create Dummy Data (Strong Feature)
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'target': np.random.randint(0, 2, n),
        'weak_feat': np.random.randn(n),
    })
    # Strong feature: highly correlated with target
    df['strong_feat'] = df['target'] * 10 + np.random.randn(n)

    print("--- Created Dummy Data ---")
    
    # 2. Run Analyzer
    service = AnalyzerService()
    params = AnalysisParams(target_col='target', task_type='classification', top_k=5)
    
    print("--- Running Analysis ---")
    result = service.run_analysis(df, params)
    
    # 3. Inspect Raw Result
    print("\n[Raw Result] Feature Importances:")
    fi_data = result.feature_importances
    for fi in fi_data:
        print(fi)
        
    # 4. Simulate Dash App Logic
    if fi_data:
        # Pydantic v2 dump
        fi_dicts = [fi.model_dump(by_alias=True) for fi in fi_data]
        df_fi = pd.DataFrame(fi_dicts)
        
        print("\n[Dash Logic] DataFrame Head:")
        print(df_fi.head())
        
        col_map = {c.lower(): c for c in df_fi.columns}
        delta_col = col_map.get('delta_score', col_map.get('deltascore', 'Delta_Score'))
        
        print(f"\n[Dash Logic] Delta Col identified as: {delta_col}")
        
        if delta_col in df_fi.columns:
            print("[Dash Logic] Top Values:")
            print(df_fi[[col_map.get('feature', 'Feature'), delta_col]])

if __name__ == "__main__":
    run_debug()
