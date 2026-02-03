import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_catdap import AdvancedCATDAP
from utils import generate_crm_data, generate_ltv_data



def run_regression_demo():

    print("\n=== Regression Demo (Pandas API) ===")
    
    # 1. Load Data (LTV)
    df = generate_ltv_data(n_samples=1000)
    print(f"Data Loaded: {len(df)} rows. Target: 'LTV'")
    
    # 2. Analyze
    model = AdvancedCATDAP(task_type='regression', verbose=True)
    results, combo = model.analyze(df, target_col='LTV')
    
    print("\n--- Top Features by AIC Reduction ---")
    print(results[['Feature', 'Score', 'Delta_Score', 'Method']].head(5))

    print("\n--- Top Interactions (Pairs) ---")
    print(combo.head(5))

    # 4. Transform (Apply rules)
    df_transformed = model.transform(df)
    print("\n--- Transformed Data (Discretized) ---")
    print(df_transformed.iloc[:5, :5])

    print("\n--- Feature Details (Impact Analysis) ---")
    if model.feature_details_ is not None:
        max_rows = 100
        details = model.feature_details_[['Feature', 'Bin_Label', 'Count', 'Target_Mean']]
        if len(details) <= max_rows:
            print(details)
        else:
            print(details.head(10))
            print(f"... ({len(details)-10} more rows hidden, max_rows={max_rows})")


if __name__ == "__main__":
    run_regression_demo()
