import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from advanced_catdap import AdvancedCATDAP
from advanced_catdap.examples.utils import generate_crm_data, generate_ltv_data

def run_classification_demo():
    print("\n=== Classification Demo (Pandas API) ===")
    
    # 1. Load Data (Churn)
    df = generate_crm_data(n_samples=1000)
    print(f"Data Loaded: {len(df)} rows. Target: 'Churn' (Mean: {df['Churn'].mean():.2%})")
    
    # 2. Initialize
    model = AdvancedCATDAP(
        verbose=True, 
        task_type='classification',
        save_rules_mode='top_k'
    )
    
    # 3. Analyze (Pandas-style)
    # Detects feature importance and interactions
    importances, interactions = model.analyze(df, target_col='Churn')
    
    print("\n--- Top Features by AIC Reduction ---")
    print(importances[['Feature', 'Score', 'Delta_Score', 'Method']].head(5))
    
    print("\n--- Top Interactions (Pairs) ---")
    print(interactions.head(5))
    
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
    run_classification_demo()

