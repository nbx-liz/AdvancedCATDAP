import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from advanced_catdap import AdvancedCATDAP, plot_importance, plot_interaction_heatmap
from advanced_catdap.examples.utils import generate_crm_data

def run_churn_analysis():
    print("\n=== Customer Churn Analysis Demo ===")
    
    # 1. Realistic CRM Data
    df = generate_crm_data(n_samples=5000)
    
    print("\n[Step 1] Initializing Model...")
    model = AdvancedCATDAP(
        verbose=True, 
        task_type='classification',
        save_rules_mode='top_k',
        n_interaction_candidates=50 # Speed up for demo
    )
    
    print("\n[Step 2] Running AIC Analysis...")
    # Using Pandas API
    importances, interactions = model.analyze(df, target_col='Churn')
    
    print("\n[Step 3] Results:")
    if not importances.empty:
        print("Top 5 Important Features:")
        print(importances.head(5)[['Feature', 'Score', 'Delta_Score']])
    
    if not interactions.empty:
        print("\nTop 5 Feature Interactions:")
        print(interactions.head(5))
        
    print("\n--- Feature Details (Impact Analysis) ---")
    if model.feature_details_ is not None:
        max_rows = 100
        details = model.feature_details_[['Feature', 'Bin_Label', 'Count', 'Target_Mean']]
        if len(details) <= max_rows:
            print(details)
        else:
            print(details.head(10))
            print(f"... ({len(details)-10} more rows hidden, max_rows={max_rows})")

        
    print("\n[Step 4] Visualizing (saving to files)...")
    # In a real environment, these would open windows. Here we could save them using matplotlib if needed.
    # We just run them to show API usage.
    try:
        plot_importance(importances)
        print(" - Importance plot generated.")
        if not interactions.empty:
            plot_interaction_heatmap(interactions)
            print(" - Interaction heatmap generated.")
    except Exception as e:
        print(f"Visualization skipped (e.g. no display): {e}")

if __name__ == "__main__":
    run_churn_analysis()
