import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from advanced_catdap import AdvancedCATDAP
from advanced_catdap.examples.utils import generate_ltv_data

def run_sklearn_demo():
    print("\n=== Sklearn API Demo (Pipeline) ===")
    
    # 1. Data
    df = generate_ltv_data(n_samples=2000)
    X = df.drop(columns=['LTV'])
    y = df['LTV']
    
    # Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 2. Create Pipeline
    # AdvancedCATDAP acts as a discretizer/preprocessor
    pipe = Pipeline([
        ('catdap', AdvancedCATDAP(task_type='regression', verbose=True, max_bins=5)),
        ('linear_reg', LinearRegression())
    ])
    
    print("\nTraining Pipeline...")
    pipe.fit(X_train, y_train)
    
    # Check internal importances from the step
    catdap_step = pipe.named_steps['catdap']
    print("\n[CATDAP] Feature Importances found during fit:")
    print(catdap_step.feature_importances_[['Feature', 'Score', 'Delta_Score', 'Method']].head(3))

    print("\n[CATDAP] Top Interactions:")
    print(catdap_step.interaction_importances_.head(3))

    
    print("\n[CATDAP] Feature Details (Impact Analysis):")
    if catdap_step.feature_details_ is not None:
        max_rows = 100
        details = catdap_step.feature_details_[['Feature', 'Bin_Label', 'Count', 'Target_Mean']]
        if len(details) <= max_rows:
            print(details)
        else:
            print(details.head(10))
            print(f"... ({len(details)-10} more rows hidden, max_rows={max_rows})")

    
    # 3. Transparent Transformation (Optional Inspection)
    print("\n[CATDAP] Transformed Data Preview (First 5 rows of Test Set):")
    # We can transform just a part of data to see what the model sees
    X_test_transformed = catdap_step.transform(X_test.head())
    print(X_test_transformed.iloc[:, :5])
    
    # 4. Predict
    print("\nPredicting...")
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.2f}")


if __name__ == "__main__":
    run_sklearn_demo()
