import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from advanced_catdap import AdvancedCATDAP

def run_error_handling_demo():
    print("\n=== Error Handling Demo ===")

    # 1. Invalid Initialization
    try:
        print("\n[Case 1] Invalid save_rules_mode...")
        model = AdvancedCATDAP(save_rules_mode='invalid')
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # 2. Empty Data
    try:
        print("\n[Case 2] Empty DataFrame...")
        model = AdvancedCATDAP(verbose=False)
        model.fit(pd.DataFrame({'A': []}), target_col='A')
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # 3. Missing Target
    try:
        print("\n[Case 3] Missing Target Column...")
        df = pd.DataFrame({'Data': [1, 2, 3]})
        model = AdvancedCATDAP(verbose=False)
        model.fit(df, target_col='Target')
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # 4. Warnings (Verbose)
    print("\n[Case 4] Transforming without rules (Warning)...")
    model = AdvancedCATDAP(verbose=True)
    # Redirect stdout to capture print if needed, but here we just let it print
    model.transform(pd.DataFrame({'A': [1]}))

if __name__ == "__main__":
    run_error_handling_demo()
