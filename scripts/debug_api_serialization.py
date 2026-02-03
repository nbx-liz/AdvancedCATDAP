import pandas as pd
import json
import numpy as np
import datetime
from advanced_catdap.service.dataset_manager import DatasetManager
import os

def debug_serialization():
    manager = DatasetManager(storage_dir="data")
    
    # Use existing dataset if possible
    files = list(manager.storage_dir.glob("*.parquet"))
    if not files:
        print("No datasets found. Creating dummy.")
        df = pd.DataFrame({
            "A": [1, 2, np.nan], 
            "B": ["x", "y", None],
            "C": [1.1, float('inf'), float('-inf')] # Potential JSON killers
        })
        df.to_csv("temp_debug.csv", index=False)
        meta = manager.register_dataset("temp_debug.csv")
        dataset_id = meta.dataset_id
        os.remove("temp_debug.csv")
    else:
        dataset_id = files[0].stem
        print(f"Using dataset: {dataset_id}")

    print("Fetching preview...")
    df = manager.get_preview(dataset_id, n_rows=100)
    print("Preview DataFrame dtypes:")
    print(df.dtypes)
    
    print("\nApplying NaN fix (Robust)...")
    # Cast to object first to allow None
    df = df.astype(object).where(pd.notnull(df), None)
    
    print("\nConverting to dict...")
    data = df.to_dict(orient="records")
    
    print("Attempting JSON serialization...")
    try:
        # FastAPI uses standard json with allow_nan=True by default? 
        # Actually Pydantic/Starlette might handle it.
        # But let's check standard strict json first.
        json_str = json.dumps(data, allow_nan=False) # standard JSON spec doesn't allow NaN/Inf
        print("Serialization SUCCESS (strict)!")
    except Exception as e:
        print(f"Serialization FAILED (strict): {e}")
        try:
            json.dumps(data, allow_nan=True)
            print("Serialization SUCCESS (allow_nan=True) - Standard Python JSON")
        except Exception as e2:
            print(f"Serialization FAILED (allow_nan=True): {e2}")

    # Inspect one row
    print("\nSample Row Data:")
    print(data[0])

if __name__ == "__main__":
    debug_serialization()
