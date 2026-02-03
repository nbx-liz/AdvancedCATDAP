import pandas as pd
import numpy as np
import time

def generate_large_data():
    n_samples = 1_000_000
    print(f"Generating {n_samples} rows (approx 100MB+)...")
    
    start = time.time()
    
    # Efficient generation using numpy
    data = {
        'id': np.arange(n_samples),
        'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'cat2': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'num1': np.random.normal(0, 1, n_samples),
        'num2': np.random.exponential(1, n_samples),
        'date': pd.date_range(start='2020-01-01', periods=n_samples, freq='s')
    }
    
    # Target
    # Simple logic
    target_prob = 0.1
    # Increase prob if cat2 is High
    mask = data['cat2'] == 'High'
    
    y = np.random.random(n_samples)
    y_thresh = np.where(mask, 0.3, 0.1)
    
    data['target'] = (y < y_thresh).astype(int)
    
    df = pd.DataFrame(data)
    
    output = "data/large_data.csv"
    print(f"Saving to {output}...")
    df.to_csv(output, index=False)
    
    end = time.time()
    print(f"Done in {end - start:.2f}s")

if __name__ == "__main__":
    generate_large_data()
