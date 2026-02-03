import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_data(output_path="demo.csv", n_samples=50000):
    print(f"Generating {n_samples} samples...")
    
    # Generate synthetic features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.05, # Noise
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(X.shape[1])])
    
    # Add categorical feature
    df['cat_feature_A'] = np.random.choice(['Group1', 'Group2', 'Group3', 'Group4'], n_samples)
    df['cat_feature_B'] = np.random.choice(['TypeX', 'TypeY', None], n_samples, p=[0.45, 0.45, 0.1])
    
    # Add some nulls to numeric
    df.loc[df.sample(frac=0.05).index, 'feature_01'] = np.nan
    
    # Target
    df['target'] = y
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_data()
