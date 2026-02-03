import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_data(output_path="demo.csv", n_samples=10000):
    print(f"Generating {n_samples} samples (Customer Churn Scenario)...")
    
    # 1. Generate base synthetic data for signal
    # We want meaningful relationships, so we'll construct them manually-ish or map them
    
    np.random.seed(42)
    
    # Base columns
    df = pd.DataFrame()
    df['Customer_ID'] = [f'CUST_{i:06d}' for i in range(n_samples)]
    
    # Demographics
    df['Age'] = np.random.randint(18, 80, n_samples)
    df['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
    df['Region'] = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    
    # Services
    df['Tenure_Months'] = np.random.randint(1, 72, n_samples)
    df['Contract_Type'] = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples, p=[0.5, 0.3, 0.2])
    df['Internet_Service'] = np.random.choice(['DSL', 'Fiber Optic', 'No'], n_samples)
    
    # Usage / Billing
    # Add some correlation: Higher tenure -> Slightly higher bill (upgrades) but lower churn
    df['Monthly_Bill'] = np.random.normal(70, 30, n_samples).clip(20, 150)
    df['Total_Usage_GB'] = np.abs(np.random.normal(200, 100, n_samples)) + (df['Monthly_Bill'] * 2)
    
    # Add noise/outliers
    df.loc[df.sample(frac=0.01).index, 'Monthly_Bill'] = 500 # Bill shock
    df.loc[df.sample(frac=0.05).index, 'Total_Usage_GB'] = np.nan # Missing data
    
    # Construct Target: Churn (1 = Yes, 0 = No)
    # Logic: High Bill + Month-to-Month + Low Tenure + High Support Calls = High Churn Probability
    
    df['Support_Calls'] = np.random.poisson(1, n_samples)
    
    prob = np.zeros(n_samples)
    prob += 0.3 # Base rate
    
    # Factors
    prob += np.where(df['Contract_Type'] == 'Month-to-Month', 0.2, -0.1)
    prob += np.where(df['Tenure_Months'] < 12, 0.1, -0.2)
    prob += np.where(df['Monthly_Bill'] > 100, 0.15, 0)
    prob += np.where(df['Support_Calls'] > 3, 0.4, 0)
    prob += np.where(df['Internet_Service'] == 'Fiber Optic', 0.05, 0) # Tech issues
    
    # Random noise
    prob += np.random.normal(0, 0.1, n_samples)
    
    # Sigmoid
    prob = 1 / (1 + np.exp(-prob * 3)) # Scale for sharpness
    
    df['Churn_Flag'] = (np.random.random(n_samples) < prob).astype(int)
    
    # Add unrelated columns (Noise)
    df['Random_Number'] = np.random.rand(n_samples)
    df['Favorite_Color'] = np.random.choice(['Blue', 'Red', 'Green'], n_samples)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_data()
