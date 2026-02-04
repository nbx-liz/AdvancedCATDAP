import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_data(output_path="data/demo.csv", n_samples=10000):
    print(f"Generating {n_samples} samples (Customer Churn Scenario - Enhanced)...")
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    # --- 1. Base Features ---
    df = pd.DataFrame()
    df['CustomerID'] = [f'CUST_{i:06d}' for i in range(n_samples)]
    
    # Demographics
    # Non-linear effect: Very young (<25) and very old (>70) might churn more
    df['Age'] = np.random.randint(18, 85, n_samples)
    
    # Contract - Strong Univariate Driver
    # Month-to-Month has much higher churn than 2-year
    df['Contract_Type'] = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_samples, p=[0.45, 0.3, 0.25])
    
    # Service Info
    df['Internet_Service'] = np.random.choice(['Fiber Optic', 'DSL', 'No'], n_samples, p=[0.4, 0.4, 0.2])
    
    # Interaction Driver: Technical Issues
    # By itself, maybe annoying. But combined with Fiber (high expectation/cost), it drives churn.
    df['Technical_Issues'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    
    # Usage Stats
    df['Tenure_Months'] = np.random.randint(0, 72, n_samples)
    df['Monthly_Charges'] = np.random.normal(70, 30, n_samples).clip(20, 150)
    
    # Noise features
    df['Payment_Method'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
    df['Gender'] = np.random.choice(['Male', 'Female'], n_samples)
    df['Random_Metric_A'] = np.random.rand(n_samples)
    df['Random_Metric_B'] = np.random.normal(0, 1, n_samples)
    
    # --- 2. Construct Target (Probabilistic) ---
    # Base probability
    logits = np.zeros(n_samples) - 2.0  # Starts low (~12%)
    
    # Univariate Effects
    # Contract: Month-to-Month adds +2.5 log-odds (huge risk)
    logits += np.where(df['Contract_Type'] == 'Month-to-Month', 2.5, 0.0)
    logits += np.where(df['Contract_Type'] == 'Two Year', -1.0, 0.0)
    
    # Age: U-shape
    # (Age - 50)^2 / Scale -> higher for extremes
    logits += ((df['Age'] - 50) ** 2) / 600.0
    
    # Interaction Effect: Fiber Optic + Technical Issues
    # If Fiber AND Tech Issues -> +2.0 (Angry customers)
    # If DSL/No AND Tech Issues -> +0.5 (Less angry)
    mask_fiber_issues = (df['Internet_Service'] == 'Fiber Optic') & (df['Technical_Issues'] == 'Yes')
    logits += np.where(mask_fiber_issues, 2.0, 0.0)
    
    # If Tech Issues but NOT Fiber (just minor annoyance)
    mask_other_issues = (df['Internet_Service'] != 'Fiber Optic') & (df['Technical_Issues'] == 'Yes')
    logits += np.where(mask_other_issues, 0.5, 0.0)

    # Convert to probability (Sigmoid)
    probs = 1 / (1 + np.exp(-logits))
    
    # Generate Target
    df['Churn'] = (np.random.random(n_samples) < probs).astype(int)

    # --- 3. Generate Additional Target: Monthly Spend ---
    # Base spend
    spend_base = 30.0

    # Add Monthly Charges (strongest driver)
    spend_base += df['Monthly_Charges'] * 0.8 # A portion of monthly charges contributes to spend

    # Add Tenure: Longer tenure might mean more services/spend
    spend_base += (df['Tenure_Months'] / 12) * 5 # +5 per year of tenure

    # Add Fiber Optic: Often associated with higher spend
    spend_base += np.where(df['Internet_Service'] == 'Fiber Optic', 20.0, 0.0)

    # Reduce spend for churned customers (they stop spending)
    spend_base -= np.where(df['Churn'] == 1, 0.5 * df['Monthly_Charges'], 0.0) # Churned customers spend less

    # Add noise
    df['Target_Spend'] = spend_base + np.random.normal(0, 10, size=n_samples)
    
    # Ensure positive spend
    df['Target_Spend'] = df['Target_Spend'].clip(lower=5)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print("\n--- Ground Truth Summary ---")
    print("1. Contract_Type: Month-to-Month is high risk.")
    print("2. Age: Non-linear U-shape (Young/Old churn more).")
    print("3. Interaction: Internet_Service='Fiber Optic' AND Technical_Issues='Yes' is critical.")
    print(f"Overall Churn Rate: {df['Churn'].mean():.2%}")
    print(f"Average Target Spend: ${df['Target_Spend'].mean():.2f}")
    print(df.head())

if __name__ == "__main__":
    generate_data()
