import pandas as pd
import numpy as np
import time

def generate_large_data():
    n_samples = 1_000_000
    print(f"Generating {n_samples} rows (approx 100MB+)...")
    
    start = time.time()
    
    # Efficient generation using numpy
    
    # 1. Features
    ids = np.arange(n_samples)
    
    # Demographics
    age = np.random.randint(18, 85, n_samples)
    
    # Contract
    contract_opts = ['Month-to-Month', 'One Year', 'Two Year']
    contract_probs = [0.45, 0.3, 0.25]
    contract = np.random.choice(contract_opts, n_samples, p=contract_probs)
    
    # Internet
    internet_opts = ['Fiber Optic', 'DSL', 'No']
    internet_probs = [0.4, 0.4, 0.2]
    internet = np.random.choice(internet_opts, n_samples, p=internet_probs)
    
    # Tech Issues
    tech_opts = ['Yes', 'No']
    tech_probs = [0.3, 0.7]
    tech_issues = np.random.choice(tech_opts, n_samples, p=tech_probs)
    
    # Continuous
    tenure = np.random.randint(0, 72, n_samples)
    monthly = np.random.normal(70, 30, n_samples)
    monthly = np.clip(monthly, 20, 150)
    
    # 2. Target Logic (Vectorized)
    logits = np.zeros(n_samples) - 2.0
    
    # Contract effect
    logits += np.where(contract == 'Month-to-Month', 2.5, 0.0)
    logits += np.where(contract == 'Two Year', -1.0, 0.0)
    
    # Age effect
    logits += ((age - 50) ** 2) / 600.0
    
    # Interaction: Fiber + Tech Issues
    mask_fiber_issues = (internet == 'Fiber Optic') & (tech_issues == 'Yes')
    logits += np.where(mask_fiber_issues, 2.0, 0.0)
    
    mask_other_issues = (internet != 'Fiber Optic') & (tech_issues == 'Yes')
    logits += np.where(mask_other_issues, 0.5, 0.0)
    
    # Sigmoid
    probs = 1 / (1 + np.exp(-logits))
    # Classification Target
    churn = (np.random.random(n_samples) < probs).astype(int)
    
    # --- 3. Generate Additional Target: Monthly Spend (Regression) ---
    # Base spend calculation (vectorized)
    spend_base = 30.0 + (monthly * 0.8) + ((tenure / 12) * 5)
    spend_base += np.where(internet == 'Fiber Optic', 20.0, 0.0)
    
    # Add noise
    target_spend = spend_base + np.random.normal(0, 10, size=n_samples)
    target_spend = np.clip(target_spend, 5, None)
    
    # Assemble DataFrame
    print("Assembling DataFrame...")
    df = pd.DataFrame({
        'CustomerID': ids,
        'Age': age,
        'Contract_Type': contract,
        'Internet_Service': internet,
        'Technical_Issues': tech_issues,
        'Tenure_Months': tenure,
        'Monthly_Charges': monthly,
        'Churn': churn,
        'Target_Spend': target_spend
    })
    
    output = "data/large_data.csv"
    print(f"Saving to {output}...")
    df.to_csv(output, index=False)
    
    end = time.time()
    print(f"Done in {end - start:.2f}s")
    print(f"Churn Rate: {churn.mean():.2%}")

if __name__ == "__main__":
    generate_large_data()
