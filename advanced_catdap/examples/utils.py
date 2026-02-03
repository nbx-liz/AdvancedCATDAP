import pandas as pd
import numpy as np

def generate_crm_data(n_samples=2000, seed=42):
    """
    Generates a realistic Customer CRM dataset for classification (Churn).
    """
    rng = np.random.default_rng(seed)
    
    # Demographics
    age = rng.normal(40, 15, size=n_samples).astype(int)
    age = np.clip(age, 18, 90)
    
    gender = rng.choice(['M', 'F', 'NB', np.nan], size=n_samples, p=[0.48, 0.48, 0.02, 0.02])
    
    # Subscription Details
    plan_type = rng.choice(['Basic', 'Standard', 'Premium'], size=n_samples, p=[0.5, 0.3, 0.2])
    tenure_months = rng.choice(range(1, 61), size=n_samples)
    
    # Usage Stats (with noise and missing)
    monthly_usage = rng.exponential(scale=100, size=n_samples)
    # Correlation: Premium users use more
    monthly_usage[plan_type == 'Premium'] *= 1.5
    
    # Support Calls category
    support_calls = rng.choice(['None', 'Low', 'High'], size=n_samples, p=[0.6, 0.3, 0.1])
    
    # Target: Churn (Binary)
    # Log-odds calculation
    logit = -2.0 # Base churn rate
    
    # Effects
    logit += np.where(plan_type == 'Basic', 0.5, 0.0) # Basic users churn more
    logit -= tenure_months * 0.05 # Longer tenure = less churn
    logit += np.where(support_calls == 'High', 1.0, 0.0) # High support calls = unhappy
    
    # Interaction: Young Basic users churn most
    logit += np.where((age < 30) & (plan_type == 'Basic'), 1.0, 0.0)
    
    # Add noise
    logit += rng.normal(0, 0.5, size=n_samples)
    
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'PlanType': plan_type,
        'Tenure': tenure_months,
        'Usage': monthly_usage,
        'SupportCalls': support_calls,
        'Churn': churn
    })
    
    # Add some noise/edge case columns
    df['EmptyCol'] = np.nan
    df['ConstantCol'] = 'Fixed'
    
    return df

def generate_ltv_data(n_samples=2000, seed=42):
    """
    Generates a realistic LTV dataset for regression.
    """
    rng = np.random.default_rng(seed)
    
    # Features
    purchases_past_year = rng.poisson(5, size=n_samples)
    avg_order_value = rng.gamma(shape=2, scale=50, size=n_samples) # Skewed
    
    campaign_source = rng.choice(['Email', 'Social', 'Organic', 'Affiliate'], size=n_samples)
    
    days_since_last_order = rng.exponential(30, size=n_samples).astype(int)
    
    # Target: Predicted LTV (Numeric)
    # Base LTV
    ltv = purchases_past_year * avg_order_value
    
    # Campaign effect
    ltv[campaign_source == 'Email'] *= 1.2
    
    # Decay
    ltv *= np.exp(-days_since_last_order / 365)
    
    # Noise
    ltv += rng.normal(0, 50, size=n_samples)
    ltv = np.maximum(ltv, 0)
    
    df = pd.DataFrame({
        'Purchases': purchases_past_year,
        'AOV': avg_order_value,
        'Source': campaign_source,
        'Recency': days_since_last_order,
        'LTV': ltv
    })
    
    return df
