import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def realistic_data_factory():
    def _generate(n_samples=1000, task_type='classification', seed=42):
        """
        Generates realistic Customer CRM data.
        
        Variables:
        - CustomerID: Unique ID (should be ignored/id_col).
        - Age: Numeric, Normal dist, some missing (should be handled).
        - AnnualIncome: Numeric, Log-normal.
        - TenureMonths: Numeric, Uniform.
        - AvgSupportCalls: Numeric, Poisson (driver for Churn).
        - WebSiteVisits: Numeric, rare massive outliers.
        - SubscriptionType: Categorical (Basic, Standard, Premium).
        - PaymentMethod: Categorical.
        - Country: Categorical (Medium cardinality).
        - ReferralSource: Categorical (High cardinality/Zipfian).
        - ConstantColumn: Single value (should be ignored).
        
        Target:
        - ChurnStatus (Classification): Driven by SupportCalls, Tenure, Subscription.
        - CustomerLTV (Regression): Driven by Income, Tenure, Subscription.
        """
        rng = np.random.default_rng(seed)
        data = {}
        
        # --- IDs ---
        data['CustomerID'] = [f'CUST_{i:06d}' for i in range(n_samples)]
        
        # --- Demographics (Numeric) ---
        # Age: Normal(40, 15), clipped 18-90. 10% missing.
        age = rng.normal(40, 15, n_samples)
        age = np.clip(age, 18, 90)
        mask_age_nan = rng.random(n_samples) < 0.1
        age[mask_age_nan] = np.nan
        data['Age'] = age
        
        # Income: LogNormal. 
        income = np.exp(rng.normal(10.5, 0.5, n_samples)) # ~36k median
        data['AnnualIncome'] = np.round(income, -2)
        
        # --- Behavior (Numeric) ---
        # Tenure: 0 to 120 months
        data['TenureMonths'] = rng.integers(0, 121, n_samples)
        
        # Support Calls: Poisson(lambda=1) but higher for churners usually
        data['AvgSupportCalls'] = rng.poisson(1.0, n_samples).astype(float)
        
        # Outliers: Web Visits
        visits = rng.negative_binomial(5, 0.5, n_samples).astype(float)
        # Add massive outliers to 1%
        outlier_mask = rng.random(n_samples) < 0.01
        visits[outlier_mask] *= 100
        data['WebSiteVisits'] = visits
        
        # --- Service (Categorical) ---
        # Subscription
        data['SubscriptionType'] = rng.choice(['Basic', 'Standard', 'Premium'], size=n_samples, p=[0.5, 0.3, 0.2])
        
        # Payment
        data['PaymentMethod'] = rng.choice(['CreditCard', 'PayPal', 'BankTransfer', 'Check'], size=n_samples)
        
        # Country (Medium Card)
        countries = ['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'AU', 'IN', 'BR', 'CN']
        data['Country'] = rng.choice(countries, size=n_samples)
        
        # Referral (High Card / Zipfian)
        n_refs = 100
        weights = 1.0 / np.arange(1, n_refs + 1)**1.5 # Zipf
        weights /= weights.sum()
        ref_ids = rng.choice(np.arange(n_refs), size=n_samples, p=weights)
        data['ReferralSource'] = [f'Ref_{i:03d}' for i in ref_ids]
        
        # --- Edge Cases ---
        data['ConstantColumn'] = 'SameValue'
        
        df = pd.DataFrame(data)
        
        # --- Target Generation ---
        
        if task_type == 'classification':
            # Churn (1) vs Stay (0)
            # Factors: High Support Calls (+), Low Tenure (+), Low Income (+), Monthly Contract (implicit via Basic)
            
            logit = -3.0 # Base churn rate low
            
            # Support calls effect
            logit += 0.8 * data['AvgSupportCalls']
            
            # Tenure effect (longer = less churn)
            logit -= 0.05 * data['TenureMonths']
            
            # Subscription effect (Premium = less churn)
            sub_map = {'Basic': 0.5, 'Standard': 0, 'Premium': -0.8}
            logit += np.array([sub_map[s] for s in data['SubscriptionType']])
            
            # Age effect (U-shape? Let's say younger churn more)
            age_fill = np.nan_to_num(data['Age'], 40)
            logit -= 0.02 * (age_fill - 30)
            
            churn_prob = 1 / (1 + np.exp(-logit))
            data['ChurnStatus'] = rng.binomial(1, churn_prob)
            
        else:
            # Customer LTV (Regression)
            # Factors: Income (+), Tenure (++), Premium (++)
            
            base_ltv = 500.0
            
            # Income effect (small linear)
            base_ltv += 0.01 * data['AnnualIncome']
            
            # Tenure effect (linear)
            base_ltv += 10.0 * data['TenureMonths']
            
            # Subscription
            sub_map_ltv = {'Basic': 0, 'Standard': 200, 'Premium': 1000}
            base_ltv += np.array([sub_map_ltv[s] for s in data['SubscriptionType']])
            
            # Noise
            noise = rng.normal(0, 200, n_samples)
            data['CustomerLTV'] = np.maximum(0, base_ltv + noise)
        
        return pd.DataFrame(data)
        
    return _generate
