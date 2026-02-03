import pytest
import pandas as pd
import numpy as np
from advanced_catdap.core import AdvancedCATDAP

def test_feature_details_regression():
    # Synthetic data
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.linspace(0, 10, 100),
        'B': np.random.choice(['X', 'Y', 'Z'], 100),
        'target': np.linspace(0, 10, 100) + np.random.normal(0, 0.1, 100)
    })
    
    # Force A and B to be considered
    model = AdvancedCATDAP(task_type='regression', top_k=5, max_bins=3, delta_threshold=-999) 
    model.fit(df, target_col='target')
    
    assert model.feature_details_ is not None
    assert isinstance(model.feature_details_, pd.DataFrame)
    assert not model.feature_details_.empty
    
    print(model.feature_details_)
    
    cols = model.feature_details_.columns
    expected_cols = ['Feature', 'Bin_Idx', 'Bin_Label', 'Count', 'Target_Mean']
    for c in expected_cols:
        assert c in cols
    
    # Check 'A' details (should be numeric bins)
    a_details = model.feature_details_[model.feature_details_['Feature'] == 'A']
    assert not a_details.empty
    # Sum of counts should be 100
    assert a_details['Count'].sum() == 100
    
    # Check labels format for numeric
    label = a_details.iloc[0]['Bin_Label']
    # Should look like "(...]" or "[...]"
    assert "(" in label or "[" in label
    
    # Check 'B' details
    # Even if B is not significant, we forced delta_threshold low so it keeps features
    # But selection logic also depends on processed_codes being populated
    
    if 'B' in model.feature_details_['Feature'].values:
        b_details = model.feature_details_[model.feature_details_['Feature'] == 'B']
        assert not b_details.empty
        # Labels should be X, Y or Z
        labels = b_details['Bin_Label'].tolist()
        assert any(x in labels for x in ['X', 'Y', 'Z'])

def test_feature_details_classification():
    # Synthetic data
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 200),
        'target': np.random.choice(['Yes', 'No'], 200) # String target
    })
    
    model = AdvancedCATDAP(task_type='classification', delta_threshold=-999)
    model.fit(df, target_col='target')
    
    assert model.feature_details_ is not None
    # Target mean should be within [0, 1] (probability)
    # Since target is string, it uses internal integer coding
    means = model.feature_details_['Target_Mean']
    assert means.min() >= 0
    assert means.max() <= 1
    
    # Check that we have details for A
    assert 'A' in model.feature_details_['Feature'].values
