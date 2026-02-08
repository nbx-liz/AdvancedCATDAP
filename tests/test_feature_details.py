import pytest
import pandas as pd
import numpy as np
from advanced_catdap.core import AdvancedCATDAP
from advanced_catdap.components.discretizer import Discretizer
from advanced_catdap.components.scoring import Scorer

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
        assert any(any(tag in str(lbl) for tag in ['X', 'Y', 'Z']) for lbl in labels)

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


def test_feature_details_integer_levels_keep_numeric_order():
    np.random.seed(0)
    values = [1, 2, 3, 10, 11] * 20
    df = pd.DataFrame(
        {
            "int_feature": values,
            "target": [0, 1, 0, 1, 0] * 20,
        }
    )
    model = AdvancedCATDAP(task_type="classification", top_k=5, delta_threshold=-999)
    model.fit(df, target_col="target")

    details = model.feature_details_[model.feature_details_["Feature"] == "int_feature"]
    assert not details.empty
    labels = details["Bin_Label"].tolist()
    # Default style prefixes numeric bins: 01_1, 02_2, ...
    assert labels == sorted(labels)


@pytest.mark.parametrize("dtype", [float, int])
def test_axis_metadata_numeric_bins_use_lower_bound_order_for_sort_key(dtype):
    discretizer = Discretizer(task_type="classification", scorer=Scorer())
    raw = pd.Series([74, 84, 29, 64, 65, 73, 18, 22, 23, 28], dtype=dtype)
    # Intentionally non-monotonic code assignment by value range (tree-like behavior)
    codes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
    rule = {"type": "tree", "missing_code": 5}

    code_order, labels, sort_keys = discretizer.get_axis_metadata(raw, codes, rule)

    assert code_order == [0, 1, 2, 3, 4]
    # sort keys follow lower bound order, not bin code order
    assert sort_keys == ["05", "03", "04", "01", "02"]

    ordered_labels = [label for _, label in sorted(zip(sort_keys, labels), key=lambda x: x[0])]
    assert ordered_labels[0].startswith("01_[18.000, 22.000]")
    assert ordered_labels[1].startswith("02_[23.000, 28.000]")
    assert ordered_labels[2].startswith("03_[29.000, 64.000]")
    assert ordered_labels[3].startswith("04_[65.000, 73.000]")
    assert ordered_labels[4].startswith("05_[74.000, 84.000]")
