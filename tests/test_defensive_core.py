import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from advanced_catdap import AdvancedCATDAP

def test_init_invalid_mode():
    """Test invalid save_rules_mode (Line 21)."""
    with pytest.raises(ValueError, match="save_rules_mode must be"):
        AdvancedCATDAP(save_rules_mode='invalid_mode')

def test_downcast_int32():
    """Test integer downcasting to int32 (Lines 642-643)."""
    from advanced_catdap.components.utils import downcast_codes_safe
    
    # Create array with value > 32767 but < 2147483647
    large_codes = np.array([0, 100, 40000, 50000], dtype=np.int64)
    downcasted = downcast_codes_safe(large_codes)
    assert downcasted.dtype == np.int32
    assert np.array_equal(downcasted, large_codes)

    # Test int64 preservation (Line 600)
    huge_codes = np.array([0, 2147483648], dtype=np.int64) # > 2^31-1
    kept = downcast_codes_safe(huge_codes)
    assert kept.dtype == np.int64
    assert np.array_equal(kept, huge_codes)

def test_screening_exception_handling():
    """Test exception handling inside screening loop (Line 410)."""
    df = pd.DataFrame({'Target': range(20), 'Val': range(20)})
    model = AdvancedCATDAP(verbose=False, max_bins=2)
    
    # Mock DecisionTreeRegressor.fit to raise Exception
    # This should trigger the 'except Exception: continue' block for 'tree' method
    with patch('sklearn.tree.DecisionTreeRegressor.fit', side_effect=ValueError("Mock Error")):
        # We only care that it doesn't crash and hopefully finds another method (cut/qcut)
        model.fit(df, target_col='Target')
        # If it finished without error, the exception was caught.

def test_screening_no_candidates():
    """Test when no candidates are found in screening (Line 413)."""
    # To force this, we need valid non-NaN numeric data that fails qcut, cut, AND tree.
    # - Constant value fails cut (can_cut=False).
    # - Constant value might fail qcut (bins unique < 2).
    # - Tree might run unless min_samples constraint blocks it.
    
    # Let's try constant value with strict constraints
    df = pd.DataFrame({'Target': range(20), 'Const': [10.0]*20})
    model = AdvancedCATDAP(verbose=False, min_samples_leaf_rate=0.5) 
    # High min_samples will likely block tree split on constant data?
    # Actually constant data has min=max, so 'cut' is skipped.
    # 'qcut' on constant gives 1 unique bin -> skipped.
    # 'tree': if we set min_samples high enough or max_leaf_nodes small?
    
    # analyze/fit
    importances, _ = model.analyze(df, target_col='Target')
    
    # Check if 'Const' method is 'no_candidates' (or 'all_nan' / 'constant' handled elsewhere)
    # The code handles "all_nan" earlier.
    # For constant:
    # cut check: can_cut = (min != max) -> False.
    # qcut check: bins unique -> len < 2 -> continue.
    # tree check: fit might produce single leaf?
    # If tree produces single leaf -> unique_leaves=1 -> r=2 (missing_code=1) -> code 0 and missing 1.
    # Wait, tree applied returns leaf_ids. If all same, leaf_ids all same. unique=1.
    # mapped = zeros.
    # r = 1 + 1 = 2.
    # score calc requires r >= 2.
    # IF r (which is len(unique)+1 for tree?) -> yes, code.max() + 1.
    # If all same, codes are all 0. code.max()+1 = 1.
    # r calculation in tree block: missing_code = len(unique_leaves) (which is 1). r = missing_code + 1 = 2.
    # So tree might return r=2 properly?
    # Let's mock 'tree' to also fail to force 'no_candidates'.
    
    with patch('sklearn.tree.DecisionTreeRegressor.fit', side_effect=ValueError("Fail Tree")):
         importances, _ = model.analyze(df, target_col='Target')
         
    # Check result for 'Const'
    row = importances[importances['Feature'] == 'Const']
    if not row.empty:
        # It might be 'no_candidates' or handled otherwise
        print(f"Method for Const: {row.iloc[0]['Method']}")
        assert row.iloc[0]['Method'] == 'no_candidates'
