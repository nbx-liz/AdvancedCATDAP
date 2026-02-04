
import numpy as np
import pandas as pd
from typing import Tuple

def downcast_codes_safe(codes: np.ndarray) -> np.ndarray:
    """
    Downcasts integer array to smallest safe numpy integer type.
    """
    if codes.size == 0: return codes.astype(np.int8)
    max_v = codes.max()
    if max_v <= 127: return codes.astype(np.int8)
    elif max_v <= 32767: return codes.astype(np.int16)
    elif max_v <= 2147483647: return codes.astype(np.int32)
    return codes

def check_cardinality_and_id(series: pd.Series, sample_indices: np.ndarray, 
                             max_estimated_uniques: int) -> Tuple[bool, bool]:
    """
    Checks if a series has high cardinality or looks like an ID column.
    """
    sample_vals = series.iloc[sample_indices] if hasattr(series, 'iloc') else series[sample_indices]
    sample_nunique = sample_vals.nunique()
    sample_len = len(sample_vals)
    n_total = len(series)
    
    estimated_total = n_total if (sample_nunique == sample_len) else sample_nunique * (n_total / sample_len)
    unique_ratio = sample_nunique / sample_len if sample_len > 0 else 0
    
    is_high_card = estimated_total > max_estimated_uniques
    is_id_like = (unique_ratio > 0.99) and (estimated_total > 0.99 * n_total)
    return is_high_card, is_id_like
