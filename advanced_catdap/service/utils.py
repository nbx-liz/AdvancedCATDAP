import math
from typing import Any
import numpy as np

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively replace NaN/Infinity with None or strings for JSON safety.
    Also converts numpy types to native python types and stringifies unknown objects.
    """
    if obj is None:
        return None
    
    # Basic primitives
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, str):
        return obj
        
    # Recursive structures
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    # Numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return sanitize_for_json(float(obj))
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(item) for item in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
        
    # Fallback for anything else (e.g. sklearn models, datetime, etc.)
    return str(obj)
