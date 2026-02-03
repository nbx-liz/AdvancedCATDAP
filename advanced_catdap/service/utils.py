import math
from typing import Any

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively replace NaN/Infinity with None or strings for JSON safety.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj
