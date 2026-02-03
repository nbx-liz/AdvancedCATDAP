from advanced_catdap.service.utils import sanitize_for_json
import math
import numpy as np

def test_sanitize_scalars():
    assert sanitize_for_json(1.0) == 1.0
    assert sanitize_for_json("text") == "text"
    assert sanitize_for_json(None) is None
    assert sanitize_for_json(float('nan')) is None
    assert sanitize_for_json(float('inf')) == "Infinity"
    assert sanitize_for_json(float('-inf')) == "-Infinity"

def test_sanitize_dict_list():
    data = {
        "a": [1.0, float('nan')],
        "b": {"nested": float('inf')}
    }
    clean = sanitize_for_json(data)
    assert clean["a"] == [1.0, None]
    assert clean["b"]["nested"] == "Infinity"

def test_sanitize_numpy():
    # Numpy scalars often behave like floats but might need check
    assert sanitize_for_json(np.nan) is None
    assert sanitize_for_json(np.inf) == "Infinity"
