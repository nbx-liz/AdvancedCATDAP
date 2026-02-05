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


def test_sanitize_numpy_integer():
    """Test numpy integer types (lines 34-35)."""
    assert sanitize_for_json(np.int32(42)) == 42
    assert sanitize_for_json(np.int64(123)) == 123
    assert isinstance(sanitize_for_json(np.int64(1)), int)


def test_sanitize_numpy_float():
    """Test numpy float types (lines 36-37)."""
    assert sanitize_for_json(np.float32(3.14)) == pytest.approx(3.14, rel=1e-5)
    assert sanitize_for_json(np.float64(2.71)) == pytest.approx(2.71, rel=1e-5)
    assert sanitize_for_json(np.float64(np.nan)) is None
    assert sanitize_for_json(np.float64(np.inf)) == "Infinity"


def test_sanitize_numpy_bool():
    """Test numpy bool type (lines 40-41)."""
    assert sanitize_for_json(np.bool_(True)) is True
    assert sanitize_for_json(np.bool_(False)) is False
    assert isinstance(sanitize_for_json(np.bool_(True)), bool)


def test_sanitize_numpy_array():
    """Test numpy ndarray (lines 38-39)."""
    arr = np.array([1, 2, np.nan, np.inf])
    result = sanitize_for_json(arr)
    assert result == [1.0, 2.0, None, "Infinity"]


def test_sanitize_fallback():
    """Test fallback for unknown objects (line 44)."""
    class CustomObject:
        def __str__(self):
            return "CustomObject"
    
    result = sanitize_for_json(CustomObject())
    assert result == "CustomObject"


def test_sanitize_tuple():
    """Test tuple handling (line 30-31)."""
    result = sanitize_for_json((1, 2, 3))
    assert result == [1, 2, 3]


def test_sanitize_bool():
    """Test bool handling (lines 14-15)."""
    assert sanitize_for_json(True) is True
    assert sanitize_for_json(False) is False


def test_sanitize_int():
    """Test int handling (lines 16-17)."""
    assert sanitize_for_json(42) == 42
    assert isinstance(sanitize_for_json(42), int)


# Import pytest for approx
import pytest
