import pytest
from bug1 import average

def test_average_empty_list():
    """Tests that average returns None for an empty list."""
    assert average([]) is None

def test_average_normal_list():
    """Tests average calculation for a standard list of numbers."""
    # Test case: [1, 2, 3] -> (1+2+3)/3 = 6/3 = 2.0
    assert average([1, 2, 3]) == pytest.approx(2.0)

def test_average_single_element():
    """Tests average calculation for a single element list."""
    # Test case: [5] -> 5/1 = 5.0
    assert average([5]) == pytest.approx(5.0)
