import pytest
from bug1 import average

def test_average_of_non_empty_list():
    """Tests the average of a standard list."""
    assert average([1, 2, 3]) == 2.0

def test_average_of_single_element_list():
    """Tests the average of a single element."""
    assert average([5]) == 5.0

def test_average_of_empty_list():
    """Tests that an empty list returns None, preventing ZeroDivisionError."""
    # The current implementation raises ZeroDivisionError for []
    assert average([]) is None