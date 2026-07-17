import pytest
from bug1 import numbers_up_to

def test_numbers_up_to_includes_n():
    # Test case where n is a small positive number (e.g., 3). Expected: [0, 1, 2, 3]
    assert numbers_up_to(3) == [0, 1, 2, 3]

def test_numbers_up_to_includes_zero():
    # Test case where n is 0. Expected: [0]
    assert numbers_up_to(0) == [0]

def test_numbers_up_to_with_larger_number():
    # Test case with a larger number (e.g., 10). Expected: [0, ..., 10]
    expected = list(range(11))
    assert numbers_up_to(10) == expected