import pytest
from bug1 import numbers

def test_numbers_function():
    """Tests the numbers function to ensure correct lambda capture."""
    # The original code intends for methods[i] to be a function that adds i.
    methods = numbers()

    # Test case 0: Should add 0 (index=0)
    assert methods[0](10) == 10

    # Test case 1: Should add 1 (index=1)
    assert methods[1](10) == 11

    # Test case 2: Should add 2 (index=2)
    assert methods[2](10) == 12

    # Test case 9: Should add 9 (index=9)
    assert methods[9](10) == 19