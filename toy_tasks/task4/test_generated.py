import pytest
from bug1 import numbers

def test_numbers_functionality():
    """Tests that the generated functions correctly add their index to the input value."""
    # Get the list of functions once
    funcs = numbers()
    input_value = 10

    # Test index 0: Expected result 10 + 0 = 10
    assert funcs[0](input_value) == input_value + 0

    # Test index 5: Expected result 10 + 5 = 15
    assert funcs[5](input_value) == input_value + 5

    # Test index 9: Expected result 10 + 9 = 19
    assert funcs[9](input_value) == input_value + 9

    # Test a different input value (e.g., 20) to ensure generalization
    funcs_20 = numbers()
    assert funcs_20[3](20) == 20 + 3