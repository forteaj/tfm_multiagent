import pytest
from bug1 import remove_even_numbers

def test_consecutive_even_numbers():
    # Test case where consecutive even numbers are present.
    # The buggy implementation modifies the list while iterating over it,
    # causing elements to be skipped when multiple removals occur.
    input_list = [1, 2, 4, 5]
    expected_output = [1, 5]
    
    # We must pass a copy because remove_even_numbers modifies the list in place.
    result = remove_even_numbers(list(input_list))
    assert result == expected_output

def test_all_even_numbers():
    # Test case where all numbers are even.
    input_list = [2, 4, 6]
    expected_output = []
    result = remove_even_numbers(list(input_list))
    assert result == expected_output

def test_mixed_numbers():
    # Test case with a mix of odd and even numbers.
    input_list = [1, 2, 3, 4, 5]
    expected_output = [1, 3, 5]
    result = remove_even_numbers(list(input_list))
    assert result == expected_output
