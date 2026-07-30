import pytest
from bug1 import remove_even_numbers

def test_remove_even_numbers_consecutive_evens():
    # Test case where multiple even numbers appear consecutively
    input_list = [1, 2, 4, 5]
    expected = [1, 5]
    result = remove_even_numbers(input_list.copy())
    assert result == expected

def test_remove_even_numbers_all_evens():
    # Test case where all numbers are even
    input_list = [2, 4, 6]
    expected = []
    result = remove_even_numbers(input_list.copy())
    assert result == expected

def test_remove_even_numbers_all_odds():
    # Test case where all numbers are odd
    input_list = [1, 3, 5]
    expected = [1, 3, 5]
    result = remove_even_numbers(input_list.copy())
    assert result == expected
