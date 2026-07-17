import pytest
from bug1 import add_item

def test_add_item_no_list_provided():
    # Test case 1: Calling without an explicit list argument.
    # The function should return a new list and not modify the state for subsequent calls.
    result1 = add_item("apple")
    assert result1 == ["apple"]
    assert isinstance(result1, list)

    # Test case 2: Calling again without an explicit list argument.
    # If the default mutable argument is used (the bug), result2 will contain "apple".
    result2 = add_item("banana")
    # Expected behavior: A fresh call should start with a new, empty list.
    assert result2 == ["banana"]

def test_add_item_with_empty_list():
    # Test case 3: Providing an explicit empty list argument.
    items = []
    result = add_item("cherry", items)
    assert result == ["cherry"]