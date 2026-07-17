import pytest
from bug1 import reverse_items

def test_reverse_items_returns_reversed_copy(items):
    original = list(items)
    result = reverse_items(original)
    # Check if the result is correctly reversed
    assert result == original[::-1]
    # Crucially, check that the original list was not modified
    assert original == list(items)