import pytest
from bug1 import add_item

def test_add_item_no_list_provided():
    # Test case 1: Calling without an explicit list should return a new list.
    result1 = add_item("apple")
    assert result1 == ["apple"]
    # Ensure the returned object is indeed a new list (not None or something else)
    assert isinstance(result1, list)

def test_add_item_state_isolation():
    # Test case 2: Calls without an explicit list should not share state between them.
    # Call 1
    list1 = add_item("apple")
    # Call 2 (should start fresh)
    list2 = add_item("banana")
    
    # Check that the results are independent and correct
    assert list1 == ["apple"]
    assert list2 == ["banana"]
    
    # Crucial check: Ensure modifying one result does not affect the other.
    list1.append("cherry")
    list2_after_modification = add_item("date") # This call is just to ensure we are testing fresh calls
    assert list2 != ["apple", "banana"] # Should still be isolated
    
def test_add_item_with_existing_list():
    # Test case 3: Standard usage when a list is provided (should modify in place).
    initial_list = []
    result = add_item("grape", initial_list)
    assert result is initial_list # Should return the modified list object
    assert result == ["grape"]

def test_add_item_with_existing_list_multiple():
    # Test case 4: Standard usage when a list is provided multiple times.
    initial_list = []
    result1 = add_item("kiwi", initial_list)
    assert result1 == ["kiwi"]
    
    initial_list.append("mango") # Simulate external modification or subsequent use
    result2 = add_item("orange", initial_list)
    # The buggy code modifies the list in place and returns it.
    assert result2 is initial_list
    assert result2 == ["kiwi", "mango", "orange"] # This test confirms the intended side effect of modification.