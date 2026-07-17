import pytest
from bug1 import duplicate_matrix

def test_duplicate_matrix_is_independent():
    original_matrix = [
        [1, 2],
        [3, 4]
    ]
    # Create the copy using the function under test
    copied_matrix = duplicate_matrix(original_matrix)

    # 1. Verify that modifying the copied matrix does not affect the original.
    # Modify an element in the nested list of the copy
    copied_matrix[0][0] = 99

    # Assert that the original matrix remains unchanged
    assert original_matrix[0][0] == 1

    # 2. Verify that modifying the original matrix does not affect the copy.
    original_matrix[1][1] = 88
    
    # Re-run the modification on the copy to ensure it still holds its value (99)
    copied_matrix[0][0] = 99 # This line is just for clarity, but we check the original state of the copy after modifying the source.
    assert copied_matrix[1][1] == 4
