import pytest
from bug1 import duplicate_matrix

def test_duplicate_matrix_independence():
    original_matrix = [
        [1, 2],
        [3, 4]
    ]
    # Create the copy using the function under test
    copied_matrix = duplicate_matrix(original_matrix)

    # 1. Modify an element in the copied matrix
    copied_matrix[0][0] = 99

    # 2. Verify that the original matrix remains unchanged
    assert original_matrix[0][0] == 1
    
    # 3. Modify another element in the copied matrix to ensure deep copy
    copied_matrix[1].append(5)
    
    # 4. Verify that the original matrix structure is unchanged
    assert len(original_matrix[1]) == 2
    assert original_matrix[1][0] == 3