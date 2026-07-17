from bug1 import duplicate_matrix


def test_duplicate_matrix_returns_equal_matrix():
    matrix = [[1, 2], [3, 4]]

    copied = duplicate_matrix(matrix)

    assert copied == matrix


def test_duplicate_matrix_is_independent():
    matrix = [[1, 2], [3, 4]]

    copied = duplicate_matrix(matrix)
    copied[0][0] = 99

    assert matrix[0][0] == 1
    assert copied[0][0] == 99