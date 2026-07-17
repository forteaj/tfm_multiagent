from bug1 import remove_even_numbers


def test_remove_even_numbers_with_consecutive_even_values():
    result = remove_even_numbers([1, 2, 4, 6, 7])

    assert result == [1, 7]


def test_remove_even_numbers_keeps_odd_values():
    result = remove_even_numbers([1, 3, 5])

    assert result == [1, 3, 5]


def test_remove_even_numbers_empty_list():
    result = remove_even_numbers([])

    assert result == []