from bug1 import average


def test_average_normal_values():
    assert average([2, 4, 6]) == 4


def test_average_single_value():
    assert average([10]) == 10


def test_average_empty_list_returns_none():
    assert average([]) is None