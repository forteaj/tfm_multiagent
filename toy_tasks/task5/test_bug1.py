from bug1 import numbers_up_to


def test_numbers_up_to_includes_limit():
    assert numbers_up_to(5) == [0, 1, 2, 3, 4, 5]


def test_numbers_up_to_zero():
    assert numbers_up_to(0) == [0]


def test_numbers_up_to_one():
    assert numbers_up_to(1) == [0, 1]