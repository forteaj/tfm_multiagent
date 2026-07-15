from bug1 import numbers


def test_numbers_returns_ten_functions():
    funcs = numbers()

    assert len(funcs) == 10


def test_each_function_adds_its_own_index():
    funcs = numbers()

    assert funcs[0](10) == 10
    assert funcs[1](10) == 11
    assert funcs[2](10) == 12
    assert funcs[9](10) == 19