from solution import is_degree_in_degree_range

def test_range_crossing_anti_meridian():
    assert is_degree_in_degree_range(170, -170, 50) is True

def test_anti_meridian_crossing_zero():
    assert is_degree_in_degree_range(150, -150, 0) is True

def test_simple_range_outside():
    assert is_degree_in_degree_range(10, 50, 60) is False

def test_simple_range_outside_positive():
    assert is_degree_in_degree_range(30, 70, 20) is False

def test_boundary_on_negative_side():
    assert is_degree_in_degree_range(-10, 50, -10) is True
