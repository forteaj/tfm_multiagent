import pytest
from bug1 import substract, add, multiply, divide

def test_substract():
    # Test basic subtraction
    assert substract(5, 3) == 2
    # Test with negative numbers
    assert substract(-1, -1) == 0
    # Test subtracting a larger number from a smaller one
    assert substract(3, 5) == -2

def test_add():
    # Test basic addition
    assert add(5, 3) == 8
    # Test with negative numbers
    assert add(-1, -1) == -2
    # Test adding zero
    assert add(5, 0) == 5

def test_multiply():
    # Test basic multiplication
    assert multiply(4, 3) == 12
    # Test with zero
    assert multiply(5, 0) == 0
    # Test negative numbers
    assert multiply(-2, 3) == -6
    assert multiply(-2, -3) == 6

def test_divide():
    # Test basic division (integer result)
    assert divide(10, 5) == 2
    # Test with remainder
    assert divide(10, 3) == 3
    # Test dividing by one
    assert divide(5, 1) == 5
    # Test zero division (should handle gracefully or fail as expected)
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)