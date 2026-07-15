import pytest
from bug1 import substract, add, multiply, divide

def test_substract():
    # Test case for subtraction (a - b)
    assert substract(5, 3) == 2
    assert substract(0, 10) == -10
    assert substract(-5, 5) == -10

def test_add():
    # Test case for addition (a + b)
    assert add(5, 3) == 8
    assert add(-5, 5) == 0
    assert add(-1, -1) == -2

def test_multiply():
    # Test case for multiplication (a * b)
    assert multiply(5, 3) == 15
    assert multiply(-5, 2) == -10
    assert multiply(-5, -2) == 10

def test_divide():
    # Test case for division (a // b)
    # Note: The buggy implementation uses integer division (//), so tests should reflect that.
    assert divide(7, 3) == 2
    assert divide(6, 3) == 2
    assert divide(-7, 3) == -3