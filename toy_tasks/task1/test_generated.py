import pytest
from bug1 import substract, add, multiply, divide

def test_substract():
    # Expected: a - b
    assert substract(5, 3) == 2

def test_add():
    # Expected: a + b
    assert add(5, 3) == 8

def test_multiply():
    # Expected: a * b
    assert multiply(4, 5) == 20

def test_divide():
    # Expected: a // b
    assert divide(10, 3) == 3