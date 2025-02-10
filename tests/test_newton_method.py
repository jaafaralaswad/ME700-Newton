
import numpy as np
import pytest
from newton import newton
from newton_raphson import newton_raphson

# Define test functions
def f1(x):
    return x**2 - 4

def df1(x):
    return 2*x

def f2(x, y):
    return x**2 + y**2 - 4

def df2x(x, y):
    return 2*x

def df2y(x, y):
    return 2*y

def g2(x, y):
    return x - y

def dg2x(x, y):
    return 1

def dg2y(x, y):
    return -1

# Test Newton method for single variable
def test_newton_success():
    x0 = 2.0
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 50
    x = newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)
    assert np.isclose(x, 2.0) or np.isclose(x, -2.0)

def test_newton_nonconvergence():
    x0 = 0.0  # Derivative is zero, should raise error
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 10
    with pytest.raises(ZeroDivisionError):
        newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)

# Test Newton-Raphson method for system of equations
def test_newton_raphson_success():
    x0, y0 = 1.0, 1.0
    epsilon = 1e-6
    max_iter = 50
    x, y = newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)
    assert np.isclose(x**2 + y**2, 4)
    assert np.isclose(x, y)

def test_newton_raphson_nonconvergence():
    x0, y0 = 0.0, 0.0  # Singular point
    epsilon = 1e-6
    max_iter = 10
    with pytest.raises(ZeroDivisionError):
        newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)
