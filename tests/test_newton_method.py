
import numpy as np
import pytest
from newton import newton, evaluate_functions, check_convergence_function, check_derivative_nonzero, update_x
from newton_raphson import newton_raphson, evaluate_functions as eval_raphson, check_convergence, check_jacobian_singular, update_variables

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

# Test evaluate_functions for Newton
def test_evaluate_functions_newton():
    x = 2
    fx, dfx = evaluate_functions(f1, df1, x)
    assert np.isclose(fx, 0)
    assert np.isclose(dfx, 4)

# Test check_convergence_function
@pytest.mark.parametrize("fx, epsilon_2, expected", [
    (0.001, 0.01, True),  
    (0.1, 0.01, False),
])
def test_check_convergence_function(fx, epsilon_2, expected):
    assert check_convergence_function(fx, epsilon_2) == expected

# Test check_derivative_nonzero
def test_check_derivative_nonzero():
    assert check_derivative_nonzero(2) == True
    with pytest.raises(ZeroDivisionError):
        check_derivative_nonzero(0)

# Test update_x
def test_update_x():
    x_old = 2.0
    fx = -1.0
    dfx = 2.0
    x_new = update_x(x_old, fx, dfx)
    assert np.isclose(x_new, 2.5)

# Test Newton method
def test_newton_success():
    x0 = 2.0
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 50
    x = newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)
    assert np.isclose(x, 2.0) or np.isclose(x, -2.0)

def test_newton_nonconvergence():
    x0 = 0.0
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 10
    with pytest.raises(ZeroDivisionError):
        newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)

# Test evaluate_functions for Newton-Raphson
def test_evaluate_functions_newton_raphson():
    x, y = 1.0, 1.0
    fx, dfx = eval_raphson(f2, df2x, df2y, x, y)
    assert np.isclose(fx, -2.0)
    assert np.isclose(dfx, 2.0)

# Test check_convergence for Newton-Raphson
def test_check_convergence():
    assert check_convergence(0.001, 0.01) == True
    assert check_convergence(0.1, 0.01) == False

# Test check_jacobian_singular
def test_check_jacobian_singular():
    assert check_jacobian_singular(1, 1, 1, 1) == False
    with pytest.raises(ZeroDivisionError):
        check_jacobian_singular(0, 0, 0, 0)

# Test update_variables
def test_update_variables():
    x_old, y_old = 1.0, 1.0
    fx, fy = -1.0, -1.0
    J_inv = np.array([[0.5, 0], [0, 0.5]])
    x_new, y_new = update_variables(x_old, y_old, fx, fy, J_inv)
    assert np.isclose(x_new, 1.5)
    assert np.isclose(y_new, 1.5)

# Test Newton-Raphson method
def test_newton_raphson_success():
    x0, y0 = 1.0, 1.0
    epsilon = 1e-6
    max_iter = 50
    x, y = newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)
    assert np.isclose(x**2 + y**2, 4)
    assert np.isclose(x, y)

def test_newton_raphson_nonconvergence():
    x0, y0 = 0.0, 0.0
    epsilon = 1e-6
    max_iter = 10
    with pytest.raises(ZeroDivisionError):
        newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)
