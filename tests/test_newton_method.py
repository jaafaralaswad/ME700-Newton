import numpy as np
import pytest
from newtonmethod import newton, newton_raphson

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
    fx, dfx = newton.evaluate_functions(f1, df1, x)
    assert np.isclose(fx, 0)
    assert np.isclose(dfx, 4)

# Test check_convergence_function
@pytest.mark.parametrize("fx, epsilon_2, expected", [
    (0.001, 0.01, True),  
    (0.1, 0.01, False),
])
def test_check_convergence_function(fx, epsilon_2, expected):
    assert newton.check_convergence_function(fx, epsilon_2) == expected

# Test check_derivative_nonzero
def test_check_derivative_nonzero():
    assert newton.check_derivative_nonzero(2.0) is None
    with pytest.raises(ValueError):
        newton.check_derivative_nonzero(0.0)

# Test update_x for Newton method
def test_update_x():
    updated_x = newton.update_x(2.0, 4.0, 2.0, 1e-6)
    assert np.isclose(updated_x, 0.0).all()

# Test Newton method for single variable
def test_newton_success():
    x0 = 2.0
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 50
    x = newton.newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)
    assert np.isclose(x, [2.0, -2.0]).any()

def test_newton_nonconvergence():
    x0 = 0.0  # Derivative is zero, should raise error
    epsilon_1 = 1e-6
    epsilon_2 = 1e-6
    max_iter = 10
    with pytest.raises(ValueError):
        newton.newton(f1, df1, x0, epsilon_1, epsilon_2, max_iter)

# Test evaluate_functions for Newton-Raphson
def test_evaluate_functions_newton_raphson():
    x, y = 1, 1
    F, J = newton_raphson.evaluate_functions(f2, df2x, df2y, g2, dg2x, dg2y, x, y)  # ✅ Corrected function call
    assert np.isclose(F[0], -2)
    assert np.isclose(F[1], 0)
    assert np.isclose(J[0, 0], 2)
    assert np.isclose(J[1, 1], -1)

# Test check_convergence for Newton-Raphson
@pytest.mark.parametrize("F, epsilon, expected", [
    (np.array([0.0001, 0.0001]), 1e-3, True),
    (np.array([0.1, 0.1]), 1e-3, False),
])
def test_check_convergence(F, epsilon, expected):
    assert newton_raphson.check_convergence(F, epsilon) == expected  # ✅ Correct function call

# Test check_jacobian_singular
def test_check_jacobian_singular():
    J = np.array([[2.0, 2.0], [-2.0, -2.0]])
    with pytest.raises(ValueError):
        newton_raphson.check_jacobian_singular(J)  # ✅ Allow expected ValueError

def test_update_variables():
    J = np.array([[2.0, 1.0], [1.0, 2.0]])  # ✅ Square (2x2) matrix
    F = np.array([-2.0, -2.0]).reshape(2, 1)  # ✅ Convert to column vector
    x, y = newton_raphson.update_variables(1.0, 1.0, J, F)  # Compute update

    print("\nDEBUG: Expected x = 1.66667, Actual x =", x)
    print("DEBUG: Expected y = 1.66667, Actual y =", y)

    assert np.isclose(x, 1.66667)  # ✅ Fixed expected value
    assert np.isclose(y, 1.66667)  # ✅ Fixed expected value

# Test Newton-Raphson method for system of equations
def test_newton_raphson_success():
    x0, y0 = 1.0, 1.0
    epsilon = 1e-6
    max_iter = 50
    x, y = newton_raphson.newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)
    assert np.isclose(x**2 + y**2, 4)
    assert np.isclose(x, y)

def test_newton_raphson_nonconvergence():
    x0, y0 = 0.0, 0.0  # Singular point
    epsilon = 1e-6
    max_iter = 10
    with pytest.raises(ValueError):
        newton_raphson.newton_raphson(f2, df2x, df2y, g2, dg2x, dg2y, x0, y0, epsilon, max_iter)
