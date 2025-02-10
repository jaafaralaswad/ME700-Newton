from newtonmethod import newton, evaluate_functions, check_convergence_function, check_derivative_nonzero
import numpy as np
from pathlib import Path
import pytest
import re



# Test function and its derivative is being calculated properly using f(x) = x^2 - 4 and f'(x) = 2x
def test_evaluate_functions():
    def f(x):
        return x**2 - 4
    def df(x):
        return 2*x
    
    x = 2
    fx, dfx = evaluate_functions(f, df, x)
    assert np.isclose(fx, 0)
    assert np.isclose(dfx, 4)



# Test for convergence criteria
@pytest.mark.parametrize("fx, epsilon_2, expected", [
    (0.001, 0.01, True),   # Test for fx is smaller than epsilon_2
    (0.1, 0.01, False),    # Test for fx is greater than epsilon_2
])

def test_check_convergence_function(fx, epsilon_2, expected):
    """Test that the convergence check works correctly."""
    assert check_convergence_function(fx, epsilon_2) == expected



# Test for derivative zero error
def test_check_derivative_nonzero_nonzero():
    """Test that the check for non-zero derivative works correctly."""
    try:
        check_derivative_nonzero(2)  # Non-zero derivative
    except ValueError:
        pytest.fail("check_derivative_nonzero raised ValueError unexpectedly!")

def test_check_derivative_nonzero_zero():
    """Test that the check for zero derivative raises an error."""
    with pytest.raises(ValueError, match="Derivative is zero. Newton's method fails!"):
        check_derivative_nonzero(0)  # Zero derivative should raise an error



# Test for maximum iterations exceeded
def test_newton_max_iterations_exceeded():
    """Test that Newton's method raises RuntimeError when max iterations are exceeded."""
    
    # Define a simple function f(x) and its derivative df(x) where the method won't converge quickly
    def f(x):
        return x**10 - 1  # This has roots at x = 2 and x = -2
    
    def df(x):
        return 10*x**9
    
    # Run the method with a very small epsilon and a low max_iter (to make sure it exceeds)
    with pytest.raises(RuntimeError, match="Maximum iterations reached without convergence."):
        newton(f, df, x0=0.5, epsilon_1=0.01, epsilon_2=0.01, max_iter=3)
