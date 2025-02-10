![Python Version](https://img.shields.io/badge/python-3.12-blue)
![OS](https://img.shields.io/badge/os-ubuntu%20%7C%20macos%20%7C%20windows-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[![codecov](https://codecov.io/gh/jaafaralaswad/ME700-Newton/branch/main/graph/badge.svg)](https://codecov.io/gh/jaafaralaswad/ME700-Newton) ![GitHub Actions](https://github.com/jaafaralaswad/ME700-Newton/actions/workflows/tests.yml/badge.svg)



# ME700 Assignment 1: Part 1

This repository presents a newton-raphson method solver developed for the first assignment in the ME700 course. The first three numerical examples demonstrate solving algebraic equations. The fourth example applies the method to find the reaction bending moment in a cantilever beam, while the fifth example computes the components of a velocity vector.

# Newton's Method

Newton's method is an efficient numerical technique for finding real roots of equations. It iteratively refines an initial guess, $x_0$, using the formula:  

$$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $$

The method stops when either $|x_{n+1} - x_n| < \epsilon_1$ or  $|f(x_n)| < \epsilon_2$, ensuring sufficient accuracy.  

Unlike the bisection method, Newton-Raphson converges quadratically when $x_0$ is close to the root. However, it requires $f'(x)$, may fail if $f'(x) = 0$, and can diverge from poor initial guesses.

For systems of equations  $\mathbf{F}(\mathbf{x}) = 0$, the method extends to multiple dimensions using the Jacobian matrix $\mathbf{J}$:  

$$ \mathbf{x}_{n+1} = \mathbf{x}_n - \mathbf{J}^{-1} \mathbf{F}(\mathbf{x}_n)$$

Here, $\mathbf{J}$ is the matrix of partial derivatives $\frac{\partial F_i}{\partial x_j}$.


# Conda environment, install, and testing

This procedure is very similar to what we did in class. First, you need to download the repository and unzip it. Then, to install the package, use:

```bash
conda create --name newton-method-env python=3.12
```

After creating the environment (it might have already been created by you earlier), make sure to activate it, use:

```bash
conda activate newton-method-env
```

Check that you have Python 3.12 in the environment. To do so, use:

```bash
python --version
```

Create an editable install of the newton's method code. Use the following line making sure you are in the correct directory:

```bash
pip install -e .
```

You must do this in the correct directory; in order to make sure, replace the dot at the end by the directory of the folder "ME700-A1-main" that you unzipped earlier: For example, on my computer, the line would appear as follows:

```bash
pip install -e /Users/jaafaralaswad/Downloads/ME700-Newton-main
```

Now, you can test the code, make sure you are in the tests directory. You can know in which directory you are using:

```bash
pwd
```

Navigate to the tests folder using the command:

```bash
cd
```

On my computer, to be in the tests folder, I would use:

```bash
/Users/jaafaralaswad/Downloads/ME700-Newton-main/tests
```


Once you are in the tests directory, use the following to run the tests:

```bash
pytest -s test_newton_method.py
```

Code coverage should be 100%.

To run the tutorial, make sure you are in the tutorials directory. You can navigate their as you navigated to the tests folder. On my computer, I would use:

```bash
cd /Users/jaafaralaswad/Downloads/ME700-Newton-main/tutorials
```

Once you are there, you can use:

```bash
pip install jupyter
```

```bash
jupyter notebook tutorial.ipynb
```

A Jupyter notebook will pop up, containing five numerical examples.
