from setuptools import setup, find_packages

setup(
    name="ME700-Newton",
    version="0.1",
    packages=find_packages(where="src"),  # Tell setuptools to look in src/
    package_dir={"": "src"},  # Define src as the root
    install_requires=[],  # Add dependencies if needed
)
