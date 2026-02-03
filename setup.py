from setuptools import setup, find_packages

setup(
    name="advanced_catdap",
    version="0.1.0",
    description="Advanced Categorical Data Analysis Program (AIC-based variable selection)",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pydantic>=2.0"
    ],
    python_requires=">=3.7",
)
