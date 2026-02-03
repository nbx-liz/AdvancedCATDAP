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
        "pydantic>=2.0",
        "pyarrow>=14.0.0",
        "duckdb>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "celery>=5.0.0",
        "redis>=4.0.0",
        "python-multipart",
        "eventlet"
    ],
    python_requires=">=3.7",
)
