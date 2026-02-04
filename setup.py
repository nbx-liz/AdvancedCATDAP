from setuptools import setup, find_packages

setup(
    name="advanced_catdap",
    version="0.1.0",
    description="Advanced Categorical Data Analysis Program (AIC-based variable selection)",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.3.3",
        "numpy>=2.2.6",
        "scikit-learn>=1.7.2",
        "matplotlib>=3.10.8",
        "seaborn>=0.13.2",
        "pydantic>=2.0",
        "pyarrow>=14.0.0",
        "duckdb>=1.0.0",
    ],
    extras_require={
        "web": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "python-multipart>=0.0.22",
            "httpx>=0.28.1",
        ],
        "gui": [
            "streamlit>=1.25.0",
            "plotly>=6.5.2",
        ],
        "all": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "python-multipart>=0.0.22",
            "httpx>=0.28.1",
            "streamlit>=1.25.0",
            "plotly>=6.5.2",
        ]
    },
    python_requires=">=3.7",
)
