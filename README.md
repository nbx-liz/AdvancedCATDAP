# AdvancedCATDAP

Advanced AIC-based Categorical Data Analysis & Preprocessing.

`AdvancedCATDAP` is a Python library for automated feature engineering and selection. It uses the Akaike Information Criterion (AIC) to discover optimal discretizations for numeric variables, groupings for categorical variables, and significant feature interactions.

It supports both **Classification** and **Regression** tasks and is compatible with **Scikit-learn** pipelines.

## Features

- **Automated Discretization**: Finds optimal bins for numeric features using Decision Trees, Quantiles, or Uniform cuts based on AIC.
- **Categorical Grouping**: Automatically groups rare categories to improve model stability.
- **Interaction Discovery**: Detects significant pairwise interactions between features.
- **Feature Selection**: Selects top-k features that contribute most to the model quality.
- **Interpretability**: Provides a detailed `feature_details_` table showing the relationship (counts, target means) for each bin/category.
- **Scikit-learn Compatible**: Implements `BaseEstimator` and `TransformerMixin` for easy integration into `sklearn.pipeline.Pipeline`.
- **Web GUI**: Modern Dash interface with Neon/Glassmorphism design for data upload and analysis.
- **Windows Desktop App**: Standalone `.exe` with native window (no browser required).

## Requirements

- **Python**: 3.12 or higher
- **OS**: Windows (for Desktop App), Linux/macOS (for library usage)

## Installation

```bash
# Using pip
pip install .

# Using uv (Recommended for development)
uv sync --all-extras

# Install with optional dependencies
pip install ".[web]"     # FastAPI + Dash
pip install ".[gui]"     # Dash + Plotly
pip install ".[desktop]" # PyInstaller + pywebview
pip install ".[all]"     # Everything
```

## Quick Start

```python
import pandas as pd
from advanced_catdap import AdvancedCATDAP

# Load data
df = pd.read_csv('data.csv')

# Initialize and analyze
model = AdvancedCATDAP(task_type='classification', verbose=True)
importances, interactions = model.analyze(df, target_col='target')

# View results
print("Top Features:\n", importances.head())
print("Top Interactions:\n", interactions.head())

# Transform data
df_transformed = model.transform(df)
```

---

## GUI Application

AdvancedCATDAP includes a web-based GUI for easy interaction.

### Launching the Web App

```bash
# Using uv (Recommended)
uv run scripts/launch_gui.py

# Or manually
# Terminal 1: API
uvicorn advanced_catdap.service.api:app --reload --port 8000
# Terminal 2: Dash Web App
python advanced_catdap/frontend/dash_app.py
```

### Windows Desktop App (New)

Build and run as a standalone Windows executable:

```powershell
# Build executable
uv run --extra all pyinstaller build.spec

# Run the app
.\dist\AdvancedCATDAP_Native312\AdvancedCATDAP_Native312.exe

# Create installer (requires Inno Setup)
iscc setup.iss
```

The desktop app opens in a native window (not browser) and includes:
- FastAPI backend (auto-started)
- Dash frontend (auto-started)
- Clean shutdown on window close

---

## Project Structure

```
AdvancedCATDAP/
笏懌楳笏 advanced_catdap/           # Main package
笏・  笏懌楳笏 core.py                # AdvancedCATDAP class (main entry point)
笏・  笏懌楳笏 config.py              # Configuration constants
笏・  笏懌楳笏 visualizer.py          # Plotting utilities
笏・  笏懌楳笏 components/            # Core algorithm components
笏・  笏・  笏懌楳笏 discretizer.py     # Discretization strategies
笏・  笏・  笏懌楳笏 scoring.py         # AIC/AICc scoring
笏・  笏・  笏懌楳笏 task_detector.py   # Auto task type detection
笏・  笏・  笏懌楳笏 interaction_searcher.py  # Interaction discovery
笏・  笏・  笏披楳笏 utils.py           # Helper functions
笏・  笏懌楳笏 frontend/              # Dash UI
笏・  笏・  笏懌楳笏 dash_app.py        # Main Dash app
笏・  笏・  笏披楳笏 api_client.py      # API client for frontend
笏・  笏披楳笏 service/               # Backend services
笏・      笏懌楳笏 api.py             # FastAPI endpoints
笏・      笏懌楳笏 analyzer.py        # Analysis orchestration
笏・      笏懌楳笏 job_manager.py     # Background job management
笏・      笏懌楳笏 dataset_manager.py # Dataset storage
笏・      笏披楳笏 schema.py          # Pydantic models
笏懌楳笏 scripts/                   # Utility scripts
笏・  笏懌楳笏 launch_gui.py          # Launch web GUI
笏・  笏懌楳笏 windows_main.py        # Windows desktop entry point
笏・  笏披楳笏 generate_demo_data.py  # Generate sample data
笏懌楳笏 examples/                  # Example scripts
笏懌楳笏 tests/                     # Test suite
笏懌楳笏 build.spec                 # PyInstaller configuration
笏披楳笏 setup.iss                  # Inno Setup installer script
```

---

## API Reference

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/datasets` | Upload CSV/Parquet file |
| `GET` | `/datasets/{id}` | Get dataset metadata |
| `GET` | `/datasets/{id}/preview` | Preview first N rows |
| `POST` | `/jobs` | Submit analysis job |
| `GET` | `/jobs/{id}` | Get job status/results |
| `DELETE` | `/jobs/{id}` | Cancel job |

### Python API

```python
from advanced_catdap import AdvancedCATDAP

model = AdvancedCATDAP(
    task_type='auto',        # 'classification', 'regression', or 'auto'
    use_aicc=True,           # Use corrected AIC
    max_bins=5,              # Max bins for numeric features
    top_k=20,                # Number of top features to keep
    delta_threshold=2.0,     # Minimum AIC improvement
    verbose=True             # Print progress
)

# Fit and analyze
model.fit(X, y)
# or
importances, interactions = model.analyze(df, target_col='target')

# Transform data
X_transformed = model.transform(X)

# Access results
model.feature_importances_     # DataFrame
model.interaction_importances_ # DataFrame
model.feature_details_         # DataFrame
model.transform_rules_         # Dict
```

---

## Usage Examples

### Classification

```python
from advanced_catdap import AdvancedCATDAP

model = AdvancedCATDAP(task_type='classification', verbose=True)
importances, interactions = model.analyze(df, target_col='Churn')

print("Top Features:\n", importances.head())
print("Top Interactions:\n", interactions.head())
```

### Regression

```python
model = AdvancedCATDAP(task_type='regression', verbose=True)
importances, interactions = model.analyze(df, target_col='Price')
```

### Scikit-Learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('preprocessor', AdvancedCATDAP(task_type='classification')),
    ('model', LogisticRegression())
])
pipe.fit(X, y)
```

### Visualization

```python
from advanced_catdap.visualizer import plot_importance, plot_interaction_heatmap

plot_importance(model.feature_importances_)
plot_interaction_heatmap(model.interaction_importances_)
```

---

## Development

### Running Tests

```bash
# Run all tests
uv run pytest -q

# With coverage
uv run pytest --cov=advanced_catdap --cov-report=term-missing

# Run integration tests only
uv run pytest -q -m integration

# Run all tests including integration
uv run pytest -q -m "not integration or integration"

# Current coverage (exporter): 97%
```

### Code Quality

```bash
# Format code
uv run black advanced_catdap tests

# Type checking
uv run mypy advanced_catdap
```

---

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | str | `'auto'` | `'classification'`, `'regression'`, or `'auto'` |
| `use_aicc` | bool | `True` | Use AICc (corrected AIC) for small samples |
| `max_bins` | int | `5` | Maximum bins for numeric discretization |
| `top_k` | int | `20` | Number of top features to keep |
| `delta_threshold` | float | `2.0` | Minimum AIC improvement to select feature |
| `save_rules_mode` | str | `'top_k'` | `'top_k'` or `'all_valid'` |
| `min_cat_fraction` | float | `0.01` | Minimum category frequency |
| `max_categories` | int | `20` | Max categories before grouping |

---

## How It Works

AdvancedCATDAP optimizes feature engineering by minimizing the **Akaike Information Criterion (AIC)**.

1. **Univariate Discretization**: For each feature, tries multiple strategies (Trees, Quantiles, Uniform) and selects the one minimizing AIC.

2. **Feature Selection**: Ranks features by Delta Score (Baseline AIC - Feature AIC) and keeps top-k.

3. **Interaction Discovery**: Tests feature pairs and flags those with significant AIC improvement.

---

## License

MIT


