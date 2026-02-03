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

## How It Works

AdvancedCATDAP optimizes feature engineering by minimizing the **Akaike Information Criterion (AIC)**.

1.  **Univariate Discretization**:
    - For each feature, it attempts multiple discretization strategies (Decision Trees, Quantile Cuts, Uniform Cuts) with varying numbers of bins.
    - It selects the discretization that minimizes the AIC of a single-variable model (predicting the target).
    - If the best AIC is significantly lower than the baseline (null model), the feature is considered informative.

2.  **Feature Selection**:
    - Features are ranked by their "Delta Score" (Baseline AIC - Feature AIC).
    - Only the top features (controlled by `top_k`) that improve the model are retained.

3.  **Interaction Discovery**:
    - The algorithm searches for pairs of selected features.
    - It creates a new combined feature (Cartesian product of bins) and measures its AIC.
    - If the combined feature's AIC is lower than the best single feature's AIC (by a significant margin), it flags a significant interaction.

This data-driven approach ensures that features are transformed in a way that maximizes predictive power while penalizing complexity (overfitting).


## Installation

```bash
pip install .
```

## Usage



### 1. Unified API (Pandas-like)

AdvancedCATDAP uses a unified interface for both **Classification** and **Regression**. You simply specify the `task_type` (or let it auto-detect) and use the `fit()` or `analyze()` methods.

**Common Workflow:**

The API is consistent regardless of the task. You primarily switch the `task_type`.

**A. Classification Example**

```python
import pandas as pd
from advanced_catdap import AdvancedCATDAP

# 1. Load Data
df = pd.read_csv('churn_data.csv')

# 2. Initialize
# task_type='classification' treats the target as categories (or binary).
# 'save_rules_mode="top_k"' keeps only the most important features.
model = AdvancedCATDAP(
    task_type='classification', 
    verbose=True,
    save_rules_mode='top_k' 
)

# 3. Analyze (Fit & Get Importances)
# 'analyze' calculates AIC scores and finds the best discretization.
importances, interactions = model.analyze(df, target_col='Churn')

print("Top Features:\n", importances.head())
print("Top Interactions:\n", interactions.head())

# 4. Transform (Apply Discretization Rules)
# Converts numeric/categorical columns into discrete bins optimized for the target.
df_transformed = model.transform(df)
```

**B. Regression Example**

For regression, the library finds bins that explain the variance in the continuous target.

```python
import pandas as pd
from advanced_catdap import AdvancedCATDAP

# 1. Load Data
df = pd.read_csv('ltv_data.csv')

# 2. Initialize
# task_type='regression' treats the target as a continuous value.
model = AdvancedCATDAP(
    task_type='regression',
    verbose=True
)

# 3. Analyze
importances, interactions = model.analyze(df, target_col='LTV')

print("Top Drivers of LTV:\n", importances.head())

# 4. Feature Details (Interpretability)
# Inspect specific bins and their target means (e.g. "Users aged 20-30 have avg LTV of $500")
if model.feature_details_ is not None:
    print(model.feature_details_.head(10))
```

### 2. Scikit-Learn API (Pipeline Compatible)

AdvancedCATDAP is fully compatible with Scikit-learn's `fit(X, y)` and `transform(X)` API, making it ideal for Pipelines.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Input X (DataFrame) and y (Series/Array)
X = df.drop(columns=['target'])
y = df['target']

pipe = Pipeline([
    # Step 1: Discretize & Engineer Features
    ('preprocessor', AdvancedCATDAP(task_type='classification', max_bins=5)),
    # Step 2: Feed into a linear model (which benefits from discretized non-linearities)
    ('model', LogisticRegression())
])

pipe.fit(X, y)
```

### 3. Detailed Feature Analysis

After fitting, you can inspect `feature_details_` to understand *why* a feature is important. It shows the binned ranges and the average target value for each bin.

```python
# Assuming model is fitted
details = model.feature_details_

# Example Output:
#        Feature         Bin_Label  Count  Target_Mean
# 0          Age  [18.000, 41.000]    553     0.094033
# 1          Age  [42.000, 46.000]    160     0.031250
# ...
```

- **Bin_Label**: The numeric range (e.g. `(0.1, 10.5]`) or category name.
- **Count**: Number of samples in that bin.
- **Target_Mean**: Average target value (Regression) or Positive Class Probability (Binary Classification).



### 4. Visualization

(Requires `matplotlib` / `seaborn` installed separately if not included in deps)

```python
from advanced_catdap import plot_importance, plot_interaction_heatmap

# Visualize Feature Importances
plot_importance(model.feature_importances_)

# Visualize Interaction Heatmap
plot_interaction_heatmap(model.interaction_importances_)
```

## Examples

Check the `advanced_catdap/examples/` directory for ready-to-run scripts:

- `01_basic_classification.py`: Complete classification workflow.
- `02_basic_regression.py`: Complete regression workflow.
- `03_sklearn_pipeline.py`: Pipeline integration example.
- `04_visualization.py`: Visualization demo.
- `05_error_handling.py`: Handling edge cases.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | str | `'auto'` | `'classification'`, `'regression'`, or `'auto'` (detects based on target). |
| `use_aicc` | bool | `True` | Use AICc (corrected AIC) instead of raw AIC to punish complexity more in small samples. |
| `max_bins` | int | `5` | Maximum number of bins for numeric discretization. |
| `top_k` | int | `20` | Number of top features to keep/transform. |
| `save_rules_mode` | str | `'top_k'` | `'top_k'` (save only selected features) or `'all_valid'` (save all that improve AIC). |

## License

MIT
