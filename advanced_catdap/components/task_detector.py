
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from typing import Tuple, Union, Optional, Any
from ..config import EPSILON_LOG

class TaskDetector:
    """
    Detects the task type (regression/classification) and calculates baseline scores.
    """
    def __init__(self, task_type: str = 'auto', use_aicc: bool = True, extra_penalty: float = 0.0):
        self.task_type = task_type
        self.use_aicc = use_aicc
        self.extra_penalty = extra_penalty

    def detect(self, target_series: pd.Series) -> str:
        """
        Detects whether the task is regression or classification based on the target variable.
        """
        if self.task_type != 'auto':
            return self.task_type
        
        if pd.api.types.is_string_dtype(target_series) or isinstance(target_series.dtype, CategoricalDtype):
            return 'classification'
        
        n_unique = target_series.nunique()
        # Heuristic: If float and low unique count, check if they are effectively integers
        if pd.api.types.is_float_dtype(target_series) and n_unique < 20:
             if np.allclose(target_series, target_series.round(), equal_nan=False):
                 return 'classification'
                 
        # Heuristic: Low cardinality relative to data size
        if n_unique < 20 or (n_unique / len(target_series) < 0.05):
            return 'classification'
            
        return 'regression'

    def calc_baseline(self, target_values: np.ndarray, task_mode: str) -> Tuple[float, Any]:
        """
        Calculates the baseline score (AIC) for the null model.
        Returns: (baseline_score, additional_info)
        """
        n_total = len(target_values)
        
        if task_mode == 'regression':
            if not np.issubdtype(target_values.dtype, np.number):
                 raise ValueError("Target must be numeric for regression.")
                 
            target_sq = target_values ** 2
            rss_null = np.sum(target_sq) - (np.sum(target_values)**2 / n_total)
            score = self._calc_aic_score(rss_null, 2, n_total, is_regression=True)
            return score, {'target_sq': target_sq}
        
        else: # classification
            target_int, unique_classes = pd.factorize(target_values, sort=True)
            # Downcast not strictly necessary for local calc but good practice
            # target_int = ... 
            n_classes = len(unique_classes)
            counts = np.bincount(target_int)
            log_lik_null = np.sum(counts * np.log(counts / n_total + EPSILON_LOG))
            score = self._calc_aic_score(log_lik_null, n_classes - 1, n_total, is_regression=False)
            return score, {'target_int': target_int, 'n_classes': n_classes, 'unique_classes': unique_classes}

    def _calc_aic_score(self, term_val: float, k: int, n: int, is_regression: bool) -> float:
        """
        Calculates AIC/AICc score.
        term_val: RSS for regression, LogLikelihood for classification.
        """
        if n <= k + 1:
            return float('inf')
            
        if is_regression:
            rss = max(term_val, 1e-10)
            base_score = n * np.log(rss / n) + 2 * k
        else:
            # For classification, term_val is LogLik (positive or negative depending on formulation?)
            # In core.py: log_lik_null = sum(counts * log(probs)). This is negative log lik? No, log(p) < 0.
            # And core.py formula: -2 * term_val + 2 * k
            # Yes, standard AIC = 2k - 2ln(L)
            base_score = -2 * term_val + 2 * k

        if self.use_aicc:
            base_score += (2 * k * (k + 1)) / (n - k - 1)
            
        return base_score + (self.extra_penalty * k)
