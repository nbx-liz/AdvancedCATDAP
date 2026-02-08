
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Union, Optional, Tuple, Any, Callable

from .config import *
from .components.task_detector import TaskDetector
from .components.scoring import Scorer
from .components.discretizer import Discretizer
from .components.interaction_searcher import InteractionSearcher

class AdvancedCATDAP(BaseEstimator, TransformerMixin):
    """
    Advanced AIC-based Categorical Data Analysis & Preprocessing.

    This class provides automated feature discretization, grouping, and interaction discovery
    based on AIC (Akaike Information Criterion). It handles both regression and classification tasks.

    Attributes:
        feature_importances_ (pd.DataFrame): Importance scores of single variables.
        interaction_importances_ (pd.DataFrame): Importance scores of feature interactions.
        transform_rules_ (Dict): Rules for transforming features (bins, categories, etc.).
    """

    def __init__(self, use_aicc: bool = DEFAULT_USE_AICC, 
                 min_cat_fraction: float = DEFAULT_MIN_CAT_FRACTION, 
                 max_categories: int = DEFAULT_MAX_CATEGORIES,
                 task_type: str = DEFAULT_TASK_TYPE, 
                 min_samples_leaf_rate: float = DEFAULT_MIN_SAMPLES_LEAF_RATE, 
                 max_leaf_samples: int = DEFAULT_MAX_LEAF_SAMPLES, 
                 numeric_threshold: float = DEFAULT_NUMERIC_THRESHOLD, 
                 max_pairwise_bins: int = DEFAULT_MAX_PAIRWISE_BINS, 
                 n_interaction_candidates: int = DEFAULT_N_INTERACTION_CANDIDATES,
                 max_classification_bytes: int = DEFAULT_MAX_CLASSIFICATION_BYTES,
                 max_estimated_uniques: int = DEFAULT_MAX_ESTIMATED_UNIQUES, 
                 extra_penalty: float = DEFAULT_EXTRA_PENALTY, 
                 save_rules_mode: str = DEFAULT_SAVE_RULES_MODE, 
                 verbose: bool = DEFAULT_VERBOSE,
                 # Sklearn-style params (promoted from fit)
                 max_bins: int = DEFAULT_MAX_BINS, 
                 top_k: int = DEFAULT_TOP_K, 
                 delta_threshold: float = DEFAULT_DELTA_THRESHOLD):
        
        if save_rules_mode not in ('top_k', 'all_valid'):
            raise ValueError("save_rules_mode must be 'top_k' or 'all_valid'")

        self.use_aicc = use_aicc
        self.min_cat_fraction = min_cat_fraction
        self.max_categories = max_categories
        self.task_type = task_type
        self.min_samples_leaf_rate = min_samples_leaf_rate
        self.max_leaf_samples = max_leaf_samples
        self.numeric_threshold = numeric_threshold
        self.max_pairwise_bins = max_pairwise_bins
        self.n_interaction_candidates = n_interaction_candidates
        self.max_classification_bytes = max_classification_bytes
        self.max_estimated_uniques = max_estimated_uniques
        self.extra_penalty = extra_penalty
        self.save_rules_mode = save_rules_mode
        self.verbose = verbose
        
        # New attributes
        self.max_bins = max_bins
        self.top_k = top_k
        self.delta_threshold = delta_threshold
        
        self.progress_cb = None
        
        # Components
        self.scorer = Scorer(use_aicc=use_aicc, extra_penalty=extra_penalty, 
                             max_classification_bytes=max_classification_bytes)
        
        # We instantiate others lazily or efficiently in fit? 
        # Better to have them ready if check_estimator needs them? 
        # But they depend on task_type which might be auto.
        self.task_detector = TaskDetector(task_type=task_type, use_aicc=use_aicc, 
                                          extra_penalty=extra_penalty)
        
        self.discretizer = None
        self.interaction_searcher = None

        self._reset_state()

    def set_progress_callback(self, cb: Callable[[str, Dict[str, Any]], None]):
        self.progress_cb = cb

    def _notify_progress(self, stage: str, data: Dict[str, Any]):
        if self.progress_cb:
            self.progress_cb(stage, data)

    def _reset_state(self):
        self.mode = None
        self.baseline_score = None
        
        # Public attributes expected by users
        self.feature_details_ = None
        
        # Backwards compatibility attributes
        self.results = None
        self.combo_results = None
        self.transform_rules = {}

    @property
    def feature_importances_(self):
        return self.discretizer.feature_importances_ if self.discretizer else None
        
    @property
    def transform_rules_(self):
        return self.discretizer.transform_rules_ if self.discretizer else {}

    @property
    def interaction_importances_(self):
        return self.interaction_searcher.interaction_importances_ if self.interaction_searcher else None

    def fit(self, X: pd.DataFrame, y: Union[str, np.ndarray, pd.Series] = None, 
            candidates: List[str] = None, max_bins: int = None, top_k: int = None, 
            delta_threshold: float = None, force_categoricals: List[str] = None,
            ordered_categoricals: List[str] = None,
            category_orders: Dict[str, List[str]] = None,
            label_prefix_style: str = None,
            target_col: str = None):
        """
        Fit the model to dataset.
        """
        self._reset_state()

        # Resolve parameters
        _max_bins = max_bins if max_bins is not None else self.max_bins
        _top_k = top_k if top_k is not None else self.top_k
        _delta_threshold = delta_threshold if delta_threshold is not None else self.delta_threshold
        
        # --- 1. Data Preparation ---
        if target_col is not None and y is None:
            y = target_col
        
        if isinstance(y, str):
            target_col_name = y
            if target_col_name not in X.columns:
                raise ValueError(f"Target column '{target_col_name}' not found.")
            df_working = X.copy()
            target_values = df_working[target_col_name].to_numpy()
        else:
            if y is None: raise ValueError("Target 'y' must be provided.")
            df_working = X.copy()
            target_col_name = "__target__"
            target_values = np.asarray(y)
            df_working[target_col_name] = target_values

        if candidates is None:
            candidates = [c for c in df_working.columns if c != target_col_name]

        cols_needed = [target_col_name] + [c for c in candidates if c in df_working.columns]
        df_clean = df_working[cols_needed].dropna(subset=[target_col_name])
        n_total = len(df_clean)
        if n_total == 0: raise ValueError("Data is empty.")

        target_values = df_clean[target_col_name].to_numpy() # Update after dropna
        
        # Ensure target is float if regression (TaskDetector check handles this but we might need casting)
        # We rely on TaskDetector
        
        # Shared Sampling
        sample_size = min(10000, n_total)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_total, size=sample_size, replace=False)
        sample_mask_full = np.zeros(n_total, dtype=bool)
        sample_mask_full[sample_indices] = True
        
        self._notify_progress("prepare_data", {"n_rows": n_total, "n_columns": len(cols_needed)})

        # --- 2. Task Detection ---
        self.mode = self.task_detector.detect(df_clean[target_col_name])
        if self.verbose:
            print(f"--- Mode: {self.mode.upper()} | Target: '{target_col_name}' | N: {n_total} ---")
        self._notify_progress("task_detected", {"mode": self.mode, "target": target_col_name})

        # Calculate Baseline
        # Prepare targets for scoring
        target_sq = None
        target_int = None
        n_classes = 0
        
        self.baseline_score, baseline_info = self.task_detector.calc_baseline(target_values, self.mode)
        
        if self.mode == 'regression':
             target_sq = baseline_info['target_sq']
        else:
             target_int = baseline_info['target_int']
             n_classes = baseline_info['n_classes']
             
        if self.verbose: print(f"Baseline Score: {self.baseline_score:.2f}")
        self._notify_progress("baseline_calculated", {"score": self.baseline_score})

        # --- 3. Univariate Analysis (Discretization) ---
        self.discretizer = Discretizer(
            task_type=self.mode, scorer=self.scorer,
            numeric_threshold=self.numeric_threshold,
            min_samples_leaf_rate=self.min_samples_leaf_rate,
            max_leaf_samples=self.max_leaf_samples,
            max_bins=_max_bins,
            max_categories=self.max_categories,
            min_cat_fraction=self.min_cat_fraction,
            max_estimated_uniques=self.max_estimated_uniques,
            delta_threshold=_delta_threshold,
            top_k=_top_k,
            save_rules_mode=self.save_rules_mode
        )

        self.discretizer.fit(
            X=df_clean[candidates], y=target_values,
            baseline_score=self.baseline_score,
            target_sq=target_sq, target_int=target_int, n_classes=n_classes,
            sample_indices=sample_indices, sample_mask_full=sample_mask_full,
            force_categoricals=force_categoricals,
            ordered_categoricals=ordered_categoricals,
            category_orders=category_orders,
            label_prefix_style=label_prefix_style,
            progress_callback=self._notify_progress
        )

        # Alias for backward compat
        self.results = self.discretizer.feature_importances_
        self.transform_rules = self.discretizer.transform_rules_
        
        # --- 4. Interaction Search ---
        selected_features = list(self.discretizer.processed_codes_.keys())
        # Note: processed_codes_ in discretizer contains only the cached ones. 
        # Discretizer filtering logic ensures we have codes for candidates that passed delta_threshold (and top_k).
        
        self.interaction_searcher = InteractionSearcher(
            task_type=self.mode, scorer=self.scorer,
            max_pairwise_bins=self.max_pairwise_bins,
            n_interaction_candidates=self.n_interaction_candidates,
            extra_penalty=self.extra_penalty,
            verbose=self.verbose
        )

        self.interaction_searcher.search(
            selected_features=selected_features,
            processed_codes=self.discretizer.processed_codes_,
            processed_r=self.discretizer.processed_r_,
            feature_scores=dict(zip(self.results['Feature'], self.results['Score'])),
            sample_indices=sample_indices,
            target_values=target_values,
            target_sq=target_sq, target_int=target_int,
            n_classes=n_classes, n_total=n_total
        )
        
        self.combo_results = self.interaction_searcher.interaction_importances_

        self._notify_progress("done", {"top_features": selected_features})

        # --- 5. Feature Details ---
        t_stats = target_values if self.mode == 'regression' else target_int
        self.feature_details_ = self.discretizer.get_feature_details(df_clean, t_stats)

        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply learned rules to transform input data.
        """
        if self.discretizer is None:
             if self.verbose: print("Warning: Model not fitted.")
             return pd.DataFrame(index=X.index)
        return self.discretizer.transform(X)

    def analyze(self, df, target_col, **kwargs):
        """
        Fit and return analysis results (legacy/convenience API).
        """
        self.fit(df, target_col=target_col, **kwargs)
        return self.feature_importances_, self.interaction_importances_
