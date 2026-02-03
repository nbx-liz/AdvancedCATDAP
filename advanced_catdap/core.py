import pandas as pd
import numpy as np
from itertools import combinations
from pandas.api.types import CategoricalDtype
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Union, Optional, Tuple, Any

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

    def __init__(self, use_aicc: bool = True, min_cat_fraction: float = 0.001, max_categories: int = 100,
                 task_type: str = 'auto', min_samples_leaf_rate: float = 0.05, max_leaf_samples: int = 100, 
                 numeric_threshold: float = 0.5, max_pairwise_bins: int = 50000, 
                 n_interaction_candidates: int = 100,
                 max_classification_bytes: int = 200 * 1024 * 1024,
                 max_estimated_uniques: int = 20000, 
                 extra_penalty: float = 0.0, save_rules_mode: str = 'top_k', verbose: bool = True,
                 # Sklearn-style params (promoted from fit)
                 max_bins: int = 5, top_k: int = 20, delta_threshold: float = 0.0):
        
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

        self._reset_state()

    def set_progress_callback(self, cb: Callable[[str, Dict[str, Any]], None]):
        self.progress_cb = cb

    def _notify_progress(self, stage: str, data: Dict[str, Any]):
        if self.progress_cb:
            self.progress_cb(stage, data)

    def _reset_state(self):
        self.mode = None
        self.baseline_score = None
        self.feature_importances_ = None # Renamed from results
        self.interaction_importances_ = None # Renamed from combo_results
        self.transform_rules_ = {} # Renamed from transform_rules
        self.feature_details_ = None

        
        # Backwards compatibility attributes (property-like access handled if needed, or mapped)
        self.results = None
        self.combo_results = None
        self.transform_rules = {}

    def fit(self, X: pd.DataFrame, y: Union[str, np.ndarray, pd.Series] = None, 
            candidates: List[str] = None, max_bins: int = None, top_k: int = None, 
            delta_threshold: float = None, force_categoricals: List[str] = None,
            target_col: str = None):
        """
        Fit the model to dataset.

        Args:
            X: Input dataframe.
            y: Target column name (str) or target array.
            candidates: List of columns to consider.
            ...
        """
        self._reset_state()

        # Resolve parameters (Use arg if provided, else use self.param)
        _max_bins = max_bins if max_bins is not None else self.max_bins
        _top_k = top_k if top_k is not None else self.top_k
        _delta_threshold = delta_threshold if delta_threshold is not None else self.delta_threshold
        
        # --- 1. Data Preparation ---
        # Backwards compatibility: if target_col is passed and y is None
        if target_col is not None and y is None:
            y = target_col
        
        if isinstance(y, str):
            target_col_name = y
            if target_col_name not in X.columns:
                raise ValueError(f"Target column '{target_col_name}' not found.")
            df_working = X.copy()
            target_values = df_working[target_col_name].to_numpy()
        else:
            # sklearn style: fit(X, y)
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

        target_values = df_clean[target_col_name].to_numpy()
        if np.issubdtype(target_values.dtype, np.number):
             target_values = target_values.astype(np.float64)

        # Shared Sampling
        sample_size = min(10000, n_total)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_total, size=sample_size, replace=False)
        sample_mask_full = np.zeros(n_total, dtype=bool)
        sample_mask_full[sample_indices] = True
        
        force_cats_set = set(force_categoricals) if force_categoricals else set()

        self._notify_progress("prepare_data", {"n_rows": n_total, "n_columns": len(cols_needed)})

        # --- 2. Task Detection ---
        self.mode = self._detect_task_type(df_clean[target_col_name])
        if self.verbose:
            print(f"--- Mode: {self.mode.upper()} | Target: '{target_col_name}' | N: {n_total} ---")
        
        self._notify_progress("task_detected", {"mode": self.mode, "target": target_col_name})

        target_sq = None
        target_int = None
        n_classes = 0

        if self.mode == 'regression':
            if not np.issubdtype(target_values.dtype, np.number):
                raise ValueError("Target must be numeric for regression.")
            target_sq = target_values ** 2
            rss_null = np.sum(target_sq) - (np.sum(target_values)**2 / n_total)
            self.baseline_score = self._calc_score(rss_null, 2, n_total, True)

        else: # classification
            target_int, unique_classes = pd.factorize(target_values, sort=True)
            target_int = self._downcast_codes_safe(target_int)
            n_classes = len(unique_classes)
            counts = np.bincount(target_int)
            log_lik_null = np.sum(counts * np.log(counts / n_total + 1e-15))
            self.baseline_score = self._calc_score(log_lik_null, n_classes - 1, n_total, False)

        if self.verbose: print(f"Baseline Score: {self.baseline_score:.2f}")

        self._notify_progress("baseline_calculated", {"score": self.baseline_score})

        # --- 3. Univariate Analysis ---
        uni_results = []
        processed_codes = {} 
        processed_r = {} 
        temp_rules = {}

        min_samples = int(n_total * self.min_samples_leaf_rate)
        min_samples = max(5, min(min_samples, self.max_leaf_samples))

        for i, feature in enumerate(candidates):
            if feature not in df_clean.columns: continue
            
            self._notify_progress("univariate_progress", {"current": i + 1, "total": len(candidates), "feature": feature})
            
            is_forced_cat = feature in force_cats_set
            
            best_codes, score, actual_r, method, rule = self._process_feature_internal(
                df_clean[feature], target_values, target_sq, target_int, n_classes,
                _max_bins, min_samples, sample_indices, sample_mask_full,
                force_category=is_forced_cat
            )
            
            delta = self.baseline_score - score
            uni_results.append({
                'Feature': feature, 'Score': score, 'Delta_Score': delta,
                'Actual_Bins': actual_r, 'Method': method
            })
            
            if delta > _delta_threshold and best_codes is not None:
                processed_codes[feature] = best_codes
                processed_r[feature] = actual_r
                if rule is not None:
                    rule.update({'meta_score': score, 'meta_r': actual_r, 'meta_method': method})
                    temp_rules[feature] = rule

        self.feature_importances_ = pd.DataFrame(uni_results).sort_values('Score').reset_index(drop=True)
        self.results = self.feature_importances_ # alias

        # --- 4. Selection ---
        valid_features = self.feature_importances_[self.feature_importances_['Delta_Score'] > _delta_threshold]['Feature'].tolist()
        available = [f for f in valid_features if f in processed_codes]
        selected = available[:_top_k]
        
        save_targets = selected if self.save_rules_mode == 'top_k' else available
        self.transform_rules_ = {k: temp_rules[k] for k in save_targets if k in temp_rules}
        self.transform_rules = self.transform_rules_ # alias

        processed_codes = {k: processed_codes[k] for k in selected}
        processed_r = {k: processed_r[k] for k in selected}
        
        # --- 5. Interaction Search ---
        self._search_interactions(selected, processed_codes, processed_r, sample_indices, 
                                  target_values, target_sq, target_int, n_classes, n_total)
        
        self._notify_progress("done", {"top_features": selected})

        # --- 6. Feature Details ---
        t_stats = target_values if self.mode == 'regression' else target_int
        self._generate_feature_details(selected, processed_codes, df_clean, t_stats)

        return self
    
    def analyze(self, df, target_col, **kwargs):
        """
        Fit and return analysis results (legacy/convenience API).
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (feature_importances, interaction_importances)
        """
        self.fit(df, target_col=target_col, **kwargs)
        return self.feature_importances_, self.interaction_importances_

    # -------------------------------------------------------------------------
    # Feature Processing Modules
    # -------------------------------------------------------------------------
    def _process_feature_internal(self, raw_series, target, target_sq, target_int, n_classes, 
                                  max_bins, min_samples, sample_indices, sample_mask_full,
                                  force_category=False):
        
        is_numeric_type = pd.api.types.is_numeric_dtype(raw_series)
        numeric_values = None
        
        if not force_category:
            if is_numeric_type:
                numeric_values = raw_series.to_numpy(dtype=float)
            else:
                sample_vals = raw_series.iloc[sample_indices] if hasattr(raw_series, 'iloc') else raw_series[sample_indices]
                sample_conv = pd.to_numeric(sample_vals, errors='coerce')
                if sample_conv.notna().mean() >= self.numeric_threshold:
                    temp = pd.to_numeric(raw_series, errors='coerce')
                    numeric_values = temp.to_numpy(dtype=float)

        if numeric_values is not None:
            return self._discretize_numeric(numeric_values, target, target_sq, target_int, n_classes,
                                          max_bins, min_samples, sample_indices, sample_mask_full)
        else:
            return self._group_categorical(raw_series, target, target_sq, target_int, n_classes, sample_indices)

    def _discretize_numeric(self, numeric_values, target, target_sq, target_int, n_classes, 
                           max_bins, min_samples, sample_indices, sample_mask_full):
        n_total = len(target)
        valid_mask = ~np.isnan(numeric_values)
        if not np.any(valid_mask): 
            return None, self.baseline_score, 1, "all_nan", None
        
        vals_valid = numeric_values[valid_mask]
        valid_sample_mask = valid_mask & sample_mask_full
        
        if np.sum(valid_sample_mask) < 20:
            vals_sample = vals_valid[:1000] 
        else:
            vals_sample = numeric_values[valid_sample_mask]
        
        # Prepare screening targets
        t_screen, tsq_screen, t_int_screen = None, None, None
        if self.mode == 'regression':
            t_screen = target[valid_sample_mask]
            tsq_screen = target_sq[valid_sample_mask]
        else:
            t_int_screen = target_int[valid_sample_mask]

        stats_missing = self._calc_stats_missing(target, target_sq, target_int, n_classes, valid_mask)

        candidates = []
        X_sample_tree, y_sample_tree = None, None
        v_min, v_max = vals_valid.min(), vals_valid.max()
        can_cut = (v_min != v_max)

        for method in ['qcut', 'cut', 'tree']:
            for n_bins in range(2, max_bins + 1):
                try:
                    rule = None; r = 1; codes_sample = None
                    if method == 'tree':
                        if X_sample_tree is None:
                            X_sample_tree = vals_sample.reshape(-1, 1)
                            y_sample_tree = t_screen if self.mode=='regression' else t_int_screen
                        if len(X_sample_tree) < min_samples: continue

                        estimator = DecisionTreeRegressor if self.mode == 'regression' else DecisionTreeClassifier
                        tree = estimator(max_leaf_nodes=n_bins, min_samples_leaf=min_samples, random_state=42)
                        tree.fit(X_sample_tree, y_sample_tree)
                        leaf_ids = tree.apply(X_sample_tree)
                        unique_leaves = np.unique(leaf_ids)
                        mapped = np.searchsorted(unique_leaves, leaf_ids)
                        codes_sample = mapped
                        r = len(unique_leaves) + 1
                        rule = {'type': 'tree', 'model': tree, 'leaves': unique_leaves, 'missing_code': r-1}
                    
                    elif method == 'qcut':
                        quantiles = np.linspace(0, 1, n_bins + 1)
                        bins = np.unique(np.quantile(vals_sample, quantiles))
                        if len(bins) < 2: continue
                        bins[0], bins[-1] = -np.inf, np.inf
                        c_samp = np.searchsorted(bins, vals_sample, side='right') - 1
                        codes_sample = np.clip(c_samp, 0, len(bins) - 2)
                        r = len(bins)
                        rule = {'type': 'qcut', 'bins': bins, 'missing_code': r-1}
                        
                    elif method == 'cut':
                        if not can_cut: continue
                        bins = np.linspace(v_min, v_max, n_bins + 1)
                        bins[0], bins[-1] = -np.inf, np.inf
                        c_samp = np.searchsorted(bins, vals_sample, side='right') - 1
                        codes_sample = np.clip(c_samp, 0, len(bins) - 2)
                        r = len(bins)
                        rule = {'type': 'cut', 'bins': bins, 'missing_code': r-1}
                    
                    if r < 2: continue
                    score_sample = self._calc_score_wrapper(
                        codes_sample, t_screen, tsq_screen, t_int_screen, n_classes, int(codes_sample.max()) + 1
                    )
                    candidates.append((score_sample, method, n_bins, r, rule))
                except Exception: continue

        if not candidates:
            return None, self.baseline_score, 1, "no_candidates", None

        # Best candidate application
        best_cand = min(candidates, key=lambda x: x[0])
        _, best_name, best_nbins, best_r, best_rule = best_cand
        
        codes_valid = None
        if best_rule['type'] == 'tree':
            leaf_ids = best_rule['model'].apply(vals_valid.reshape(-1, 1))
            pos = np.searchsorted(best_rule['leaves'], leaf_ids)
            codes_valid = np.clip(pos, 0, len(best_rule['leaves'])-1)
        elif best_rule['type'] in ['qcut', 'cut']:
            c_valid = np.searchsorted(best_rule['bins'], vals_valid, side='right') - 1
            codes_valid = np.clip(c_valid, 0, len(best_rule['bins']) - 2)

        final_score = self._calc_score_partial(
            codes_valid, target, target_sq, target_int, n_classes, valid_mask, stats_missing, best_r
        )
        
        full_codes = np.full(n_total, best_rule['missing_code'], dtype=int)
        full_codes[valid_mask] = codes_valid
        
        return self._downcast_codes_safe(full_codes), final_score, best_r, f"{best_name}_{best_nbins}({best_r})", best_rule

    def _group_categorical(self, raw_series, target, target_sq, target_int, n_classes, sample_indices):
        n_total = len(target)
        is_high_card, is_id_like = self._check_cardinality_and_id(raw_series, sample_indices)
        if is_id_like: return None, self.baseline_score, 1, "id_col", None
        if is_high_card: return None, self.baseline_score, 1, "high_cardinality", None

        s_obj = raw_series.astype(object)
        codes_raw, uniques = pd.factorize(s_obj, sort=False)
        nan_code = len(uniques)
        codes_for_count = np.where(codes_raw == -1, nan_code, codes_raw)
        counts = np.bincount(codes_for_count, minlength=nan_code+1)
        
        valid_counts = counts[:nan_code]
        n_cats = len(valid_counts)
        freq_threshold = n_total * self.min_cat_fraction
        pass_mask = valid_counts >= freq_threshold
        
        if n_cats > self.max_categories:
            k_idx = self.max_categories
            top_indices = np.argpartition(-valid_counts, kth=min(k_idx, n_cats)-1)[:k_idx]
            final_indices = top_indices[pass_mask[top_indices]]
        else:
            final_indices = np.where(pass_mask)[0]
        
        if len(final_indices) == 0:
            return None, self.baseline_score, 1, "no_valid_category", None

        final_indices = final_indices[np.argsort(-valid_counts[final_indices])]
        n_keep = len(final_indices)
        other_code = n_keep
        missing_code = n_keep + 1
        
        translator = np.full(nan_code + 1, other_code, dtype=int)
        translator[final_indices] = np.arange(n_keep)
        translator[nan_code] = missing_code
        codes = translator[codes_for_count]
        r = missing_code + 1
        
        keep_values = uniques[final_indices]
        value_map = {val: i for i, val in enumerate(keep_values)}
        rule = {'type': 'category', 'value_map': value_map, 'other_code': other_code, 'missing_code': missing_code}
        
        score = self._calc_score_wrapper(codes, target, target_sq, target_int, n_classes, r)
        return self._downcast_codes_safe(codes), score, r, f"category({r})", rule

    # -------------------------------------------------------------------------
    # Interactions
    # -------------------------------------------------------------------------
    def _search_interactions(self, selected_features, processed_codes, processed_r, sample_indices,
                           target_values, target_sq, target_int, n_classes, n_total):
        
        m = len(selected_features)
        if self.verbose: print(f"Combination Search: Top {m} features...")

        combo_candidates = []
        t_samp_reg = target_values[sample_indices] if self.mode == 'regression' else None
        tsq_samp_reg = target_sq[sample_indices] if self.mode == 'regression' else None
        t_samp_cls = target_int[sample_indices] if self.mode == 'classification' else None
        
        sample_codes = {f: processed_codes[f][sample_indices] for f in selected_features}
        
        sample_scores = {}
        for f in selected_features:
            c = sample_codes[f]; r = int(processed_r[f])
            if self.mode == 'regression':
                s, _ = self._calc_score_reg_bincount_idx(t_samp_reg, tsq_samp_reg, c, r)
            else:
                s, _ = self._calc_score_cls_bincount_idx(t_samp_cls, n_classes, c, r, check_memory=False)
            sample_scores[f] = s

        for f1, f2 in combinations(selected_features, 2):
            r1, r2 = int(processed_r[f1]), int(processed_r[f2])
            n_groups = r1 * r2
            if n_groups > self.max_pairwise_bins: continue
            
            c1, c2 = sample_codes[f1], sample_codes[f2]
            comb = c1.astype(np.int64) * r2 + c2.astype(np.int64)
            comb = self._downcast_codes_safe(comb)
            
            if self.mode == 'regression':
                s, _ = self._calc_score_reg_bincount_idx(t_samp_reg, tsq_samp_reg, comb, n_groups)
            else:
                s, _ = self._calc_score_cls_bincount_idx(t_samp_cls, n_classes, comb, n_groups, check_memory=False)
            
            gain = min(sample_scores[f1], sample_scores[f2]) - s
            combo_candidates.append({'f1': f1, 'f2': f2, 'gain_approx': gain, 'r1': r1, 'r2': r2, 'n_groups': n_groups})

        combo_candidates.sort(key=lambda x: x['gain_approx'], reverse=True)
        top_pairs = combo_candidates[:self.n_interaction_candidates] 
        
        final_results = []
        pair_gain_thresh = 2.0 + self.extra_penalty
        score_map = dict(zip(self.feature_importances_["Feature"], self.feature_importances_["Score"]))

        for item in top_pairs:
            f1, f2 = item['f1'], item['f2']
            n_groups = item['n_groups']
            
            if self.mode == 'classification' and (n_groups * n_classes * 8 > self.max_classification_bytes): continue

            c1, c2 = processed_codes[f1], processed_codes[f2]
            comb = c1.astype(np.int64) * item['r2'] + c2.astype(np.int64)
            comb = self._downcast_codes_safe(comb)
            
            if self.mode == 'regression':
                score, k = self._calc_score_reg_bincount_idx(target_values, target_sq, comb, n_groups)
            else:
                score, k = self._calc_score_cls_bincount_idx(target_int, n_classes, comb, n_groups, check_memory=True)
            
            if n_total <= k + 1: continue

            gain = min(score_map[f1], score_map[f2]) - score
            if gain > pair_gain_thresh:
                final_results.append({'Feature_1': f1, 'Feature_2': f2, 'Pair_Score': score, 'Gain': gain})

        if final_results:
            self.interaction_importances_ = pd.DataFrame(final_results).sort_values('Gain', ascending=False)
        else:
            self.interaction_importances_ = pd.DataFrame(columns=['Feature_1', 'Feature_2', 'Gain'])
        
        self.combo_results = self.interaction_importances_ # alias

    # -------------------------------------------------------------------------
    # Core Calculation Methods (Preserved)
    # -------------------------------------------------------------------------
    def _calc_stats_missing(self, target, target_sq, target_int, n_classes, valid_mask):
        missing_mask = ~valid_mask
        n_missing = np.count_nonzero(missing_mask)
        if n_missing == 0: return {'n': 0, 'rss': 0.0, 'loglik_part': 0.0}
        
        stats = {'n': n_missing}
        if self.mode == 'regression':
            t_miss = target[missing_mask]
            sum_y = np.sum(t_miss)
            sum_y2 = np.sum(target_sq[missing_mask])
            rss = sum_y2 - (sum_y**2 / n_missing)
            stats['rss'] = max(rss, 0.0)
        else:
            t_miss = target_int[missing_mask]
            counts = np.bincount(t_miss, minlength=n_classes)
            nz_counts = counts[counts > 0]
            term1 = np.sum(nz_counts * np.log(nz_counts))
            term2 = n_missing * np.log(n_missing)
            stats['loglik_part'] = term1 - term2
        return stats

    def _calc_score_partial(self, codes_valid, target, target_sq, target_int, n_classes, 
                            valid_mask, stats_missing, r):
        n_valid = len(codes_valid)
        n_total = n_valid + stats_missing['n']
        
        if self.mode == 'regression':
            t_valid = target[valid_mask]
            tsq_valid = target_sq[valid_mask]
            counts = np.bincount(codes_valid, minlength=r)
            sum_y = np.bincount(codes_valid, weights=t_valid, minlength=r)
            sum_y2 = np.bincount(codes_valid, weights=tsq_valid, minlength=r)
            
            valid_k_mask = counts > 0
            k_valid = np.count_nonzero(valid_k_mask)
            term2 = np.zeros_like(sum_y)
            term2[valid_k_mask] = (sum_y[valid_k_mask] ** 2) / counts[valid_k_mask]
            rss_valid = np.sum(sum_y2 - term2)
            
            rss_total = max(rss_valid + stats_missing.get('rss', 0.0), 1e-10)
            k_total = k_valid + (1 if stats_missing['n'] > 0 else 0) + 1 
            return self._calc_score(rss_total, k_total, n_total, True)
        else: 
            if r * n_classes * 8 > self.max_classification_bytes: return float('inf')
            t_valid = target_int[valid_mask]
            flat_idx = codes_valid.astype(np.int64) * n_classes + t_valid
            ct_flat = np.bincount(flat_idx, minlength=r * n_classes)
            row_sums = np.bincount(codes_valid, minlength=r)
            
            nz_indices = np.flatnonzero(ct_flat)
            nz_counts = ct_flat[nz_indices]
            group_indices = nz_indices // n_classes
            nz_row_sums = row_sums[group_indices]
            
            term1 = np.sum(nz_counts * np.log(nz_counts))
            term2 = np.sum(nz_counts * np.log(nz_row_sums))
            loglik_valid = term1 - term2
            k_valid = np.count_nonzero(row_sums > 0) * (n_classes - 1)
            
            loglik_total = loglik_valid + stats_missing.get('loglik_part', 0.0)
            k_miss = (n_classes - 1) if stats_missing['n'] > 0 else 0
            k_total = k_valid + k_miss
            return self._calc_score(loglik_total, k_total, n_total, False)

    def _calc_score_wrapper(self, codes, target, target_sq, target_int, n_classes, r):
        if self.mode == 'regression':
            if target_sq is None: target_sq = target ** 2
            score, _ = self._calc_score_reg_bincount_idx(target, target_sq, codes, r)
        else:
            score, _ = self._calc_score_cls_bincount_idx(target_int, n_classes, codes, r, check_memory=False)
        return score

    def _calc_score(self, term_val, k, n, is_regression):
        if n <= k + 1: return float('inf')
        if is_regression:
            rss = max(term_val, 1e-10)
            base_score = n * np.log(rss / n) + 2 * k
        else:
            base_score = -2 * term_val + 2 * k
        if self.use_aicc: base_score += (2 * k * (k + 1)) / (n - k - 1)
        return base_score + (self.extra_penalty * k)

    def _calc_score_reg_bincount_idx(self, target, target_sq, indices, minlength):
        counts = np.bincount(indices, minlength=minlength)
        valid_mask = counts > 0
        k = np.count_nonzero(valid_mask) + 1
        sum_y = np.bincount(indices, weights=target, minlength=minlength)
        sum_y2 = np.bincount(indices, weights=target_sq, minlength=minlength)
        term2 = np.zeros_like(sum_y)
        term2[valid_mask] = (sum_y[valid_mask] ** 2) / counts[valid_mask]
        rss_total = np.sum(sum_y2 - term2)
        return self._calc_score(rss_total, k, len(target), True), k

    def _calc_score_cls_bincount_idx(self, target_int, n_classes, indices, minlength, check_memory=True):
        if check_memory:
            if minlength * n_classes * 8 > self.max_classification_bytes: return float('inf'), 0
        
        flat_idx = indices.astype(np.int64) * n_classes + target_int
        ct_flat = np.bincount(flat_idx, minlength=minlength * n_classes)
        row_sums = np.bincount(indices, minlength=minlength)
        
        nz_indices = np.flatnonzero(ct_flat)
        nz_counts = ct_flat[nz_indices]
        group_indices = nz_indices // n_classes
        nz_row_sums = row_sums[group_indices]
        
        log_lik = np.sum(nz_counts * np.log(nz_counts)) - np.sum(nz_counts * np.log(nz_row_sums))
        k = np.count_nonzero(row_sums > 0) * (n_classes - 1)
        return self._calc_score(log_lik, k, len(target_int), False), k

    def _check_cardinality_and_id(self, series, sample_indices):
        sample_vals = series.iloc[sample_indices] if hasattr(series, 'iloc') else series[sample_indices]
        sample_nunique = sample_vals.nunique()
        sample_len = len(sample_vals)
        n_total = len(series)
        
        estimated_total = n_total if (sample_nunique == sample_len) else sample_nunique * (n_total / sample_len)
        unique_ratio = sample_nunique / sample_len if sample_len > 0 else 0
        
        is_high_card = estimated_total > self.max_estimated_uniques
        is_id_like = (unique_ratio > 0.99) and (estimated_total > 0.99 * n_total)
        return is_high_card, is_id_like

    def _downcast_codes_safe(self, codes):
        if codes.size == 0: return codes.astype(np.int8)
        max_v = codes.max()
        if max_v <= 127: return codes.astype(np.int8)
        elif max_v <= 32767: return codes.astype(np.int16)
        elif max_v <= 2147483647: return codes.astype(np.int32)
        return codes
    
    def _detect_task_type(self, target_series):
        if self.task_type != 'auto': return self.task_type
        if pd.api.types.is_string_dtype(target_series) or isinstance(target_series.dtype, CategoricalDtype): return 'classification'
        n_unique = target_series.nunique()
        if pd.api.types.is_float_dtype(target_series) and n_unique < 20:
             if np.allclose(target_series, target_series.round(), equal_nan=False): return 'classification'
        if n_unique < 20 or (n_unique / len(target_series) < 0.05): return 'classification'
        return 'regression'

    def transform(self, X: pd.DataFrame):
        """
        Apply learned rules to transform input data.

        Args:
            X: Input DataFrame.
        Returns:
            pd.DataFrame: Transformed data with discretized numeric and grouped categorical features.
        """
        if not self.transform_rules_:
            if self.verbose: print("Warning: No rules saved.")
            return pd.DataFrame(index=X.index)
        
        transformed_df = pd.DataFrame(index=X.index)
        for feature, rule in self.transform_rules_.items():
            if feature not in X.columns: continue
            raw_series = X[feature]
            rule_type = rule['type']; missing_code = rule['missing_code']; codes = None
            
            if rule_type in ['qcut', 'cut', 'tree']:
                vals = pd.to_numeric(raw_series, errors='coerce').to_numpy(dtype=float)
                nan_mask = np.isnan(vals); valid_mask = ~nan_mask
                codes = np.full(len(vals), missing_code, dtype=int)
                
                if np.any(valid_mask):
                    vals_valid = vals[valid_mask]
                    if rule_type in ['qcut', 'cut']:
                        c_valid = np.searchsorted(rule['bins'], vals_valid, side='right') - 1
                        codes[valid_mask] = np.clip(c_valid, 0, len(rule['bins']) - 2)
                    elif rule_type == 'tree':
                        l_ids = rule['model'].apply(vals_valid.reshape(-1, 1))
                        pos = np.searchsorted(rule['leaves'], l_ids)
                        codes[valid_mask] = np.clip(pos, 0, len(rule['leaves'])-1)
            
            elif rule_type == 'category':
                s_obj = raw_series.astype(object); isna_mask = s_obj.isna()
                codes = np.full(len(s_obj), rule['other_code'], dtype=int)
                codes[isna_mask] = missing_code
                
                if not isna_mask.all():
                    mapped = s_obj[~isna_mask].map(rule['value_map'])
                    found = mapped.notna()
                    if found.any(): codes[np.where(~isna_mask)[0][found]] = mapped[found].astype(int)
            
            if codes is not None: transformed_df[feature] = self._downcast_codes_safe(codes)
            
        return transformed_df

    def _generate_feature_details(self, selected_features, processed_codes, df_origin, target_vals):
        details_list = []
        
        for feature in selected_features:
            if feature not in self.transform_rules_:
                continue
            
            rule = self.transform_rules_[feature]
            codes = processed_codes[feature]
            
            rule_type = rule['type']
            is_numeric_rule = rule_type in ['qcut', 'cut', 'tree']
            
            if is_numeric_rule:
                feat_vals = pd.to_numeric(df_origin[feature], errors='coerce').to_numpy()
            else:
                feat_vals = df_origin[feature].to_numpy()
            
            tmp_df = pd.DataFrame({
                'code': codes,
                'target': target_vals,
                'feat': feat_vals
            })
            
            groups = tmp_df.groupby('code')
            
            for code, grp in groups:
                 cnt = len(grp)
                 t_mean = grp['target'].mean()
                 
                 f_min = None
                 f_max = None
                 if is_numeric_rule:
                     f_min = grp['feat'].min()
                     f_max = grp['feat'].max()
                 
                 label = self._get_bin_label(code, rule, f_min, f_max)
                 
                 details_list.append({
                     'Feature': feature,
                     'Bin_Idx': int(code),
                     'Bin_Label': label,
                     'Count': int(cnt),
                     'Target_Mean': float(t_mean)
                 })
            
        if details_list:
            self.feature_details_ = pd.DataFrame(details_list)
            self.feature_details_ = self.feature_details_.sort_values(['Feature', 'Bin_Idx']).reset_index(drop=True)
        else:
            self.feature_details_ = pd.DataFrame(columns=['Feature', 'Bin_Idx', 'Bin_Label', 'Count', 'Target_Mean'])

    def _get_bin_label(self, code, rule, min_val=None, max_val=None):
        if code == rule.get('missing_code'):
            return "Missing"
        
        r_type = rule['type']
        
        if r_type == 'category':
            if code == rule.get('other_code'):
                return "Other"
            
            val_map = rule.get('value_map', {})
            cats = [str(k) for k, v in val_map.items() if v == code]
            if len(cats) == 1:
                return cats[0]
            elif len(cats) > 1:
                return ",".join(cats[:3]) + ("..." if len(cats)>3 else "")
            return f"Cat_{code}"
            
        elif r_type in ['qcut', 'cut']:
            bins = rule['bins']
            if 0 <= code < len(bins) - 1:
                lower, upper = bins[code], bins[code+1]
                l_str = "-inf" if np.isinf(lower) else f"{lower:.3f}"
                u_str = "inf" if np.isinf(upper) else f"{upper:.3f}"
                return f"({l_str}, {u_str}]"
        
        elif r_type == 'tree':
            if min_val is not None and max_val is not None:
                return f"[{min_val:.3f}, {max_val:.3f}]"
            return f"TreeLeaf_{code}"
            
        return f"Bin_{code}"

