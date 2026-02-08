
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from ..config import *
from .scoring import Scorer
from .utils import downcast_codes_safe, check_cardinality_and_id
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class DiscretizationStrategy(ABC):
    """Abstract base class for discretization strategies."""
    
    @abstractmethod
    def discretize(self, vals_sample: np.ndarray, n_bins: int, 
                   min_samples: int, task_type: str, 
                   y_sample: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], int, Optional[Dict[str, Any]]]:
        """
        Returns:
            codes_sample: Discretized codes for the sample.
            r: Number of bins (cardinality).
            rule: Rule dictionary for reproduction.
        """
        pass

class TreeStrategy(DiscretizationStrategy):
    def discretize(self, vals_sample: np.ndarray, n_bins: int, 
                   min_samples: int, task_type: str, 
                   y_sample: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], int, Optional[Dict[str, Any]]]:
        if y_sample is None: return None, 1, None
        X_sample = vals_sample.reshape(-1, 1)
        if len(X_sample) < min_samples: return None, 1, None
        
        estimator = DecisionTreeRegressor if task_type == 'regression' else DecisionTreeClassifier
        tree = estimator(max_leaf_nodes=n_bins, min_samples_leaf=min_samples, random_state=42)
        tree.fit(X_sample, y_sample)
        
        leaf_ids = tree.apply(X_sample)
        unique_leaves = np.unique(leaf_ids)
        codes_sample = np.searchsorted(unique_leaves, leaf_ids)
        r = len(unique_leaves) + 1
        rule = {'type': 'tree', 'model': tree, 'leaves': unique_leaves, 'missing_code': r-1}
        return codes_sample, r, rule

class QuantileStrategy(DiscretizationStrategy):
    def discretize(self, vals_sample: np.ndarray, n_bins: int, 
                   min_samples: int, task_type: str, 
                   y_sample: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], int, Optional[Dict[str, Any]]]:
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.unique(np.quantile(vals_sample, quantiles))
        if len(bins) < 2: return None, 1, None
        
        bins[0], bins[-1] = -np.inf, np.inf
        c_samp = np.searchsorted(bins, vals_sample, side='right') - 1
        codes_sample = np.clip(c_samp, 0, len(bins) - 2)
        r = len(bins)
        rule = {'type': 'qcut', 'bins': bins, 'missing_code': r-1}
        return codes_sample, r, rule

class UniformStrategy(DiscretizationStrategy):
    def discretize(self, vals_sample: np.ndarray, n_bins: int, 
                   min_samples: int, task_type: str, 
                   y_sample: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], int, Optional[Dict[str, Any]]]:
        v_min, v_max = vals_sample.min(), vals_sample.max()
        if v_min == v_max: return None, 1, None
        
        bins = np.linspace(v_min, v_max, n_bins + 1)
        bins[0], bins[-1] = -np.inf, np.inf
        c_samp = np.searchsorted(bins, vals_sample, side='right') - 1
        codes_sample = np.clip(c_samp, 0, len(bins) - 2)
        r = len(bins)
        rule = {'type': 'cut', 'bins': bins, 'missing_code': r-1}
        return codes_sample, r, rule


class Discretizer(BaseEstimator, TransformerMixin):
    """
    Handles discretization of numeric features and grouping of categorical features.
    """
    def __init__(self, 
                 task_type: str,
                 scorer: Scorer,
                 numeric_threshold: float = DEFAULT_NUMERIC_THRESHOLD,
                 integer_as_level_threshold: int = DEFAULT_INTEGER_AS_LEVEL_THRESHOLD,
                 min_samples_leaf_rate: float = DEFAULT_MIN_SAMPLES_LEAF_RATE,
                 max_leaf_samples: int = DEFAULT_MAX_LEAF_SAMPLES,
                 max_bins: int = DEFAULT_MAX_BINS,
                 max_categories: int = DEFAULT_MAX_CATEGORIES,
                 min_cat_fraction: float = DEFAULT_MIN_CAT_FRACTION,
                 max_estimated_uniques: int = DEFAULT_MAX_ESTIMATED_UNIQUES,
                 delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
                 top_k: int = DEFAULT_TOP_K,
                 save_rules_mode: str = DEFAULT_SAVE_RULES_MODE,
                 label_prefix_style: str = DEFAULT_LABEL_PREFIX_STYLE):
        
        self.task_type = task_type
        self.scorer = scorer
        self.numeric_threshold = numeric_threshold
        self.integer_as_level_threshold = integer_as_level_threshold
        self.min_samples_leaf_rate = min_samples_leaf_rate
        self.max_leaf_samples = max_leaf_samples
        self.max_bins = max_bins
        self.max_categories = max_categories
        self.min_cat_fraction = min_cat_fraction
        self.max_estimated_uniques = max_estimated_uniques
        self.delta_threshold = delta_threshold
        self.top_k = top_k
        self.save_rules_mode = save_rules_mode
        self.label_prefix_style = label_prefix_style
        self.ordered_categoricals_ = set()
        self.category_orders_ = {}
        
        # State
        self.transform_rules_ = {}
        self.feature_importances_ = None
        self.processed_codes_ = {} # Caches for InteractionSearcher
        self.processed_r_ = {}
        
        # Strategies
        self.strategies = {
            'tree': TreeStrategy(),
            'qcut': QuantileStrategy(),
            'cut': UniformStrategy()
        }
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            baseline_score: float,
            target_sq: Optional[np.ndarray] = None,
            target_int: Optional[np.ndarray] = None,
            n_classes: int = 0,
            sample_indices: Optional[np.ndarray] = None,
            sample_mask_full: Optional[np.ndarray] = None,
            force_categoricals: List[str] = None,
            ordered_categoricals: List[str] = None,
            category_orders: Optional[Dict[str, List[str]]] = None,
            label_prefix_style: Optional[str] = None,
            progress_callback: Optional[Callable] = None):
        
        n_total = len(X)
        candidates = X.columns.tolist()
        force_cats_set = set(force_categoricals) if force_categoricals else set()
        ordered_cats_set = set(ordered_categoricals) if ordered_categoricals else set()
        self.ordered_categoricals_ = ordered_cats_set
        self.category_orders_ = category_orders or {}
        if label_prefix_style in {"none", "numeric_only", "all_bins"}:
            self.label_prefix_style = label_prefix_style
        
        uni_results = []
        temp_rules = {}
        self.processed_codes_ = {}
        self.processed_r_ = {}
        
        # Calculate leaf parameters
        min_samples = int(n_total * self.min_samples_leaf_rate)
        min_samples = max(5, min(min_samples, self.max_leaf_samples))
        
        for i, feature in enumerate(candidates):
            if progress_callback:
                progress_callback("univariate_progress", {"current": i + 1, "total": len(candidates), "feature": feature})
            
            is_forced_cat = feature in force_cats_set
            explicit_order = self.category_orders_.get(feature)
            is_ordered_cat = feature in ordered_cats_set or explicit_order is not None
            
            best_codes, score, actual_r, method, rule = self._process_feature_internal(
                X[feature], y, target_sq, target_int, n_classes,
                self.max_bins, min_samples, sample_indices, sample_mask_full,
                self.scorer.calc_score_wrapper if hasattr(self.scorer, 'calc_score_wrapper') else None, # We don't have wrapper in scorer
                baseline_score,
                force_category=is_forced_cat,
                ordered_category=is_ordered_cat,
                category_order=explicit_order,
                feature_name=feature
            )
            
            delta = baseline_score - score
            uni_results.append({
                'Feature': feature, 'Score': score, 'Delta_Score': delta,
                'Actual_Bins': actual_r, 'Method': method
            })
            
            if delta > self.delta_threshold and best_codes is not None:
                self.processed_codes_[feature] = best_codes
                self.processed_r_[feature] = actual_r
                if rule is not None:
                    rule.update({'meta_score': score, 'meta_r': actual_r, 'meta_method': method})
                    temp_rules[feature] = rule

        self.feature_importances_ = pd.DataFrame(uni_results).sort_values('Score').reset_index(drop=True)
        
        # Selection
        valid_features = self.feature_importances_[self.feature_importances_['Delta_Score'] > self.delta_threshold]['Feature'].tolist()
        available = [f for f in valid_features if f in self.processed_codes_]
        selected = available[:self.top_k]
        
        save_targets = selected if self.save_rules_mode == 'top_k' else available
        self.transform_rules_ = {k: temp_rules[k] for k in save_targets if k in temp_rules}
        
        # Filter cache to selected only (or save targets?)
        # For interactions, we usually want Top K
        self.processed_codes_ = {k: self.processed_codes_[k] for k in selected}
        self.processed_r_ = {k: self.processed_r_[k] for k in selected}
        
        return self

    def _process_feature_internal(self, raw_series, target, target_sq, target_int, n_classes, 
                                  max_bins, min_samples, sample_indices, sample_mask_full,
                                  scorer_wrapper_unused, baseline_score,
                                  force_category=False,
                                  ordered_category=False,
                                  category_order: Optional[List[str]] = None,
                                  feature_name="unknown"):
        
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
            if self._should_treat_as_integer_levels(numeric_values):
                return self._group_integer_levels(
                    numeric_values, target, target_sq, target_int, n_classes, baseline_score
                )
            return self._discretize_numeric(numeric_values, target, target_sq, target_int, n_classes,
                                          max_bins, min_samples, sample_indices, sample_mask_full, baseline_score, feature_name)
        else:
            return self._group_categorical(
                raw_series,
                target,
                target_sq,
                target_int,
                n_classes,
                sample_indices,
                baseline_score,
                ordered=ordered_category,
                category_order=category_order,
            )

    def _discretize_numeric(self, numeric_values, target, target_sq, target_int, n_classes, 
                           max_bins, min_samples, sample_indices, sample_mask_full, baseline_score, feature_name="unknown"):
        # ... Implementation from core.py ...
        valid_mask = ~np.isnan(numeric_values)
        if not np.any(valid_mask): 
            return None, baseline_score, 1, "all_nan", None
        
        vals_valid = numeric_values[valid_mask]
        valid_sample_mask = valid_mask & sample_mask_full
        
        if np.sum(valid_sample_mask) < 20:
            vals_sample = vals_valid[:1000] 
        else:
            vals_sample = numeric_values[valid_sample_mask]
        
        # Prepare screening targets
        t_screen, tsq_screen, t_int_screen = None, None, None
        if self.task_type == 'regression':
            t_screen = target[valid_sample_mask]
            tsq_screen = target_sq[valid_sample_mask]
        else:
            t_int_screen = target_int[valid_sample_mask]

        stats_missing = self.scorer.calc_stats_missing(target, target_sq, target_int, n_classes, valid_mask, self.task_type)

        candidates = []
        X_sample_tree, y_sample_tree = None, None
        
        # Prepare targets for tree strategy once
        if self.task_type == 'regression':
             y_sample_tree = t_screen
        else:
             y_sample_tree = t_int_screen

        for method, strategy in self.strategies.items():
            for n_bins in range(2, max_bins + 1):
                try:
                    codes_sample, r, rule = strategy.discretize(vals_sample, n_bins, min_samples, self.task_type, y_sample_tree)
                    
                    if codes_sample is None or r < 2: continue
                    
                    score_sample = self._calc_score_wrapper(
                        codes_sample, t_screen, tsq_screen, t_int_screen, n_classes, int(codes_sample.max()) + 1
                    )
                    candidates.append((score_sample, method, n_bins, r, rule))
                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.debug(f"Discretization candidate failed for feature {feature_name} with method {method}, bins {n_bins}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected error in discretization screening for feature {feature_name}, method {method}: {e}")
                    continue

        if not candidates:
            return None, baseline_score, 1, "no_candidates", None

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
        
        full_codes = np.full(len(target), best_rule['missing_code'], dtype=int)
        full_codes[valid_mask] = codes_valid
        
        return downcast_codes_safe(full_codes), final_score, best_r, f"{best_name}_{best_nbins}({best_r})", best_rule

    def _should_treat_as_integer_levels(self, numeric_values: np.ndarray) -> bool:
        valid = numeric_values[~np.isnan(numeric_values)]
        if valid.size == 0:
            return False
        if not np.all(np.isclose(valid, np.round(valid))):
            return False
        n_unique = int(pd.Series(valid).nunique())
        return 2 <= n_unique <= int(self.integer_as_level_threshold)

    def _group_integer_levels(self, numeric_values, target, target_sq, target_int, n_classes, baseline_score):
        valid_mask = ~np.isnan(numeric_values)
        if not np.any(valid_mask):
            return None, baseline_score, 1, "all_nan", None

        valid_vals = np.round(numeric_values[valid_mask]).astype(np.int64)
        unique_vals = np.unique(valid_vals)
        unique_vals.sort()
        n_keep = len(unique_vals)
        if n_keep == 0:
            return None, baseline_score, 1, "no_integer_levels", None

        other_code = n_keep
        missing_code = n_keep + 1
        value_map = {int(v): i for i, v in enumerate(unique_vals)}

        codes = np.full(len(numeric_values), other_code, dtype=int)
        valid_idx = np.where(valid_mask)[0]
        mapped = pd.Series(valid_vals).map(value_map).to_numpy()
        codes[valid_idx] = mapped
        codes[~valid_mask] = missing_code
        r = missing_code + 1

        rule = {
            'type': 'integer_levels',
            'value_map': value_map,
            'other_code': other_code,
            'missing_code': missing_code
        }
        score = self._calc_score_wrapper(codes, target, target_sq, target_int, n_classes, r)
        return downcast_codes_safe(codes), score, r, f"integer_levels({r})", rule

    def _group_categorical(
        self,
        raw_series,
        target,
        target_sq,
        target_int,
        n_classes,
        sample_indices,
        baseline_score,
        ordered: bool = False,
        category_order: Optional[List[str]] = None,
    ):
        n_total = len(target)
        is_high_card, is_id_like = check_cardinality_and_id(raw_series, sample_indices, self.max_estimated_uniques)
        if is_id_like: return None, baseline_score, 1, "id_col", None
        if is_high_card: return None, baseline_score, 1, "high_cardinality", None

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
            return None, baseline_score, 1, "no_valid_category", None

        if ordered:
            if category_order:
                rank_map = {str(v): i for i, v in enumerate(category_order)}
                ranked = [(idx, rank_map.get(str(uniques[idx]))) for idx in final_indices]
                ranked = [(idx, r) for idx, r in ranked if r is not None]
                if ranked:
                    ranked.sort(key=lambda x: x[1])
                    min_rank = ranked[0][1]
                    max_rank = ranked[-1][1]
                    contiguous_indices = []
                    for idx, val in enumerate(uniques):
                        rank = rank_map.get(str(val))
                        if rank is not None and min_rank <= rank <= max_rank:
                            contiguous_indices.append(idx)
                    final_indices = np.array(contiguous_indices, dtype=np.int64)
                else:
                    final_indices = np.sort(final_indices)
            elif isinstance(raw_series.dtype, pd.CategoricalDtype) and raw_series.dtype.ordered:
                order_map = {str(v): i for i, v in enumerate(raw_series.dtype.categories.tolist())}
                ranked = [(idx, order_map.get(str(uniques[idx]))) for idx in final_indices]
                ranked = [(idx, r) for idx, r in ranked if r is not None]
                if ranked:
                    ranked.sort(key=lambda x: x[1])
                    min_rank = ranked[0][1]
                    max_rank = ranked[-1][1]
                    contiguous_indices = []
                    for idx, val in enumerate(uniques):
                        rank = order_map.get(str(val))
                        if rank is not None and min_rank <= rank <= max_rank:
                            contiguous_indices.append(idx)
                    final_indices = np.array(contiguous_indices, dtype=np.int64)
                else:
                    final_indices = np.sort(final_indices)
            else:
                # Keep first-seen order if no explicit order metadata exists.
                final_indices = np.sort(final_indices)
        else:
            label_keys = np.array([str(uniques[idx]) for idx in final_indices], dtype=object)
            freq_keys = -valid_counts[final_indices]
            final_indices = final_indices[np.lexsort((label_keys, freq_keys))]
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
        rule = {
            'type': 'category',
            'value_map': value_map,
            'other_code': other_code,
            'missing_code': missing_code,
            'ordered': bool(ordered),
        }
        
        score = self._calc_score_wrapper(codes, target, target_sq, target_int, n_classes, r)
        return downcast_codes_safe(codes), score, r, f"category({r})", rule

    def _calc_score_wrapper(self, codes, target, target_sq, target_int, n_classes, r):
        if self.task_type == 'regression':
            if target_sq is None: target_sq = target ** 2
            score, _ = self.scorer.calc_score_reg_bincount_idx(target, target_sq, codes, r)
        else:
            score, _ = self.scorer.calc_score_cls_bincount_idx(target_int, n_classes, codes, r, check_memory=False)
        return score

    def _calc_score_partial(self, codes_valid, target, target_sq, target_int, n_classes, valid_mask, stats_missing, r):
        n_total = len(target) if target is not None else len(target_int)
        
        if self.task_type == 'regression':
            t_valid = target[valid_mask]
            tsq_valid = target_sq[valid_mask]
            
            return self.scorer.calc_score_regression_partial(
                codes_valid, t_valid, tsq_valid, r, stats_missing, n_total
            )
        else: 
            t_valid = target_int[valid_mask]
            return self.scorer.calc_score_classification_partial(
                codes_valid, t_valid, n_classes, r, stats_missing, n_total
            )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.transform_rules_:
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
            
            if codes is not None: transformed_df[feature] = downcast_codes_safe(codes)
            
        return transformed_df

    def _format_sort_key(self, rank: int, total_bins: int) -> str:
        width = max(2, len(str(max(total_bins, 1))))
        return f"{int(rank):0{width}d}"

    def _should_prefix_labels(self, rule_type: str) -> bool:
        style = str(self.label_prefix_style or "numeric_only")
        if style == "none":
            return False
        if style == "all_bins":
            return True
        if style == "numeric_only":
            return rule_type in {"qcut", "cut", "tree", "integer_levels"}
        return False

    def _build_sort_rank_map(
        self,
        code_order: List[int],
        rule: Dict[str, Any],
        code_min_map: Optional[Dict[int, float]] = None,
    ) -> Dict[int, int]:
        missing_code = rule.get("missing_code")
        other_code = rule.get("other_code")
        rule_type = rule.get("type")
        normal_codes = [c for c in code_order if c not in {missing_code, other_code}]

        if rule_type in {"qcut", "cut", "tree"}:
            code_min_map = code_min_map or {}

            def _numeric_key(c: int):
                lo = code_min_map.get(c, np.inf)
                if lo is None or (isinstance(lo, float) and np.isnan(lo)):
                    lo = np.inf
                return (float(lo), int(c))

            normal_codes = sorted(normal_codes, key=_numeric_key)
        elif rule_type == "integer_levels":
            inv_map = {int(v): int(k) for k, v in (rule.get("value_map", {}) or {}).items()}
            normal_codes = sorted(normal_codes, key=lambda c: (inv_map.get(int(c), np.inf), int(c)))
        else:
            # Categorical codes already encode intended order.
            normal_codes = sorted(normal_codes)

        rank_map: Dict[int, int] = {}
        rank = 1
        for c in normal_codes:
            rank_map[int(c)] = rank
            rank += 1
        if other_code in code_order:
            rank_map[int(other_code)] = rank
            rank += 1
        if missing_code in code_order:
            rank_map[int(missing_code)] = rank

        return rank_map

    def _build_bin_label_and_key(
        self,
        code,
        rule,
        min_val=None,
        max_val=None,
        total_bins: int = 1,
        rank_map: Optional[Dict[int, int]] = None,
        total_slots: Optional[int] = None,
    ):
        if rank_map is not None and int(code) in rank_map:
            rank = int(rank_map[int(code)])
            slot_count = int(total_slots if total_slots is not None else max(rank_map.values()))
            sort_key = self._format_sort_key(rank, slot_count)
        else:
            if code == rule.get('missing_code'):
                sort_key = self._format_sort_key(total_bins + 2, total_bins + 2)
            elif code == rule.get('other_code'):
                sort_key = self._format_sort_key(total_bins + 1, total_bins + 2)
            else:
                sort_key = self._format_sort_key(int(code) + 1, total_bins + 2)

        base_label = self._get_bin_label(code, rule, min_val=min_val, max_val=max_val)
        if self._should_prefix_labels(rule.get("type", "")):
            display_label = f"{sort_key}_{base_label}"
        else:
            display_label = base_label
        return sort_key, display_label

    def get_axis_metadata(self, raw_series: pd.Series, codes: np.ndarray, rule: Dict[str, Any]):
        code_order = sorted(int(c) for c in np.unique(codes))
        total_bins = int(rule.get('missing_code', 0)) + 1
        series_numeric = pd.to_numeric(raw_series, errors="coerce")
        is_numeric_interval_rule = rule.get("type") in {"qcut", "cut", "tree"}
        code_min_map: Dict[int, float] = {}
        if is_numeric_interval_rule:
            for code in code_order:
                mask = codes == code
                if mask.any():
                    vals = series_numeric[mask]
                    if not vals.empty:
                        code_min_map[int(code)] = float(vals.min())
        rank_map = self._build_sort_rank_map(code_order, rule, code_min_map=code_min_map)
        total_slots = max(rank_map.values()) if rank_map else total_bins + 2

        labels = []
        sort_keys = []
        for code in code_order:
            min_val = None
            max_val = None
            if is_numeric_interval_rule:
                mask = codes == code
                if mask.any():
                    vals = series_numeric[mask]
                    if not vals.empty:
                        min_val = vals.min()
                        max_val = vals.max()
            sort_key, display_label = self._build_bin_label_and_key(
                code,
                rule,
                min_val=min_val,
                max_val=max_val,
                total_bins=total_bins,
                rank_map=rank_map,
                total_slots=total_slots,
            )
            sort_keys.append(sort_key)
            labels.append(display_label)
        return code_order, labels, sort_keys

    def get_feature_details(self, df_origin: pd.DataFrame, target_vals: np.ndarray) -> pd.DataFrame:
        details_list = []
        
        # We need processed_codes for this.
        # If we didn't save them (e.g. loaded from pickle without processed_codes_), we might need to re-transform.
        # But for the standard flow, we have them.
        
        # If processed_codes_ is empty but we have rules, maybe re-transform? 
        # For now assume we are in same session or handled externally. 
        # Actually core.py relied on processed_codes being available after fit.
        
        # If this method is called after loading a model where processed_codes_ are lost,
        # we would need to run transform on df_origin to get codes.
        # Let's verify if we have codes.
        
        for feature in self.transform_rules_:
            rule = self.transform_rules_[feature]
            
            # Check if we have cached codes, else calculate
            if feature in self.processed_codes_:
                 codes = self.processed_codes_[feature]
                 # Validate code length matches data length? 
                 # get_feature_details usually called with the training data in core.py immediately after fit.
                 # If df_origin is different length than training data, this will crash if we use cached codes.
                 if len(codes) != len(df_origin):
                     # Recalculate codes
                     codes = self._transform_single_feature(df_origin[feature], rule)
            else:
                 codes = self._transform_single_feature(df_origin[feature], rule)
            
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
            total_bins = int(rule.get('missing_code', 0)) + 1
            group_stats: Dict[int, Dict[str, Any]] = {}
            for code, grp in groups:
                cnt = len(grp)
                t_mean = grp['target'].mean()
                f_min = None
                f_max = None
                if is_numeric_rule:
                    f_min = grp['feat'].min()
                    f_max = grp['feat'].max()
                group_stats[int(code)] = {
                    "count": int(cnt),
                    "target_mean": float(t_mean),
                    "min_val": f_min,
                    "max_val": f_max,
                }

            code_order = sorted(group_stats.keys())
            code_min_map: Dict[int, float] = {}
            if is_numeric_rule:
                for c in code_order:
                    lo = group_stats[c]["min_val"]
                    if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
                        code_min_map[c] = float(lo)
            rank_map = self._build_sort_rank_map(code_order, rule, code_min_map=code_min_map)
            total_slots = max(rank_map.values()) if rank_map else total_bins + 2

            for code in code_order:
                stats = group_stats[code]
                sort_key, display_label = self._build_bin_label_and_key(
                    code,
                    rule,
                    stats["min_val"],
                    stats["max_val"],
                    total_bins=total_bins,
                    rank_map=rank_map,
                    total_slots=total_slots,
                )
                details_list.append({
                    'Feature': feature,
                    'Bin_Idx': int(code),
                    'Bin_Label': display_label,
                    'Bin_Sort_Key': sort_key,
                    'Bin_Display_Label': display_label,
                    'Count': stats["count"],
                    'Target_Mean': stats["target_mean"]
                })
        
        if details_list:
            df_details = pd.DataFrame(details_list)
            return df_details.sort_values(['Feature', 'Bin_Idx']).reset_index(drop=True)
        else:
            return pd.DataFrame(
                columns=[
                    'Feature',
                    'Bin_Idx',
                    'Bin_Label',
                    'Bin_Sort_Key',
                    'Bin_Display_Label',
                    'Count',
                    'Target_Mean',
                ]
            )

    def _transform_single_feature(self, raw_series, rule):
        # Helper to transform single feature (logic duplicated from transform but simplified)
        rule_type = rule['type']; missing_code = rule['missing_code']
        codes = None
        
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
        
        elif rule_type == 'integer_levels':
            vals = pd.to_numeric(raw_series, errors='coerce').to_numpy(dtype=float)
            nan_mask = np.isnan(vals)
            rounded = np.round(vals[~nan_mask]).astype(np.int64)
            codes = np.full(len(vals), rule.get('other_code', missing_code), dtype=int)
            if rounded.size:
                mapped = pd.Series(rounded).map(rule['value_map'])
                found = mapped.notna().to_numpy()
                valid_idx = np.where(~nan_mask)[0]
                if found.any():
                    codes[valid_idx[found]] = mapped[found].astype(int)
            codes[nan_mask] = missing_code

        if codes is None:
             codes = np.full(len(raw_series), missing_code, dtype=int)

        return downcast_codes_safe(codes)

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
        elif r_type == 'integer_levels':
            if code == rule.get('other_code'):
                return "Other"
            val_map = rule.get('value_map', {})
            values = [k for k, v in val_map.items() if v == code]
            if values:
                return str(values[0])
            return f"Level_{code}"
            
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
