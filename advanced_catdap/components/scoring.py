
import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..config import EPSILON_RSS, EPSILON_LOG, DEFAULT_MAX_CLASSIFICATION_BYTES

class Scorer:
    """
    Helper class for calculating AIC/AICc scores for regression and classification tasks.
    """
    def __init__(self, use_aicc: bool = True, extra_penalty: float = 0.0, max_classification_bytes: int = DEFAULT_MAX_CLASSIFICATION_BYTES):
        self.use_aicc = use_aicc
        self.extra_penalty = extra_penalty
        self.max_classification_bytes = max_classification_bytes

    def calc_score(self, term_val: float, k: int, n: int, is_regression: bool) -> float:
        """
        Calculates AIC/AICc score.
        term_val: RSS for regression, LogLikelihood sum for classification.
        """
        if n <= k + 1: return float('inf')
        
        if is_regression:
            rss = max(term_val, EPSILON_RSS)
            base_score = n * np.log(rss / n) + 2 * k
        else:
            # term_val for classification in this context corresponds to the log_lik used in core.py
            # which is sum(count * log(p)).
            # AIC = 2k - 2ln(L).
            base_score = -2 * term_val + 2 * k
            
        if self.use_aicc:
            base_score += (2 * k * (k + 1)) / (n - k - 1)
            
        return base_score + (self.extra_penalty * k)

    def calc_score_reg_bincount_idx(self, target: np.ndarray, target_sq: np.ndarray, 
                                   indices: np.ndarray, minlength: int) -> Tuple[float, int]:
        """
        Calculates score for regression task using bincounts.
        """
        counts = np.bincount(indices, minlength=minlength)
        valid_mask = counts > 0
        k = np.count_nonzero(valid_mask) + 1 # +1 for variance param
        
        sum_y = np.bincount(indices, weights=target, minlength=minlength)
        sum_y2 = np.bincount(indices, weights=target_sq, minlength=minlength)
        
        term2 = np.zeros_like(sum_y)
        term2[valid_mask] = (sum_y[valid_mask] ** 2) / counts[valid_mask]
        
        rss_total = np.sum(sum_y2 - term2)
        return self.calc_score(rss_total, k, len(target), True), k

    def calc_score_cls_bincount_idx(self, target_int: np.ndarray, n_classes: int, 
                                   indices: np.ndarray, minlength: int, 
                                   check_memory: bool = True) -> Tuple[float, int]:
        """
        Calculates score for classification task using bincounts.
        """
        if check_memory:
            if minlength * n_classes * 8 > self.max_classification_bytes:
                return float('inf'), 0
        
        flat_idx = indices.astype(np.int64) * n_classes + target_int
        ct_flat = np.bincount(flat_idx, minlength=minlength * n_classes)
        row_sums = np.bincount(indices, minlength=minlength)
        
        nz_indices = np.flatnonzero(ct_flat)
        nz_counts = ct_flat[nz_indices]
        group_indices = nz_indices // n_classes
        nz_row_sums = row_sums[group_indices]
        
        # Log-likelihood = Sum(n_ij * log(n_ij / n_i))
        #                = Sum(n_ij * (log(n_ij) - log(n_i)))
        #                = Sum(n_ij * log(n_ij)) - Sum(n_ij * log(n_i))
        log_lik = np.sum(nz_counts * np.log(nz_counts)) - np.sum(nz_counts * np.log(nz_row_sums))
        
        k = np.count_nonzero(row_sums > 0) * (n_classes - 1)
        return self.calc_score(log_lik, k, len(target_int), False), k

    def calc_stats_missing(self, target, target_sq, target_int, n_classes, valid_mask, task_mode):
        missing_mask = ~valid_mask
        n_missing = np.count_nonzero(missing_mask)
        if n_missing == 0: return {'n': 0, 'rss': 0.0, 'loglik_part': 0.0}
        
        stats = {'n': n_missing}
        if task_mode == 'regression':
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
