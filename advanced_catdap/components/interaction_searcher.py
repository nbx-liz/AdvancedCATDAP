
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional, Tuple, Any

from ..config import *
from .scoring import Scorer
from .utils import downcast_codes_safe

class InteractionSearcher:
    """
    Searches for pairwise interactions between selected features that improve the model score.
    """
    def __init__(self, 
                 task_type: str,
                 scorer: Scorer,
                 max_pairwise_bins: int = DEFAULT_MAX_PAIRWISE_BINS,
                 n_interaction_candidates: int = DEFAULT_N_INTERACTION_CANDIDATES,
                 extra_penalty: float = DEFAULT_EXTRA_PENALTY,
                 verbose: bool = DEFAULT_VERBOSE):
        
        self.task_type = task_type
        self.scorer = scorer
        self.max_pairwise_bins = max_pairwise_bins
        self.n_interaction_candidates = n_interaction_candidates
        self.extra_penalty = extra_penalty
        self.verbose = verbose
        
        self.interaction_importances_ = None

    def search(self, selected_features: List[str], 
               processed_codes: Dict[str, np.ndarray], 
               processed_r: Dict[str, int], 
               feature_scores: Dict[str, float],
               sample_indices: np.ndarray,
               target_values: np.ndarray, 
               target_sq: Optional[np.ndarray], 
               target_int: Optional[np.ndarray], 
               n_classes: int, 
               n_total: int) -> pd.DataFrame:
        
        m = len(selected_features)
        if self.verbose: print(f"Combination Search: Top {m} features...")

        combo_candidates = []
        t_samp_reg = target_values[sample_indices] if self.task_type == 'regression' else None
        tsq_samp_reg = target_sq[sample_indices] if self.task_type == 'regression' else None
        t_samp_cls = target_int[sample_indices] if self.task_type == 'classification' else None
        
        sample_codes = {f: processed_codes[f][sample_indices] for f in selected_features}
        
        sample_scores = {}
        # Calculate sample scores for single features to estimate gain
        for f in selected_features:
            c = sample_codes[f]; r = int(processed_r[f])
            if self.task_type == 'regression':
                s, _ = self.scorer.calc_score_reg_bincount_idx(t_samp_reg, tsq_samp_reg, c, r)
            else:
                s, _ = self.scorer.calc_score_cls_bincount_idx(t_samp_cls, n_classes, c, r, check_memory=False)
            sample_scores[f] = s

        # Pairwise search on sample
        for f1, f2 in combinations(selected_features, 2):
            r1, r2 = int(processed_r[f1]), int(processed_r[f2])
            n_groups = r1 * r2
            if n_groups > self.max_pairwise_bins: continue
            
            c1, c2 = sample_codes[f1], sample_codes[f2]
            comb = c1.astype(np.int64) * r2 + c2.astype(np.int64)
            comb = downcast_codes_safe(comb)
            
            if self.task_type == 'regression':
                s, _ = self.scorer.calc_score_reg_bincount_idx(t_samp_reg, tsq_samp_reg, comb, n_groups)
            else:
                s, _ = self.scorer.calc_score_cls_bincount_idx(t_samp_cls, n_classes, comb, n_groups, check_memory=False)
            
            gain = min(sample_scores[f1], sample_scores[f2]) - s
            combo_candidates.append({'f1': f1, 'f2': f2, 'gain_approx': gain, 'r1': r1, 'r2': r2, 'n_groups': n_groups})

        combo_candidates.sort(key=lambda x: x['gain_approx'], reverse=True)
        top_pairs = combo_candidates[:self.n_interaction_candidates] 
        
        final_results = []
        pair_gain_thresh = PAIR_GAIN_BASE_THRESH + self.extra_penalty
        
        # Verify top pairs on full dataset
        for item in top_pairs:
            f1, f2 = item['f1'], item['f2']
            n_groups = item['n_groups']
            
            if self.task_type == 'classification' and (n_groups * n_classes * 8 > self.scorer.max_classification_bytes): continue

            c1, c2 = processed_codes[f1], processed_codes[f2]
            comb = c1.astype(np.int64) * item['r2'] + c2.astype(np.int64)
            comb = downcast_codes_safe(comb)
            
            if self.task_type == 'regression':
                score, k = self.scorer.calc_score_reg_bincount_idx(target_values, target_sq, comb, n_groups)
            else:
                score, k = self.scorer.calc_score_cls_bincount_idx(target_int, n_classes, comb, n_groups, check_memory=True)
            
            if n_total <= k + 1: continue

            gain = min(feature_scores[f1], feature_scores[f2]) - score
            if gain > pair_gain_thresh:
                final_results.append({'Feature_1': f1, 'Feature_2': f2, 'Pair_Score': score, 'Gain': gain})

        if final_results:
            self.interaction_importances_ = pd.DataFrame(final_results).sort_values('Gain', ascending=False)
        else:
            self.interaction_importances_ = pd.DataFrame(columns=['Feature_1', 'Feature_2', 'Gain'])
            
        return self.interaction_importances_
