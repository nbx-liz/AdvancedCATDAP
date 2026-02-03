import pandas as pd
import numpy as np
from itertools import combinations
from pandas.api.types import CategoricalDtype
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class PracticalAICExplorer:
    def __init__(self, use_aicc=True, min_cat_fraction=0.001, max_categories=100,
                 task_type='auto', min_samples_leaf_rate=0.05, max_leaf_samples=100, 
                 numeric_threshold=0.5, max_pairwise_bins=50000, 
                 # ペア探索の候補数 (Sample -> Full に持ち込む数)
                 n_interaction_candidates=100,
                 # メモリガード: バイト数指定 (200MB)
                 max_classification_bytes=200 * 1024 * 1024,
                 max_estimated_uniques=20000, 
                 extra_penalty=0.0, save_rules_mode='top_k', verbose=True):
        """
        実務用AIC構造探索クラス (Final Optimized Edition)
        """
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
        
        self._reset_state()

    def _reset_state(self):
        self.mode = None
        self.baseline_score = None
        self.results = None
        self.combo_results = None
        self.transform_rules = {}

    def fit(self, df, target_col, candidates, max_bins=5, top_k=20, delta_threshold=0.0, force_categoricals=None):
        self._reset_state()

        # --- 1. データ準備 ---
        cols_needed = [target_col] + [c for c in candidates if c in df.columns]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        # メモリ効率のため .copy() は極力避けるが、dropnaはコピーを返す
        df_clean = df[cols_needed].dropna(subset=[target_col])
        n_total = len(df_clean)
        if n_total == 0: raise ValueError("Data is empty.")

        target_values = df_clean[target_col].to_numpy()
        if np.issubdtype(target_values.dtype, np.number):
             target_values = target_values.astype(np.float64)

        # 共有サンプリング
        sample_size = min(10000, n_total)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_total, size=sample_size, replace=False)
        
        # boolean mask
        sample_mask_full = np.zeros(n_total, dtype=bool)
        sample_mask_full[sample_indices] = True
        
        force_cats_set = set(force_categoricals) if force_categoricals else set()

        # --- 2. タスク判定と事前計算 ---
        self.mode = self._detect_task_type(df_clean[target_col])
        if self.verbose:
            print(f"--- Mode: {self.mode.upper()} | Target: '{target_col}' | N: {n_total} ---")

        target_sq = None
        target_int = None
        n_classes = 0

        if self.mode == 'regression':
            if not np.issubdtype(target_values.dtype, np.number):
                raise ValueError("Target must be numeric for regression.")
            target_sq = target_values ** 2
            
            y_mean = np.mean(target_values)
            rss_null = np.sum(target_sq) - (np.sum(target_values)**2 / n_total)
            k_null = 2
            self.baseline_score = self._calc_score(rss_null, k_null, n_total, True)

        else: # classification
            target_int, unique_classes = pd.factorize(target_values, sort=True)
            target_int = self._downcast_codes_safe(target_int)
            n_classes = len(unique_classes)
            
            counts = np.bincount(target_int)
            log_lik_null = np.sum(counts * np.log(counts / n_total + 1e-15))
            k_null = n_classes - 1
            self.baseline_score = self._calc_score(log_lik_null, k_null, n_total, False)

        if self.verbose: print(f"Baseline Score: {self.baseline_score:.2f}")

        # --- 3. 単変量解析 ---
        uni_results = []
        processed_codes = {} 
        processed_r = {} 
        temp_rules = {}

        min_samples = int(n_total * self.min_samples_leaf_rate)
        min_samples = max(5, min(min_samples, self.max_leaf_samples))

        for feature in candidates:
            if feature not in df_clean.columns: continue
            
            is_forced_cat = feature in force_cats_set
            
            # 特徴量処理
            best_codes, score, actual_r, method, rule = self._process_feature_optimized(
                df_clean[feature], target_values, target_sq, target_int, n_classes,
                max_bins, min_samples, sample_indices, sample_mask_full,
                feature_name=feature, force_category=is_forced_cat
            )
            
            delta = self.baseline_score - score
            
            uni_results.append({
                'Feature': feature,
                'Score': score,
                'Delta_Score': delta,
                'Actual_Bins': actual_r,
                'Method': method
            })
            
            if delta > delta_threshold and best_codes is not None:
                processed_codes[feature] = best_codes
                processed_r[feature] = actual_r
                if rule is not None:
                    rule.update({'meta_score': score, 'meta_r': actual_r, 'meta_method': method})
                    temp_rules[feature] = rule

        self.results = pd.DataFrame(uni_results).sort_values('Score').reset_index(drop=True)

        # --- 4. フィルタリング ---
        valid_features_list = self.results[self.results['Delta_Score'] > delta_threshold]['Feature'].tolist()
        available_features = [f for f in valid_features_list if f in processed_codes]
        selected_features = available_features[:top_k]
        
        save_targets = selected_features if self.save_rules_mode == 'top_k' else available_features
        self.transform_rules = {k: temp_rules[k] for k in save_targets if k in temp_rules}

        processed_codes = {k: processed_codes[k] for k in selected_features}
        processed_r = {k: processed_r[k] for k in selected_features}
        
        # --- 5. ペア探索 (Fix: Gainベースの二段階選抜) ---
        m = len(selected_features)
        if self.verbose: print(f"Combination Search: Top {m} features...")

        combo_candidates = []
        
        # Phase 1: サンプルデータでのGain計算
        # Step A: 単独スコア(サンプル)の事前計算
        t_samp_reg = target_values[sample_indices] if self.mode == 'regression' else None
        tsq_samp_reg = target_sq[sample_indices] if self.mode == 'regression' else None
        t_samp_cls = target_int[sample_indices] if self.mode == 'classification' else None
        
        sample_codes_map = {
            f: processed_codes[f][sample_indices] for f in selected_features
        }
        
        sample_single_scores = {}
        for f in selected_features:
            c_samp = sample_codes_map[f]
            r = int(processed_r[f])
            if self.mode == 'regression':
                s, _ = self._calc_score_reg_bincount_idx(t_samp_reg, tsq_samp_reg, c_samp, r)
            else:
                s, _ = self._calc_score_cls_bincount_idx(t_samp_cls, n_classes, c_samp, r, check_memory=False)
            sample_single_scores[f] = s

        # Step B: ペアスコア概算 & Gain算出
        for f1, f2 in combinations(selected_features, 2):
            c1_samp = sample_codes_map[f1]
            c2_samp = sample_codes_map[f2]
            r1 = int(processed_r[f1])
            r2 = int(processed_r[f2])
            
            n_groups = r1 * r2
            if n_groups > self.max_pairwise_bins: continue
            
            combined_samp = c1_samp.astype(np.int64) * r2 + c2_samp.astype(np.int64)
            combined_samp = self._downcast_codes_safe(combined_samp)
            
            if self.mode == 'regression':
                score_pair_samp, _ = self._calc_score_reg_bincount_idx(
                    t_samp_reg, tsq_samp_reg, combined_samp, n_groups
                )
            else:
                score_pair_samp, _ = self._calc_score_cls_bincount_idx(
                    t_samp_cls, n_classes, combined_samp, n_groups, check_memory=False
                )
            
            # Gain = min(S1, S2) - PairScore (大きいほど良い)
            min_single = min(sample_single_scores[f1], sample_single_scores[f2])
            gain_samp = min_single - score_pair_samp
            
            combo_candidates.append({
                'f1': f1, 'f2': f2, 'gain_approx': gain_samp,
                'r1': r1, 'r2': r2, 'n_groups': n_groups
            })

        # Phase 2: Gain上位の確定計算
        # Gainが大きい順にソート
        combo_candidates.sort(key=lambda x: x['gain_approx'], reverse=True)
        top_pairs = combo_candidates[:self.n_interaction_candidates] 
        
        final_combo_results = []
        pair_gain_threshold = 2.0 + self.extra_penalty
        score_map = dict(zip(self.results["Feature"], self.results["Score"]))

        for item in top_pairs:
            f1, f2 = item['f1'], item['f2']
            n_groups = item['n_groups']
            
            c1 = processed_codes[f1]
            c2 = processed_codes[f2]
            
            if self.mode == 'classification':
                est_bytes = n_groups * n_classes * 8
                if est_bytes > self.max_classification_bytes: continue

            combined_idx = c1.astype(np.int64) * item['r2'] + c2.astype(np.int64)
            combined_idx = self._downcast_codes_safe(combined_idx)
            
            if self.mode == 'regression':
                score, k = self._calc_score_reg_bincount_idx(
                    target_values, target_sq, combined_idx, n_groups
                )
            else:
                score, k = self._calc_score_cls_bincount_idx(
                    target_int, n_classes, combined_idx, n_groups, check_memory=True
                )
            
            if n_total <= k + 1: continue

            min_single = min(score_map[f1], score_map[f2])
            gain = min_single - score
            
            if gain > pair_gain_threshold:
                final_combo_results.append({
                    'Feature_1': f1, 'Feature_2': f2, 'Pair_Score': score, 'Gain': gain
                })

        if final_combo_results:
            self.combo_results = pd.DataFrame(final_combo_results).sort_values('Gain', ascending=False)
        else:
            self.combo_results = pd.DataFrame(columns=['Feature_1', 'Feature_2', 'Gain'])

        return self.results, self.combo_results

    # -------------------------------------------------------------------------
    # 特徴量処理 (Optimized)
    # -------------------------------------------------------------------------
    def _process_feature_optimized(self, raw_series, target, target_sq, target_int, n_classes, 
                                   max_bins, min_samples, sample_indices, sample_mask_full,
                                   feature_name="", force_category=False):
        
        is_numeric_type = pd.api.types.is_numeric_dtype(raw_series)
        numeric_values = None
        
        if not force_category:
            if is_numeric_type:
                numeric_values = raw_series.to_numpy(dtype=float)
            else:
                # Fix Speed 2: サンプルだけで数値変換を試す
                sample_vals = raw_series.iloc[sample_indices] if hasattr(raw_series, 'iloc') else raw_series[sample_indices]
                sample_conv = pd.to_numeric(sample_vals, errors='coerce')
                
                if sample_conv.notna().mean() >= self.numeric_threshold:
                    # サンプルでOKなら全件変換 (コストはかかるが、無駄打ちは減る)
                    temp = pd.to_numeric(raw_series, errors='coerce')
                    numeric_values = temp.to_numpy(dtype=float)

        n_total = len(target)
        best_score = self.baseline_score
        best_codes = None
        best_r = 1
        best_method = "baseline"
        best_rule = None

        # --- A. 数値扱い ---
        if numeric_values is not None:
            valid_mask = ~np.isnan(numeric_values)
            if not np.any(valid_mask): 
                return None, best_score, 1, "all_nan", None
            
            vals_valid = numeric_values[valid_mask]
            
            # Screening用サンプルデータ
            valid_sample_mask = valid_mask & sample_mask_full
            
            if np.sum(valid_sample_mask) < 20:
                vals_sample = vals_valid[:1000] # Fallback
            else:
                vals_sample = numeric_values[valid_sample_mask]
            
            # Screening用ターゲット
            if self.mode == 'regression':
                t_screen = target[valid_sample_mask]
                tsq_screen = target_sq[valid_sample_mask]
                t_int_screen = None
            else:
                t_screen = None
                tsq_screen = None
                t_int_screen = target_int[valid_sample_mask]

            stats_missing_full = self._calc_stats_missing(
                target, target_sq, target_int, n_classes, valid_mask
            )

            # Phase 1: Screening
            candidates = []
            X_sample_tree = None; y_sample_tree = None
            
            # Fix Speed 1: min/max をループ外へ
            v_min, v_max = vals_valid.min(), vals_valid.max()
            can_cut = (v_min != v_max)

            methods = ['qcut', 'cut', 'tree']
            for method in methods:
                for n_bins in range(2, max_bins + 1):
                    try:
                        rule = None; r = 1; codes_sample = None
                        
                        if method == 'tree':
                            if X_sample_tree is None:
                                X_sample_tree = vals_sample.reshape(-1, 1)
                                y_sample_tree = t_screen if self.mode=='regression' else t_int_screen
                            
                            if len(X_sample_tree) < min_samples: continue

                            if self.mode == 'regression':
                                tree = DecisionTreeRegressor(max_leaf_nodes=n_bins, min_samples_leaf=min_samples, random_state=42)
                            else:
                                tree = DecisionTreeClassifier(max_leaf_nodes=n_bins, min_samples_leaf=min_samples, random_state=42)
                            tree.fit(X_sample_tree, y_sample_tree)
                            
                            leaf_ids = tree.apply(X_sample_tree)
                            unique_leaves = np.unique(leaf_ids)
                            mapped_leaves = np.searchsorted(unique_leaves, leaf_ids)
                            
                            missing_code = len(unique_leaves)
                            r = missing_code + 1
                            codes_sample = mapped_leaves
                            rule = {'type': 'tree', 'model': tree, 'leaves': unique_leaves, 'missing_code': missing_code}

                        elif method == 'qcut':
                            quantiles = np.linspace(0, 1, n_bins + 1)
                            bins = np.quantile(vals_sample, quantiles)
                            bins = np.unique(bins)
                            if len(bins) < 2: continue
                            bins[0] = -np.inf; bins[-1] = np.inf
                            
                            c_samp = np.searchsorted(bins, vals_sample, side='right') - 1
                            codes_sample = np.clip(c_samp, 0, len(bins) - 2)
                            missing_code = len(bins) - 1
                            r = missing_code + 1
                            rule = {'type': 'qcut', 'bins': bins, 'missing_code': missing_code}

                        else: # cut
                            if not can_cut: continue
                            bins = np.linspace(v_min, v_max, n_bins + 1)
                            bins[0] = -np.inf; bins[-1] = np.inf
                            
                            # cutは全体分布依存なのでサンプルでなく全体で...いや、
                            # ScreeningなのでサンプルでOK (相対評価のため)
                            c_samp = np.searchsorted(bins, vals_sample, side='right') - 1
                            codes_sample = np.clip(c_samp, 0, len(bins) - 2)
                            missing_code = len(bins) - 1
                            r = missing_code + 1
                            rule = {'type': 'cut', 'bins': bins, 'missing_code': missing_code}

                        if r < 2: continue
                        
                        # Fix Critical 1: minlength を r ではなく実際の最大値+1 にする
                        # (Screeningでは MissingCode が入らない前提のコードだが、
                        #  tree/qcutの実装上 mapped_leaves は 0..L-1 なので max()+1 でOK)
                        minlength_screen = int(codes_sample.max()) + 1
                        
                        score_sample = self._calc_score_wrapper(
                            codes_sample, t_screen, tsq_screen, t_int_screen, n_classes, minlength_screen
                        )
                        
                        candidates.append((score_sample, method, n_bins, r, rule))

                    except Exception: continue

            if not candidates:
                return None, self.baseline_score, 1, "no_candidates", None

            # Phase 2: Finalizing
            best_cand = min(candidates, key=lambda x: x[0])
            _, best_method_name, best_nbins, best_r, best_rule = best_cand
            
            # 全件適用
            codes_valid = None
            
            if best_rule['type'] == 'tree':
                leaf_ids = best_rule['model'].apply(vals_valid.reshape(-1, 1))
                # Fix Critical 2: 未知葉ガードの緩和
                # clipで強制的に有効範囲に収める (未知葉 -> 既存の近似葉扱い)
                pos = np.searchsorted(best_rule['leaves'], leaf_ids)
                codes_valid = np.clip(pos, 0, len(best_rule['leaves'])-1)
                
            elif best_rule['type'] in ['qcut', 'cut']:
                c_valid = np.searchsorted(best_rule['bins'], vals_valid, side='right') - 1
                codes_valid = np.clip(c_valid, 0, len(best_rule['bins']) - 2)
            
            final_score = self._calc_score_partial(
                codes_valid, target, target_sq, target_int, n_classes, 
                valid_mask, stats_missing_full, best_r
            )
            
            full_codes = np.full(n_total, best_rule['missing_code'], dtype=int)
            full_codes[valid_mask] = codes_valid
            
            return self._downcast_codes_safe(full_codes), final_score, best_r, f"{best_method_name}_{best_nbins}({best_r})", best_rule

        # --- B. カテゴリ ---
        else:
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
            pass_threshold_mask = valid_counts >= freq_threshold
            
            if n_cats > self.max_categories:
                k_idx = self.max_categories
                top_indices = np.argpartition(-valid_counts, kth=min(k_idx, n_cats)-1)[:k_idx]
                final_indices = top_indices[pass_threshold_mask[top_indices]]
            else:
                final_indices = np.where(pass_threshold_mask)[0]
            
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
            
            if r > 1:
                score = self._calc_score_wrapper(codes, target, target_sq, target_int, n_classes, r)
                if score < best_score:
                    best_score = score
                    best_codes = self._downcast_codes_safe(codes)
                    best_r = r
                    best_method = f"category({r})"
                    best_rule = rule

        return best_codes, best_score, best_r, best_method, best_rule

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
            est_bytes = r * n_classes * 8
            if est_bytes > self.max_classification_bytes: return float('inf')

            t_valid = target_int[valid_mask]
            flat_idx = codes_valid.astype(np.int64) * n_classes + t_valid
            ct_flat = np.bincount(flat_idx, minlength=r * n_classes)
            row_sums = np.bincount(codes_valid, minlength=r)
            
            nz_indices = np.flatnonzero(ct_flat)
            nz_counts = ct_flat[nz_indices]
            group_indices = nz_indices // n_classes
            nz_row_sums = row_sums[group_indices]
            
            safe_mask = nz_row_sums > 0
            if not np.all(safe_mask):
                nz_counts = nz_counts[safe_mask]
                nz_row_sums = nz_row_sums[safe_mask]

            term1 = np.sum(nz_counts * np.log(nz_counts))
            term2 = np.sum(nz_counts * np.log(nz_row_sums))
            loglik_valid = term1 - term2
            
            k_valid = np.count_nonzero(row_sums > 0) * (n_classes - 1)
            
            loglik_total = loglik_valid + stats_missing.get('loglik_part', 0.0)
            k_miss = (n_classes - 1) if stats_missing['n'] > 0 else 0
            k_total = k_valid + k_miss
            
            return self._calc_score(loglik_total, k_total, n_total, False)

    # --- Utility & Transform (Fix Stability) ---
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
        valid_groups = row_sums > 0
        k = np.count_nonzero(valid_groups) * (n_classes - 1)
        
        nz_indices = np.flatnonzero(ct_flat)
        nz_counts = ct_flat[nz_indices]
        group_indices = nz_indices // n_classes
        nz_row_sums = row_sums[group_indices]
        
        safe_mask = nz_row_sums > 0
        if not np.all(safe_mask):
            nz_counts = nz_counts[safe_mask]
            nz_row_sums = nz_row_sums[safe_mask]
            
        log_lik = np.sum(nz_counts * np.log(nz_counts)) - np.sum(nz_counts * np.log(nz_row_sums))
        return self._calc_score(log_lik, k, len(target_int), False), k

    def _check_cardinality_and_id(self, series, sample_indices):
        if hasattr(series, 'iloc'): sample_vals = series.iloc[sample_indices]
        else: sample_vals = series[sample_indices]
        sample_nunique = sample_vals.nunique()
        sample_len = len(sample_vals)
        n_total = len(series)
        if sample_nunique == sample_len: estimated_total = n_total
        else: estimated_total = sample_nunique * (n_total / sample_len)
        is_high_card = estimated_total > self.max_estimated_uniques
        unique_ratio = sample_nunique / sample_len if sample_len > 0 else 0
        is_id_like = (unique_ratio > 0.99) and (estimated_total > 0.99 * n_total)
        return is_high_card, is_id_like

    def _downcast_codes_safe(self, codes):
        if codes.size == 0: return codes.astype(np.int8)
        min_v = codes.min(); max_v = codes.max()
        if min_v >= 0:
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

    def transform(self, df):
        if not self.transform_rules:
            if self.verbose: print("Warning: No rules saved.")
            return pd.DataFrame(index=df.index)
        transformed_df = pd.DataFrame(index=df.index)
        for feature, rule in self.transform_rules.items():
            if feature not in df.columns: continue
            raw_series = df[feature]
            rule_type = rule['type']; missing_code = rule['missing_code']; codes = None
            if rule_type in ['qcut', 'cut', 'tree']:
                vals = pd.to_numeric(raw_series, errors='coerce').to_numpy(dtype=float)
                nan_mask = np.isnan(vals); codes = np.full(len(vals), missing_code, dtype=int); valid_mask = ~nan_mask
                if np.any(valid_mask):
                    vals_valid = vals[valid_mask]
                    if rule_type in ['qcut', 'cut']:
                        c_valid = np.searchsorted(rule['bins'], vals_valid, side='right') - 1
                        codes[valid_mask] = np.clip(c_valid, 0, len(rule['bins']) - 2)
                    elif rule_type == 'tree':
                        l_ids = rule['model'].apply(vals_valid.reshape(-1, 1))
                        pos = np.searchsorted(rule['leaves'], l_ids)
                        safe_pos = np.clip(pos, 0, len(rule['leaves'])-1)
                        # Fix Critical 2 (Transform): 未知葉は近傍葉へ (clip) + 念のため一致確認はせず、clipで有効化する(＝未知葉は最も近い有効葉へ)
                        # 以前のコードでは is_found で落としていたが、Finalizingのロジック(clip)と合わせるなら
                        # ここも clip した値をそのまま使うのが整合的。
                        final = codes[valid_mask]; final[:] = safe_pos; codes[valid_mask] = final
            elif rule_type == 'category':
                s_obj = raw_series.astype(object); codes = np.full(len(s_obj), rule['other_code'], dtype=int); isna_mask = s_obj.isna()
                codes[isna_mask] = missing_code
                if not isna_mask.all():
                    mapped = s_obj[~isna_mask].map(rule['value_map'])
                    found = mapped.notna()
                    if found.any(): codes[np.where(~isna_mask)[0][found]] = mapped[found].astype(int)
            if codes is not None: transformed_df[feature] = self._downcast_codes_safe(codes)
        return transformed_df