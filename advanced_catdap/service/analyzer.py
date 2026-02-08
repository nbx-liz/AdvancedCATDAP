import pandas as pd
import logging
from typing import Optional, Callable, Dict, Any
import time

from advanced_catdap.core import AdvancedCATDAP
from advanced_catdap.service.schema import (
    AnalysisParams, AnalysisResult, FeatureImportance, InteractionImportance, FeatureDetail
)

class AnalyzerService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _validate_target_column(self, dataset: pd.DataFrame, params: AnalysisParams) -> None:
        if params.target_col not in dataset.columns:
            raise ValueError(f"Target column '{params.target_col}' not found in dataset.")

        target = dataset[params.target_col]
        non_null = target.dropna()
        n_non_null = int(non_null.shape[0])
        if n_non_null == 0:
            return

        if pd.api.types.is_numeric_dtype(non_null):
            return

        n_unique = int(non_null.nunique())
        unique_ratio = n_unique / n_non_null
        if unique_ratio >= 0.95 and n_unique >= 100:
            raise ValueError(
                f"Target column '{params.target_col}' appears ID-like "
                f"(unique_ratio={unique_ratio:.2f}). Please choose an outcome variable "
                "(e.g., Churn or Target_Spend)."
            )

    def run_analysis(
        self, 
        dataset: pd.DataFrame, 
        params: AnalysisParams, 
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> AnalysisResult:
        """
        Run the AdvancedCATDAP analysis and return structured results.
        """
        self.logger.info(f"Starting analysis for target: {params.target_col}")
        self._validate_target_column(dataset, params)
        
        # Initialize Core model
        model = AdvancedCATDAP(
            max_bins=params.max_bins,
            top_k=params.top_k,
            delta_threshold=params.delta_threshold,
            use_aicc=params.use_aicc,
            verbose=False # We handle logging/progress via callback
        )
        
        # Add progress callback to model (needs to be implemented in Core)
        if hasattr(model, 'set_progress_callback'):
            model.set_progress_callback(progress_cb)
        
        start_time = time.time()
        
        # Fit model
        try:
            model.fit(
                dataset, 
                target_col=params.target_col, 
                candidates=params.candidates,
                force_categoricals=params.force_categoricals,
                ordered_categoricals=params.ordered_categoricals,
                category_orders=params.category_orders,
                label_prefix_style=params.label_prefix_style,
            )
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise e

        detected_mode = str(getattr(model, 'mode', 'unknown'))

        # Construct Result
        result = AnalysisResult(
            target_col=params.target_col,
            mode=detected_mode,
            task_type=detected_mode.lower(),
            n_rows_used=len(dataset),
            sampled=params.sample_size is not None and len(dataset) <= params.sample_size,
            baseline_score=getattr(model, 'baseline_score', 0.0), # Core uses baseline_score (no underscore)
            transform_rules=getattr(model, 'transform_rules_', {}),
            feature_details={} # Populate below
        )

        # Populate Feature Importances
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            fi_list = []
            for _, row in model.feature_importances_.iterrows():
                fi_list.append(FeatureImportance(
                    Feature=row['Feature'],
                    Score=row['Score'],
                    Delta_Score=row['Delta_Score'],
                    Actual_Bins=row['Actual_Bins'],
                    Method=row['Method']
                ))
            result.feature_importances = fi_list

        # Populate Interaction Importances
        if hasattr(model, 'interaction_importances_') and model.interaction_importances_ is not None:
            ii_list = []
            for _, row in model.interaction_importances_.iterrows():
                ii_list.append(InteractionImportance(
                    Feature_1=row['Feature_1'],
                    Feature_2=row['Feature_2'],
                    Pair_Score=row['Pair_Score'],
                    Gain=row['Gain']
                ))
            result.interaction_importances = ii_list

        # Populate Feature Details (converting to dict structure)
        if hasattr(model, 'feature_details_') and model.feature_details_ is not None:
            fd_dict = {}
            # feature_details_ is a DataFrame with columns: Feature, Bin_Idx, Bin_Label, Count, Target_Mean
            details_df = model.feature_details_
            if isinstance(details_df, pd.DataFrame) and "Feature" in details_df.columns:
                for feat, group in details_df.groupby("Feature"):
                    # Sort by Bin_Idx to ensure order
                    if "Bin_Idx" in group.columns:
                        group = group.sort_values("Bin_Idx")
                    
                    fd = FeatureDetail(
                        bin_labels=group['Bin_Label'].tolist() if 'Bin_Label' in group.columns else None,
                        bin_sort_keys=group['Bin_Sort_Key'].tolist() if 'Bin_Sort_Key' in group.columns else None,
                        bin_display_labels=group['Bin_Display_Label'].tolist() if 'Bin_Display_Label' in group.columns else None,
                        bin_counts=group['Count'].tolist() if 'Count' in group.columns else None,
                        bin_means=group['Target_Mean'].tolist() if 'Target_Mean' in group.columns else None,
                    )
                    fd_dict[feat] = fd
            result.feature_details = fd_dict

        # Populate Interaction Details (Calculate from dataset if supported)
        if hasattr(model, 'interaction_importances_') and model.interaction_importances_ is not None and not model.interaction_importances_.empty:
            from advanced_catdap.service.schema import InteractionDetail
            id_dict = {}
            for _, row in model.interaction_importances_.iterrows():
                f1 = row['Feature_1']
                f2 = row['Feature_2']
                key = f"{f1}|{f2}"
                
                # Check if features exist in dataset
                if f1 in dataset.columns and f2 in dataset.columns:
                    # Create a temporary dataframe with target
                    tmp = dataset[[f1, f2, params.target_col]].copy()

                    discretizer = getattr(model, "discretizer", None)
                    rules = getattr(model, "transform_rules_", {}) or {}
                    rule_1 = rules.get(f1)
                    rule_2 = rules.get(f2)
                    if discretizer is None or rule_1 is None or rule_2 is None:
                        continue

                    codes_1 = discretizer._transform_single_feature(tmp[f1], rule_1)
                    codes_2 = discretizer._transform_single_feature(tmp[f2], rule_2)
                    code_order_1, labels_1, sort_keys_1 = discretizer.get_axis_metadata(tmp[f1], codes_1, rule_1)
                    code_order_2, labels_2, sort_keys_2 = discretizer.get_axis_metadata(tmp[f2], codes_2, rule_2)

                    tmp["_bin_code_1"] = codes_1
                    tmp["_bin_code_2"] = codes_2
                    target_series = tmp[params.target_col]
                    target_is_string_like = (
                        detected_mode.lower() == "classification"
                        and not pd.api.types.is_numeric_dtype(target_series)
                    )

                    if target_is_string_like:
                        grouped = tmp.groupby(["_bin_code_1", "_bin_code_2"])[params.target_col]
                        pivot_count = grouped.count().unstack(fill_value=0)
                        pivot_mean = grouped.apply(
                            lambda s: float(s.value_counts(normalize=True, dropna=False).iloc[0]) if len(s) else 0.0
                        ).unstack(fill_value=0.0)
                        pivot_labels = grouped.apply(
                            lambda s: str(s.value_counts(dropna=False).idxmax()) if len(s) else ""
                        ).unstack(fill_value="")
                        metric_name = "Class Purity"
                    else:
                        # Numeric target: retain existing target mean semantics.
                        pivot_mean = pd.pivot_table(
                            tmp, values=params.target_col, index="_bin_code_1", columns="_bin_code_2", aggfunc='mean'
                        )
                        pivot_count = pd.pivot_table(
                            tmp, values=params.target_col, index="_bin_code_1", columns="_bin_code_2", aggfunc='count'
                        )
                        pivot_labels = None
                        metric_name = "Target Mean"
                    
                    pivot_mean = pivot_mean.reindex(index=code_order_1, columns=code_order_2)
                    pivot_count = pivot_count.reindex(index=code_order_1, columns=code_order_2)

                    # Fill NaMs
                    pivot_mean = pivot_mean.fillna(0)
                    pivot_count = pivot_count.fillna(0).astype(int)
                    if pivot_labels is not None:
                        pivot_labels = pivot_labels.reindex(index=code_order_1, columns=code_order_2, fill_value="")
                    
                    # Convert to Lists
                    bin_labels_1 = labels_1
                    bin_labels_2 = labels_2
                    means_matrix = pivot_mean.values.tolist()
                    counts_matrix = pivot_count.values.tolist()
                    dominant_labels = pivot_labels.values.tolist() if pivot_labels is not None else None
                    
                    id_obj = InteractionDetail(
                        feature_1=f1,
                        feature_2=f2,
                        bin_labels_1=bin_labels_1,
                        bin_labels_2=bin_labels_2,
                        bin_sort_keys_1=sort_keys_1,
                        bin_sort_keys_2=sort_keys_2,
                        bin_display_labels_1=labels_1,
                        bin_display_labels_2=labels_2,
                        counts=counts_matrix,
                        means=means_matrix,
                        metric_name=metric_name,
                        dominant_labels=dominant_labels,
                    )
                    id_dict[key] = id_obj
            
            result.interaction_details = id_dict

        self.logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
        return result
