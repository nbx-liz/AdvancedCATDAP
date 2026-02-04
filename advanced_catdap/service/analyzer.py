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
                force_categoricals=params.force_categoricals
            )
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise e

        # Construct Result
        result = AnalysisResult(
            target_col=params.target_col,
            mode=getattr(model, 'mode', 'unknown'),
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
                    
                    # Binning Strategy: 
                    # If numerical, use qcut/cut logic or existing bins? 
                    # For simplicity in this "Service Layer Improvement", we use simple quantile binning (5 bins) 
                    # or categorical codes if object.
                    
                    for col in [f1, f2]:
                        if pd.api.types.is_numeric_dtype(tmp[col]) and tmp[col].nunique() > 5:
                            try:
                                tmp[f"{col}_bin"] = pd.qcut(tmp[col], q=5, duplicates='drop').astype(str)
                            except:
                                tmp[f"{col}_bin"] = tmp[col].astype(str)
                        else:
                            tmp[f"{col}_bin"] = tmp[col].astype(str)
                            
                    # Pivot Table for Mean (Target Rate)
                    # We need rows=f1, cols=f2
                    pivot_mean = pd.pivot_table(tmp, values=params.target_col, index=f"{f1}_bin", columns=f"{f2}_bin", aggfunc='mean')
                    pivot_count = pd.pivot_table(tmp, values=params.target_col, index=f"{f1}_bin", columns=f"{f2}_bin", aggfunc='count')
                    
                    # Fill NaMs
                    pivot_mean = pivot_mean.fillna(0)
                    pivot_count = pivot_count.fillna(0).astype(int)
                    
                    # Convert to Lists
                    bin_labels_1 = pivot_mean.index.tolist()
                    bin_labels_2 = pivot_mean.columns.tolist()
                    means_matrix = pivot_mean.values.tolist()
                    counts_matrix = pivot_count.values.tolist()
                    
                    id_obj = InteractionDetail(
                        feature_1=f1,
                        feature_2=f2,
                        bin_labels_1=bin_labels_1,
                        bin_labels_2=bin_labels_2,
                        counts=counts_matrix,
                        means=means_matrix
                    )
                    id_dict[key] = id_obj
            
            result.interaction_details = id_dict

        self.logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
        return result
