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
            for feat, details in model.feature_details_.items():
                # details is likely a DataFrame or dict, we need to adapt it to FeatureDetail schema
                # Assuming details has 'bin_edges', 'count', 'mean', etc.
                # If it's a DataFrame, we convert columns to lists.
                if isinstance(details, pd.DataFrame):
                    fd = FeatureDetail(
                        bin_means=details['mean'].tolist() if 'mean' in details else None,
                        bin_counts=details['count'].tolist() if 'count' in details else None,
                        # bin_edges might be in index or another column
                    )
                    fd_dict[feat] = fd
            result.feature_details = fd_dict

        self.logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
        return result
