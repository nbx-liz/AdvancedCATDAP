import pandas as pd
import io
from typing import Dict, Any

class ResultExporter:
    """Service to export analysis results to various formats."""

    @staticmethod
    def generate_excel_report(result: Dict[str, Any], meta: Dict[str, Any] = None) -> io.BytesIO:
        """
        Generate a multi-sheet Excel report from analysis results.
        
        Args:
            result: The analysis result dictionary from JobManager.
            meta: Optional metadata about the dataset.
            
        Returns:
            io.BytesIO: The generated Excel file as a byte stream.
        """
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. Summary Sheet
            summary_data = []
            
            # Basic Results
            summary_data.append(["Analysis Summary", ""])
            summary_data.append(["Task Type", result.get('mode', 'N/A')])
            summary_data.append(["Baseline AIC", result.get('baseline_score', 0)])
            
            fi_data = result.get('feature_importances', [])
            if fi_data:
                df_fi_temp = pd.DataFrame(fi_data)
                col_map = {c.lower(): c for c in df_fi_temp.columns}
                score_col = col_map.get('score', 'Score')
                if score_col in df_fi_temp.columns:
                    summary_data.append(["Optimized AIC", df_fi_temp[score_col].min()])
            
            summary_data.append(["Selected Features count", len(fi_data)])
            
            # Metadata
            if meta:
                summary_data.append(["", ""])
                summary_data.append(["Dataset Metadata", ""])
                summary_data.append(["Dataset ID", meta.get('dataset_id', 'N/A')])
                summary_data.append(["Total Columns", meta.get('n_columns', 0)])
                summary_data.append(["Total Rows", meta.get('n_rows', 0)])
            
            df_summary = pd.DataFrame(summary_data, columns=["Item", "Value"])
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            
            # 2. Feature Importances
            if fi_data:
                df_fi = pd.DataFrame(fi_data)
                df_fi.to_excel(writer, sheet_name="Feature Importances", index=False)
            
            # 3. Interaction Importances
            ii_data = result.get('interaction_importances', [])
            if ii_data:
                df_ii = pd.DataFrame(ii_data)
                df_ii.to_excel(writer, sheet_name="Interactions", index=False)
            
            # 4. Feature Details (Top 10)
            feature_details = result.get('feature_details', {})
            if feature_details and fi_data:
                df_fi_temp = pd.DataFrame(fi_data)
                col_map = {c.lower(): c for c in df_fi_temp.columns}
                feat_col = col_map.get('feature', 'Feature')
                delta_col = col_map.get('delta_score', 'Delta_Score')
                
                if feat_col in df_fi_temp.columns and delta_col in df_fi_temp.columns:
                    top_features = df_fi_temp.nlargest(10, delta_col)[feat_col].tolist()
                    
                    for feat in top_features:
                        detail = feature_details.get(feat)
                        if detail:
                            # Create a DataFrame for this feature's bins
                            bin_data = {
                                'Bin': detail.get('bin_labels', [f"Bin {i}" for i in range(len(detail.get('bin_counts', [])))]),
                                'Count': detail.get('bin_counts', []),
                                'Target Mean': detail.get('bin_means', [])
                            }
                            # Handle case where bin_labels might be missing but edges exist
                            if not detail.get('bin_labels') and detail.get('bin_edges'):
                                edges = detail['bin_edges']
                                bin_data['Bin'] = [f"[{edges[i]:.4f}, {edges[i+1]:.4f})" for i in range(len(edges)-1)]
                            
                            df_detail = pd.DataFrame(bin_data)
                            # Sheet names must be <= 31 chars
                            sheet_title = f"Det_{feat}"[:31]
                            df_detail.to_excel(writer, sheet_name=sheet_title, index=False)
                            
        output.seek(0)
        return output
