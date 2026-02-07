import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Any

# Ensure template availability
pio.templates.default = "plotly_white" # Use white for report readability or dark if preferred

NEON_CYAN = "#00f3ff"
NEON_MAGENTA = "#bc13fe"
NEON_GREEN = "#0aff0a"

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

    @staticmethod
    def generate_html_report(result: Dict[str, Any], meta: Dict[str, Any] = None) -> io.BytesIO:
        """
        Generate a standalone HTML report with interactive Plotly charts.
        """
        output = io.BytesIO()
        
        # 1. Summary Data
        baseline = result.get('baseline_score', 0)
        best_aic = baseline
        fi_data = result.get('feature_importances', [])
        if fi_data:
            df_fi = pd.DataFrame(fi_data)
            col_map = {c.lower(): c for c in df_fi.columns}
            score_col = col_map.get('score', 'Score')
            if score_col in df_fi.columns:
                best_aic = df_fi[score_col].min()
        
        # 2. Charts Generation
        
        # Feature Importance
        fig_fi = go.Figure()
        if fi_data:
            df_fi = pd.DataFrame(fi_data)
            col_map = {c.lower(): c for c in df_fi.columns}
            feat_col = col_map.get('feature', 'Feature')
            delta_col = col_map.get('delta_score', 'Delta_Score')
            
            if feat_col in df_fi.columns and delta_col in df_fi.columns:
                df_top = df_fi.nlargest(20, delta_col).sort_values(delta_col, ascending=True)
                fig_fi = px.bar(
                    df_top, x=delta_col, y=feat_col, orientation='h',
                    title="Top Features by Impact (Delta AIC)",
                    color=delta_col,
                    color_continuous_scale="Bluyl"
                )
                fig_fi.update_layout(height=600)

        # Interactions
        fig_heat = go.Figure()
        ii_data = result.get('interaction_importances', [])
        if ii_data:
            df_ii = pd.DataFrame(ii_data)
            col_map = {c.lower(): c for c in df_ii.columns}
            f1 = col_map.get('feature_1', 'Feature_1')
            f2 = col_map.get('feature_2', 'Feature_2')
            gain = col_map.get('gain', 'Gain')
            
            if f1 in df_ii and f2 in df_ii and gain in df_ii:
                fig_heat = px.density_heatmap(
                    df_ii, x=f1, y=f2, z=gain, histfunc="sum",
                    title="Interaction Network",
                    color_continuous_scale="Viridis"
                )
        
        # Top 5 Feature Details
        top_feature_charts = []
        feature_details = result.get('feature_details', {})
        if feature_details and fi_data:
             df_fi = pd.DataFrame(fi_data)
             col_map = {c.lower(): c for c in df_fi.columns}
             feat_col = col_map.get('feature', 'Feature')
             delta_col = col_map.get('delta_score', 'Delta_Score')
             if feat_col in df_fi and delta_col in df_fi:
                 top_feats = df_fi.nlargest(5, delta_col)[feat_col].tolist()
                 
                 for feat in top_feats:
                     detail = feature_details.get(feat, {})
                     if not detail: continue
                     
                     bin_counts = detail.get('bin_counts', [])
                     bin_means = detail.get('bin_means', [])
                     bin_labels = detail.get('bin_labels', [])
                     if not bin_labels and detail.get('bin_edges'):
                         edges = detail['bin_edges']
                         bin_labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges)-1)]
                     elif not bin_labels:
                         bin_labels = [f"Bin {i}" for i in range(len(bin_counts))]
                     
                     sub_fig = go.Figure()
                     sub_fig.add_trace(go.Bar(x=bin_labels, y=bin_counts, name='Count', marker_color='rgba(100, 100, 100, 0.6)', yaxis='y'))
                     if bin_means:
                        sub_fig.add_trace(go.Scatter(x=bin_labels, y=bin_means, name='Target', yaxis='y2', mode='lines+markers', line=dict(color=NEON_MAGENTA, width=3)))
                     
                     sub_fig.update_layout(
                         title=f"Detail: {feat}",
                         yaxis=dict(title="Count"),
                         yaxis2=dict(title="Target", overlaying='y', side='right', showgrid=False),
                         legend=dict(orientation='h', y=1.1)
                     )
                     top_feature_charts.append(pio.to_html(sub_fig, full_html=False, include_plotlyjs=False))

        # HTML Assembly
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AdvancedCATDAP Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ background-color: #f8f9fa; color: #333; }}
                .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(45deg, #007bff, #6610f2); color: white; padding: 20px; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <div class="header text-center rounded">
                    <h1>Analysis Report</h1>
                    <p>Dataset ID: {meta.get('dataset_id', 'N/A') if meta else 'N/A'}</p>
                </div>
                
                <div class="row">
                    <div class="col-md-3">
                        <div class="card p-3">
                            <h5>Summary</h5>
                            <p><strong>Baseline AIC:</strong> {baseline:,.0f}</p>
                            <p><strong>Optimized AIC:</strong> {best_aic:,.0f}</p>
                            <p><strong>Features Selected:</strong> {len(fi_data)}</p>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card p-3">
                            {pio.to_html(fig_fi, full_html=False, include_plotlyjs=False)}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card p-3">
                            {pio.to_html(fig_heat, full_html=False, include_plotlyjs=False)}
                        </div>
                    </div>
                </div>
                
                <h3>Top 5 Key Drivers</h3>
                {"".join([f'<div class="card p-3">{chart}</div>' for chart in top_feature_charts])}
                
            </div>
        </body>
        </html>
        """
        
        output.write(html_content.encode('utf-8'))
        output.seek(0)
        return output
