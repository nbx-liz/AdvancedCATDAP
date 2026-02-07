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
    def apply_chart_style(fig, is_dark=True):
        """Apply modern dark styling to Plotly figures (Matched to GUI)"""
        template = "plotly_dark" if is_dark else "plotly"
        text_color = "#e0e0e0" if is_dark else "#333"
        
        fig.update_layout(
            template=template,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI", size=12, color=text_color),
            title_font=dict(size=16, color="#00f3ff" if is_dark else "#333"),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        return fig

    @staticmethod
    def generate_html_report(result: Dict[str, Any], meta, theme='dark'):
        """
        Generate a stand-alone HTML report with high interactivity (Client-side JS).
        Includes charts for ALL features and ALL interactions.
        """
        output = io.BytesIO()
        
        # Determine strict mode
        is_dark = (theme == 'dark' or theme == 'cyborg')
        
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
        
        # 2. Global Charts
        
        # Feature Importance
        fig_fi = go.Figure()
        if fi_data:
            df_fi = pd.DataFrame(fi_data)
            col_map = {c.lower(): c for c in df_fi.columns}
            feat_col = col_map.get('feature', 'Feature')
            delta_col = col_map.get('delta_score', col_map.get('deltascore', 'Delta_Score'))
            
            if feat_col in df_fi.columns:
                delta_col = col_map.get('delta_score', col_map.get('deltascore', 'Delta_Score'))
                
                if delta_col in df_fi.columns:
                    # MATCH GUI LOGIC: Top 15 by Delta AIC, then sorted
                    df_top = df_fi.nlargest(15, delta_col).sort_values(delta_col, ascending=True)
                    
                    fig_fi = px.bar(
                        df_top, x=delta_col, y=feat_col, orientation='h',
                        title="Top Features by Impact (Delta AIC)",
                        color=delta_col,
                        color_continuous_scale="Bluyl"
                    )
                    # Explicitly set texttemplate to avoid 'text_auto' issues in static HTML
                    fig_fi.update_traces(texttemplate='%{x:.4s}', textposition='outside', marker_line_width=0)
                    
                    # Apply standardized style
                    ResultExporter.apply_chart_style(fig_fi, is_dark)
                    
                    # Add specific margins for bar chart text
                    fig_fi.update_layout(margin=dict(r=50))

        # Interactions Heatmap (Global)
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
                    title="Global Interaction Network",
                    color_continuous_scale="Viridis"
                )
                ResultExporter.apply_chart_style(fig_heat, is_dark)
        
        # 3. Feature Details (All Features)
        feature_details = result.get('feature_details', {})
        feature_plots = {} # feat_name -> html_div
        feature_stats = {} # feat_name -> html_table_str
        
        all_features = sorted(list(feature_details.keys()))

        for feat in all_features:
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
            
            # Chart
            sub_fig = go.Figure()
            sub_fig.add_trace(go.Bar(x=bin_labels, y=bin_counts, name='Count', marker_color='rgba(100, 100, 100, 0.6)', yaxis='y'))
            if bin_means:
               sub_fig.add_trace(go.Scatter(x=bin_labels, y=bin_means, name='Target', yaxis='y2', mode='lines+markers', line=dict(color=NEON_MAGENTA, width=3)))
            
            sub_fig.update_layout(
                title=f"Detail: {feat}",
                yaxis=dict(title="Count"),
                yaxis2=dict(title="Target", overlaying='y', side='right', showgrid=False),
                legend=dict(orientation='h', y=1.15, x=1, xanchor='right'),
                height=400
            )
            ResultExporter.apply_chart_style(sub_fig, is_dark)
            # Add specific margins
            sub_fig.update_layout(margin=dict(l=40, r=40, t=80, b=40))

            feature_plots[feat] = pio.to_html(sub_fig, full_html=False, include_plotlyjs=False)
            
            # Table
            df_stat = pd.DataFrame({'Bin': bin_labels, 'Count': bin_counts})
            if bin_means: df_stat['Target Mean'] = bin_means
            feature_stats[feat] = df_stat.to_html(classes="table table-sm table-striped table-hover", index=False, float_format="%.4f")

        # 4. Interaction Details (All Pairs)
        interaction_charts = {} # pair_key -> html_div
        interaction_stats = {} # pair_key -> html_table_str
        interaction_details = result.get('interaction_details', {})
        all_interactions = sorted(list(interaction_details.keys()))
        
        for pair_key in all_interactions:
            det = interaction_details.get(pair_key)
            if det:
                fig_int = go.Figure(data=go.Heatmap(
                    z=det['means'], x=det['bin_labels_2'], y=det['bin_labels_1'],
                    colorscale='Viridis', colorbar=dict(title="Target")
                ))
                fig_int.update_layout(
                    title=f"Interaction: {det['feature_1']} vs {det['feature_2']}",
                    height=500,
                )
                ResultExporter.apply_chart_style(fig_int, is_dark)
                # Add specific margins
                fig_int.update_layout(margin=dict(l=40, r=40, t=80, b=100))
                interaction_charts[pair_key] = pio.to_html(fig_int, full_html=False, include_plotlyjs=False)
                
                # Interaction Tables (Sample Count & Target Mean)
                df_counts = pd.DataFrame(det['counts'], index=det['bin_labels_1'], columns=det['bin_labels_2'])
                df_means = pd.DataFrame(det['means'], index=det['bin_labels_1'], columns=det['bin_labels_2'])
                
                # Format means
                df_means = df_means.map(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

                # Reset index for display
                df_counts_disp = df_counts.reset_index().rename(columns={'index': det['feature_1']})
                df_means_disp = df_means.reset_index().rename(columns={'index': det['feature_1']})

                # Two tables side-by-side
                table_counts_html = df_counts_disp.to_html(classes="table table-glass table-sm table-bordered table-hover", index=False)
                table_means_html = df_means_disp.to_html(classes="table table-glass table-sm table-bordered table-hover", index=False)
                
                # Combine into grid
                combined_html = f"""
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="text-secondary mt-3">Sample Count Matrix</h6>
                        <div class="table-responsive">{table_counts_html}</div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-secondary mt-3">Target Mean Matrix</h6>
                        <div class="table-responsive">{table_means_html}</div>
                    </div>
                </div>
                """
                interaction_stats[pair_key] = combined_html

        # HTML Assembly
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AdvancedCATDAP Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <!-- Use Bootswatch CYBORG Theme for Dark Mode Consistency -->
            <link href="https://cdn.jsdelivr.net/npm/bootswatch@5.3.0/dist/cyborg/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
            <style>
/* Hybrid Modern Dark Theme - Assets */
:root {{
    --neon-cyan: #00f3ff;
    --neon-magenta: #bc13fe;
    --neon-green: #0aff0a;
    --glass-bg: rgba(20, 20, 20, 0.6);
    --glass-border: 1px solid rgba(255, 255, 255, 0.1);
    --text-primary: #e0e0e0;
    --text-secondary: #aaaaaa;
}}

body {{
    background: radial-gradient(circle at 10% 20%, rgb(20, 20, 20) 0%, rgb(0, 0, 0) 90%);
    min-height: 100vh;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: var(--text-primary);
}}

.glass-card {{
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: var(--glass-border);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    overflow: visible !important;
}}

.kpi-value {{
    font-size: 2rem;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, var(--neon-cyan), #ffffff);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: var(--neon-cyan);
}}

.kpi-label {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.table-glass {{
    --bs-table-bg: rgba(30, 30, 40, 0.4);
    --bs-table-color: #e0e0e0;
    --bs-table-border-color: rgba(255, 255, 255, 0.1);
    color: #e0e0e0;
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    margin-bottom: 1rem;
}}

.table-glass th {{
    background-color: rgba(0, 243, 255, 0.1) !important;
    color: var(--neon-cyan) !important;
    font-weight: 600;
    border-bottom: 1px solid var(--neon-cyan);
}}

.table-glass td {{
    vertical-align: middle;
    color: #e0e0e0;
}}

.table-glass.table-hover tbody tr:hover td {{
    background-color: rgba(255, 255, 255, 0.1);
    color: white;
}}

/* Visibility Utilities */
h1, h2, h3, h4, h5, h6 {{ color: #e0e0e0; }}
.text-info {{ color: var(--neon-cyan) !important; }}
.text-muted {{ color: var(--text-secondary) !important; }}

select.form-select {{
    background-color: #222;
    color: #e0e0e0;
    border: 1px solid #444;
}}

/* Helper for Dropdown Logic */
.hidden {{
    display: none !important;
}}
            </style>
        </head>
        <body class="{'dark-mode' if is_dark else ''}">
            <div class="header-section">
                <div class="container">
                    <div class="d-flex align-items-center justify-content-between">
                         <div>
                            <h1><i class="bi bi-cpu-fill me-2"></i>AdvancedCATDAP Report</h1>
                            <p class="mb-0 opacity-75">Interactive Analysis Results</p>
                         </div>
                         <div class="text-end">
                            <p class="mb-0">{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                            <small>{meta.get('filename') if meta else 'Dataset'}</small>
                         </div>
                    </div>
                </div>
            </div>
            
            <div class="container">
                <!-- Summary KPI -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="glass-card text-center">
                            <h6 class="text-muted text-uppercase mb-2">Baseline AIC</h6>
                            <h2 class="mb-0">{baseline:,.0f}</h2>
                        </div>
                    </div>
                     <div class="col-md-3">
                        <div class="glass-card text-center">
                            <h6 class="text-muted text-uppercase mb-2">Optimized AIC</h6>
                            <h2 class="mb-0" style="color: #0aff0a">{best_aic:,.0f}</h2>
                            <small class="text-success">Improv: {(baseline - best_aic):,.0f}</small>
                        </div>
                    </div>
                     <div class="col-md-3">
                        <div class="glass-card text-center">
                            <h6 class="text-muted text-uppercase mb-2">Selected Features</h6>
                            <h2 class="mb-0">{len(fi_data)}</h2>
                        </div>
                    </div>
                     <div class="col-md-3">
                        <div class="glass-card text-center">
                            <h6 class="text-muted text-uppercase mb-2">Model Type</h6>
                             <h2 class="mb-0 text-uppercase">{result.get('mode', 'N/A')}</h2>
                        </div>
                    </div>
                </div>

                <!-- Global Charts -->
                <div class="row mb-5">
                    <div class="col-md-6">
                        <div class="glass-card">
                             <!-- Feature Importance -->
                             {pio.to_html(fig_fi, full_html=False, include_plotlyjs=False)}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card">
                             <!-- Interaction Network -->
                             {pio.to_html(fig_heat, full_html=False, include_plotlyjs=False)}
                        </div>
                    </div>
                </div>
        """
        
        # Feature Selection Dropdown
        feature_options = "".join([f'<option value="{f}">{f}</option>' for f in all_features])
        
        html_content += f"""
                <h3 class="mb-4 text-info"><i class="bi bi-bar-chart-fill me-2"></i>Feature Details</h3>
                
                <div class="glass-card mb-4">
                    <div class="row align-items-center">
                        <div class="col-md-4">
                            <label class="form-label text-muted mb-0">Select Feature to Inspect:</label>
                        </div>
                        <div class="col-md-8">
                            <select class="form-select" onchange="showFeature(this.value)" style="background-color: var(--select-bg); color: var(--select-color); border-color: var(--table-border-color);">
                                {feature_options}
                            </select>
                        </div>
                    </div>
                </div>
                
                <div id="feature-details-container" class="mb-5">
        """
        
        # Loop features - Generate Hidden Divs
        for i, feat in enumerate(all_features):
            chart_html = feature_plots.get(feat, "")
            table_html = feature_stats.get(feat, "")
            # Show first feature by default, hide others
            is_hidden = 'hidden' if i > 0 else '' 
            
            html_content += f"""
                    <div id="chart_{feat}" class="feature-chart {is_hidden}">
                        <div class="glass-card mb-4" style="border: var(--card-border); background: var(--card-bg);">
                            <h4 class="text-center mb-0 p-2" style="color: var(--text-color); border-bottom: 1px solid var(--table-border-color);">{feat}</h4>
                            <div class="p-3">
                                {chart_html}
                            </div>
                        </div>
                    </div>
                    
                    <div id="table_{feat}" class="feature-table {is_hidden}">
                        <div class="glass-card mb-4" style="border: var(--card-border); background: var(--card-bg);">
                            <div class="p-3">
                                <h6 class="text-muted mb-3">Binning Statistics</h6>
                                <div class="table-responsive">
                                    {table_html}
                                </div>
                            </div>
                        </div>
                    </div>
            """
            
        html_content += "</div>"

        # Interaction Selection Dropdown
        int_options = "".join([f'<option value="{p}">{p}</option>' for p in all_interactions])
        
        html_content += f"""
                <h3 class="mb-4 text-info"><i class="bi bi-diagram-3-fill me-2"></i>Interaction Deep Dive</h3>
                
                <div class="glass-card mb-4">
                    <div class="row align-items-center">
                         <div class="col-md-4">
                            <label class="form-label text-muted mb-0">Select Interaction Pair:</label>
                        </div>
                        <div class="col-md-8">
                            <select class="form-select" onchange="showInteraction(this.value)" style="background-color: var(--select-bg); color: var(--select-color); border-color: var(--table-border-color);">
                                {int_options}
                            </select>
                        </div>
                    </div>
                </div>
                
                <div id="interaction-details-container" class="mb-5">
        """
        
        for i, pair_key in enumerate(all_interactions):
            chart_html = interaction_charts.get(pair_key, "")
            table_html = interaction_stats.get(pair_key, "")
            is_hidden = 'hidden' if i > 0 else ''
            
            html_content += f"""
                    <div id="intchart_{pair_key}" class="int-chart {is_hidden}">
                        <div class="glass-card mb-4" style="border: var(--card-border); background: var(--card-bg);">
                            <h4 class="text-center mb-0 p-2" style="color: var(--text-color); border-bottom: 1px solid var(--table-border-color);">{pair_key}</h4>
                            <div class="p-3">
                                {chart_html}
                            </div>
                        </div>
                    </div>
                    
                    <div id="inttable_{pair_key}" class="int-table {is_hidden}">
                         <div class="glass-card mb-4" style="border: var(--card-border); background: var(--card-bg);">
                            <div class="p-3">
                                <h6 class="text-muted mb-3">Target Mean Matrix</h6>
                                <div class="table-responsive">
                                    {table_html}
                                </div>
                            </div>
                        </div>
                    </div>
            """
            
        html_content += """
                </div>
                
                <footer class="text-center text-muted mt-5 mb-3">
                    <small>Generated by AdvancedCATDAP</small>
                </footer>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                function toggleTheme() {
                    document.body.classList.toggle('dark-mode');
                    const isDark = document.body.classList.contains('dark-mode');
                    
                    const textColor = isDark ? '#e0e0e0' : '#333';
                    const update = {
                        'font.color': textColor,
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)'
                    };
                    
                    const graphs = document.querySelectorAll('.plotly-graph-div');
                    graphs.forEach(el => Plotly.relayout(el, update));
                }

                // Feature Selection Logic
                function showFeature(featName) {
                    // Hide all
                    document.querySelectorAll('.feature-chart').forEach(el => el.classList.add('hidden'));
                    document.querySelectorAll('.feature-table').forEach(el => el.classList.add('hidden'));
                    
                    // Show selected
                    const chart = document.getElementById('chart_' + featName);
                    const table = document.getElementById('table_' + featName);
                    if(chart) chart.classList.remove('hidden');
                    if(table) table.classList.remove('hidden');
                    
                    // Trigger Plotly resize just in case
                    if(chart) {
                        const plotlyDiv = chart.querySelector('.plotly-graph-div');
                        if(plotlyDiv) Plotly.Plots.resize(plotlyDiv);
                        
                        // Ensure Theme is applied to newly shown chart
                        const isDark = document.body.classList.contains('dark-mode');
                        const textColor = isDark ? '#e0e0e0' : '#333';
                        Plotly.relayout(plotlyDiv, {'font.color': textColor});
                    }
                }

                // Interaction Logic
                function showInteraction(pairKey) {
                     document.querySelectorAll('.int-chart').forEach(el => el.classList.add('hidden'));
                     document.querySelectorAll('.int-table').forEach(el => el.classList.add('hidden'));
                     
                     const chart = document.getElementById('intchart_' + pairKey);
                     const table = document.getElementById('inttable_' + pairKey);
                     
                     if(chart) chart.classList.remove('hidden');
                     if(table) table.classList.remove('hidden');
                     
                     if(chart) {
                        const plotlyDiv = chart.querySelector('.plotly-graph-div');
                        if(plotlyDiv) Plotly.Plots.resize(plotlyDiv);
                        
                        const isDark = document.body.classList.contains('dark-mode');
                        const textColor = isDark ? '#e0e0e0' : '#333';
                        Plotly.relayout(plotlyDiv, {'font.color': textColor});
                     }
                }
            </script>
        </body>
        </html>
        """
        
        output.write(html_content.encode('utf-8'))
        output.seek(0)
        return output
