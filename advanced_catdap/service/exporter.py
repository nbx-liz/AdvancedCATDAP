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


    @staticmethod
    def generate_html_report(result: Dict[str, Any], meta: Dict[str, Any] = None) -> io.BytesIO:
        """
        Generate a stand-alone HTML report with high interactivity (Client-side JS).
        Includes charts for ALL features and ALL interactions.
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
        
        # 2. Global Charts
        
        # Feature Importance
        fig_fi = go.Figure()
        if fi_data:
            df_fi = pd.DataFrame(fi_data)
            col_map = {c.lower(): c for c in df_fi.columns}
            feat_col = col_map.get('feature', 'Feature')
            delta_col = col_map.get('delta_score', 'Delta_Score')
            
            if feat_col in df_fi.columns and delta_col in df_fi.columns:
                # MATCH GUI LOGIC: Top 15 by Delta AIC, then sorted
                df_top = df_fi.nlargest(15, delta_col).sort_values(delta_col, ascending=True)
                
                fig_fi = px.bar(
                    df_top, x=delta_col, y=feat_col, orientation='h',
                    title="Top Features by Impact (Delta AIC)",
                    color=delta_col,
                    color_continuous_scale="Bluyl"
                )
                fig_fi.update_layout(
                    height=500, # Fixed height for Top 15
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

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
                fig_heat.update_layout(
                     margin=dict(l=20, r=20, t=40, b=20),
                     paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)'
                )
        
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
                height=400,
                margin=dict(l=40, r=40, t=60, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
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
                    margin=dict(l=40, r=40, t=50, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                interaction_charts[pair_key] = pio.to_html(fig_int, full_html=False, include_plotlyjs=False)
                
                # Table for Interaction (Target Mean Matrix) or similar summary
                # Let's show as a wide table to mimic the heatmap
                df_int_table = pd.DataFrame(det['means'], index=det['bin_labels_1'], columns=det['bin_labels_2'])
                interaction_stats[pair_key] = df_int_table.to_html(classes="table table-sm table-bordered table-hover", float_format="%.4f")

        # HTML Assembly
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AdvancedCATDAP Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
            <style>
                :root {{
                    /* Light Mode (Default) */
                    --bg-color: #f8f9fa;
                    --text-color: #333;
                    --card-bg: #fff;
                    --card-border: none;
                    --header-bg: linear-gradient(135deg, #0d6efd 0%, #0dcaf0 100%);
                    --table-color: #333;
                    --table-bg: #fff;
                    --table-border-color: #dee2e6;
                    --table-head-bg: #e9ecef;
                    --select-bg: #fff;
                    --select-color: #333;
                }}
                
                body.dark-mode {{
                    /* GUI-Like Dark Mode (Cyborg/Hybrid) */
                    --bg-color: #060606;
                    --text-color: #e0e0e0;
                    --card-bg: rgba(20, 20, 20, 0.6); /* Glass effect base */
                    --card-border: 1px solid rgba(255, 255, 255, 0.1);
                    --header-bg: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                    --table-color: #e0e0e0;
                    --table-bg: transparent; /* Transparent for glass effect */
                    --table-border-color: rgba(255, 255, 255, 0.1);
                    --table-head-bg: rgba(0, 243, 255, 0.1); /* Neon Cyan tint */
                    --table-head-color: #00f3ff; /* Neon Cyan */
                    --select-bg: #1a1a1a;
                    --select-color: #e0e0e0;
                }}
                
                body {{ 
                    background-color: var(--bg-color); 
                    color: var(--text-color); 
                    transition: background-color 0.3s, color 0.3s; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-image: var(--bg-image, none);
                }}
                
                body.dark-mode {{
                     background: radial-gradient(circle at 10% 20%, rgb(20, 20, 20) 0%, rgb(0, 0, 0) 90%);
                }}
                
                .card {{ 
                    background-color: var(--card-bg); 
                    color: var(--text-color); 
                    margin-bottom: 20px; 
                    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); /* Stronger shadow */
                    border: var(--card-border); 
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    border-radius: 10px;
                }}
                .header {{ background: var(--header-bg); color: white; padding: 20px; margin-bottom: 30px; border-radius: 0 0 20px 20px; position: relative; }}
                .nav-pills .nav-link.active {{ background-color: #0d6efd; }}
                .plot-container {{ width: 100%; }}
                .hidden {{ display: none; }}
                
                /* Table Styling - Glassmorphism Match */
                table {{ color: var(--table-color) !important; background-color: var(--table-bg) !important; }}
                .table {{ --bs-table-bg: var(--table-bg); --bs-table-color: var(--table-color); --bs-table-border-color: var(--table-border-color); }}
                .table thead th {{ 
                    background-color: var(--table-head-bg) !important; 
                    color: var(--table-head-color) !important; 
                    border-bottom: 1px solid var(--table-head-color) !important; 
                    font-weight: 600;
                }}
                .table-hover tbody tr:hover td {{ color: var(--text-color); background-color: rgba(255,255,255,0.05); }}
                body.dark-mode .table-hover tbody tr:hover td {{ background-color: rgba(255,255,255,0.1); }}
                
                /* Select High Contrast */
                select.form-select {{
                    background-color: var(--select-bg);
                    color: var(--select-color);
                    border-color: var(--text-color);
                    border-radius: 5px;
                }}
                
                #theme-toggle {{
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: transparent;
                    border: 1px solid white;
                    color: white;
                    border-radius: 20px;
                    padding: 5px 15px;
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <div class="header text-center">
                <h1>AdvancedCATDAP Report</h1>
                <p>Dataset: {meta.get('dataset_id', 'N/A') if meta else 'N/A'} | Target: {meta.get('target', 'N/A')}</p>
                <button id="theme-toggle" onclick="toggleTheme()">
                    <i class="bi bi-moon-stars-fill"></i> Toggle Theme
                </button>
            </div>
            
            <div class="container-fluid px-4">
                <!-- Summary Section -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card p-3 h-100">
                            <h5 class="text-primary">Summary</h5>
                            <hr>
                            <p><strong>Baseline AIC:</strong> {baseline:,.0f}</p>
                            <p><strong>Optimized AIC:</strong> {best_aic:,.0f}</p>
                            <p><strong>Feature Count:</strong> {len(fi_data)}</p>
                        </div>
                    </div>
                </div>
                
                <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="pills-dashboard-tab" data-bs-toggle="pill" data-bs-target="#pills-dashboard" type="button">Dashboard</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="pills-features-tab" data-bs-toggle="pill" data-bs-target="#pills-features" type="button">Feature Details</button>
                    </li>
                     <li class="nav-item" role="presentation">
                        <button class="nav-link" id="pills-interactions-tab" data-bs-toggle="pill" data-bs-target="#pills-interactions" type="button">Interactions</button>
                    </li>
                </ul>
                
                <div class="tab-content" id="pills-tabContent">
                    <!-- Dashboard Tab -->
                    <div class="tab-pane fade show active" id="pills-dashboard" role="tabpanel">
                        <div class="row">
                            <div class="col-lg-6">
                                <div class="card p-2">
                                    {pio.to_html(fig_fi, full_html=False, include_plotlyjs=False)}
                                </div>
                            </div>
                            <div class="col-lg-6">
                                <div class="card p-2">
                                    {pio.to_html(fig_heat, full_html=False, include_plotlyjs=False)}
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Feature Details Tab -->
                    <div class="tab-pane fade" id="pills-features" role="tabpanel">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="card p-3" style="max-height: 80vh; overflow-y: auto;">
                                    <label class="form-label fw-bold">Select Feature:</label>
                                    <select class="form-select" id="featureSelect" onchange="showFeature(this.value)">
                                        {''.join([f'<option value="{f}">{f}</option>' for f in all_features])}
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-9">
                                <div class="row">
                                    <div class="col-lg-8">
                                        <div class="card p-2" id="featureChartContainer">
                                            <!-- Feature Charts Injected Here via JS Logic (Pre-rendered hidden divs) -->
                                            {''.join([f'<div id="chart_{f}" class="feature-chart hidden">{plot}</div>' for f, plot in feature_plots.items()])}
                                        </div>
                                    </div>
                                    <div class="col-lg-4">
                                        <div class="card p-3">
                                            <h6 class="text-secondary">Statistics</h6>
                                             {''.join([f'<div id="table_{f}" class="feature-table hidden">{tbl}</div>' for f, tbl in feature_stats.items()])}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                     <!-- Interactions Tab -->
                    <div class="tab-pane fade" id="pills-interactions" role="tabpanel">
                         <div class="row">
                            <div class="col-md-3">
                                <div class="card p-3">
                                     <label class="form-label fw-bold">Select Interaction Pair:</label>
                                    <select class="form-select" id="interactionSelect" onchange="showInteraction(this.value)">
                                        {''.join([f'<option value="{k}">{k}</option>' for k in all_interactions])}
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-9">
                                <div class="row">
                                    <div class="col-lg-8">
                                        <div class="card p-2">
                                            {''.join([f'<div id="intchart_{k}" class="int-chart hidden">{plot}</div>' for k, plot in interaction_charts.items()])}
                                        </div>
                                    </div>
                                    <div class="col-lg-4">
                                         <div class="card p-3">
                                            <h6 class="text-secondary">Relation Matrix (Target Mean)</h6>
                                            <div style="overflow-x: auto;">
                                                {''.join([f'<div id="inttable_{k}" class="int-table hidden">{tbl}</div>' for k, tbl in interaction_stats.items()])}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                function toggleTheme() {{
                    document.body.classList.toggle('dark-mode');
                    const isDark = document.body.classList.contains('dark-mode');
                    
                    const textColor = isDark ? '#e0e0e0' : '#333';
                    const update = {{
                        'font.color': textColor,
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)'
                    }};
                    
                    const graphs = document.querySelectorAll('.plotly-graph-div');
                    graphs.forEach(el => Plotly.relayout(el, update));
                }}

                // Feature Selection Logic
                function showFeature(featName) {{
                    // Hide all
                    document.querySelectorAll('.feature-chart').forEach(el => el.classList.add('hidden'));
                    document.querySelectorAll('.feature-table').forEach(el => el.classList.add('hidden'));
                    
                    // Show selected
                    const chart = document.getElementById('chart_' + featName);
                    const table = document.getElementById('table_' + featName);
                    if(chart) chart.classList.remove('hidden');
                    if(table) table.classList.remove('hidden');
                    
                    // Trigger Plotly resize just in case
                    if(chart) {{
                        const plotlyDiv = chart.querySelector('.plotly-graph-div');
                        if(plotlyDiv) Plotly.Plots.resize(plotlyDiv);
                        
                        // Ensure Theme is applied to newly shown chart
                        const isDark = document.body.classList.contains('dark-mode');
                        const textColor = isDark ? '#e0e0e0' : '#333';
                        Plotly.relayout(plotlyDiv, {{'font.color': textColor}});
                    }}
                }}

                // Interaction Logic
                function showInteraction(pairKey) {{
                     document.querySelectorAll('.int-chart').forEach(el => el.classList.add('hidden'));
                     document.querySelectorAll('.int-table').forEach(el => el.classList.add('hidden'));
                     
                     const chart = document.getElementById('intchart_' + pairKey);
                     const table = document.getElementById('inttable_' + pairKey);
                     
                     if(chart) chart.classList.remove('hidden');
                     if(table) table.classList.remove('hidden');
                     
                     if(chart) {{
                        const plotlyDiv = chart.querySelector('.plotly-graph-div');
                        if(plotlyDiv) Plotly.Plots.resize(plotlyDiv);
                        
                        // Ensure Theme is applied
                        const isDark = document.body.classList.contains('dark-mode');
                        const textColor = isDark ? '#e0e0e0' : '#333';
                        Plotly.relayout(plotlyDiv, {{'font.color': textColor}});
                    }}
                }}

                // Initialize
                window.onload = function() {{
                    const firstFeat = "{all_features[0] if all_features else ''}";
                    if(firstFeat) {{
                        document.getElementById('featureSelect').value = firstFeat;
                        // Small delay to allow Plotly to render initial chart correctly
                        setTimeout(() => showFeature(firstFeat), 100);
                    }}
                    
                    const firstInt = "{all_interactions[0] if all_interactions else ''}";
                    if(firstInt) {{
                        document.getElementById('interactionSelect').value = firstInt;
                        setTimeout(() => showInteraction(firstInt), 100);
                    }}
                }};
            </script>
        </body>
        </html>
        """
        
        output.write(html_content.encode('utf-8'))
        output.seek(0)
        return output
