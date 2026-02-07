"""
AdvancedCATDAP Dashboard - Professional Modern Dark (Cyborg)
Refactored for "Hybrid Modern Dark" design.
"""
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import io
import os
import time
import json

from advanced_catdap.frontend.api_client import APIClient
from advanced_catdap.service.schema import AnalysisParams
from advanced_catdap.service.exporter import ResultExporter

# Initialize API client
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
client = APIClient(base_url=API_URL)

def configure_api_client(api_url):
    global client
    print(f"[INFO] 配置 API Client to: {api_url}")
    client = APIClient(base_url=api_url)

# Initialize Dash app with CYBORG theme (Dark)
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP],
    title="AdvancedCATDAP Analysis",
    suppress_callback_exceptions=True
)

# ============================================================
# Constants & Helpers
# ============================================================

NEON_CYAN = "#00f3ff"
NEON_MAGENTA = "#bc13fe"
NEON_GREEN = "#0aff0a"
TEMPLATE = "plotly_dark"

def create_kpi_card(value, label, subvalue=None, color=None):
    """Create a Glassmorphism KPI Card"""
    style_val = {}
    if color:
        style_val['color'] = color
        # Override the background gradient if specific color requested, or keep css class
    
    content = [
        html.Div(value, className="kpi-value", style=style_val),
        html.Div(label, className="kpi-label")
    ]
    if subvalue:
        content.append(html.Div(subvalue, className="mt-2 text-muted small"))
        
    return html.Div(content, className="glass-card text-center h-100 d-flex flex-column justify-content-center")

def apply_chart_style(fig):
    """Apply modern dark styling to Plotly figures"""
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI", size=12, color="#e0e0e0"),
        title_font=dict(size=16, color=NEON_CYAN),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig

# ============================================================
# Component Generators
# ============================================================

def create_sidebar_content():
    return html.Div([
        # Upload Area
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.Div("Drag and Drop or Click to Select File", className="text-info mb-2"),
                html.Div("Supports: CSV, Parquet, Excel", className="small text-secondary")
            ], className="upload-box"),
            multiple=False,
            className="mb-4"
        ),
        
        html.Div(id='upload-status', className="mb-3"),
        
        # Dynamic Content Area (Target/Task/Run)
        html.Div(id='sidebar-dynamic-content'),
        
        # Expert Area (Always Visible for now to fix callback ID error)
        html.Div([
            html.Hr(className="border-secondary"),
            html.H6("Export Report", className="text-info"),
            html.Label("Export Filename", className="mt-2 text-secondary small"),
            dbc.Input(id="report-filename-input", placeholder="Default: [Dataset]_[Date]", size="sm", className="mb-2"),
            dbc.Button([
                html.I(className="bi bi-download me-2"), "Download HTML Report"
            ], id='btn-export-html', color="info", className="w-100 neon-button")
        ], id='sidebar-export-area', className="d-grid gap-2 mt-4")
    ])

def render_dashboard_tab(result, meta, theme=None): # theme arg kept for compatibility but unused
    if not result:
        return dbc.Alert([html.I(className="bi bi-info-circle me-2"), "Please upload data and run analysis."], color="dark", className="glass-card border-info")
    
    # KPI Calculation
    baseline = result.get('baseline_score', 0)
    fi_data = result.get('feature_importances', [])
    best_aic = baseline
    if fi_data:
        df_fi = pd.DataFrame(fi_data)
        col_map = {c.lower(): c for c in df_fi.columns}
        score_col = col_map.get('score', 'Score')
        if score_col in df_fi.columns:
            best_aic = df_fi[score_col].min()
    
    delta = baseline - best_aic # Improvement
    n_selected = len(fi_data)
    n_total = meta['n_columns'] - 1 if meta else 0

    # KPI Row
    kpi_row = dbc.Row([
        dbc.Col(create_kpi_card(f"{baseline:,.0f}", "Baseline AIC"), width=6, lg=3),
        dbc.Col(create_kpi_card(f"{best_aic:,.0f}", "Optimized AIC", f"Improv: {delta:,.0f}", color=NEON_GREEN), width=6, lg=3),
        dbc.Col(create_kpi_card(f"{n_selected}", "Selected Features", f"Out of {n_total}"), width=6, lg=3),
        dbc.Col(create_kpi_card("Auto", "Model Type", result.get('mode', 'N/A')), width=6, lg=3),
    ], className="mb-4")

    # Feature Importance Chart
    fig_fi = go.Figure()
    if fi_data:
        df_fi = pd.DataFrame(fi_data)
        col_map = {c.lower(): c for c in df_fi.columns}
        feat_col = col_map.get('feature', 'Feature')
        delta_col = col_map.get('delta_score', col_map.get('deltascore', 'Delta_Score'))
        
        if feat_col in df_fi.columns and delta_col in df_fi.columns:
            print("[DEBUG] df_fi columns:", df_fi.columns)
            print("[DEBUG] df_fi head:\n", df_fi[[feat_col, delta_col]].head())
            df_top = df_fi.nlargest(15, delta_col).sort_values(delta_col, ascending=True)
            fig_fi = px.bar(
                df_top, x=delta_col, y=feat_col, orientation='h',
                title="Top Features by Impact (Delta AIC)",
                color=delta_col,
                color_continuous_scale="Bluyl",
                text_auto='.4s' # Show value with SI prefix (e.g. 4.2k)
            )
            fig_fi.update_traces(marker_line_width=0, textposition='outside')
            fig_fi.update_layout(coloraxis_showscale=False, margin=dict(r=50)) # Add margin for text
            apply_chart_style(fig_fi)

    # Interactions Heatmap
    fig_heat = go.Figure()
    data_ii = result.get('interaction_importances', [])
    if data_ii:
        df_ii = pd.DataFrame(data_ii)
        col_map = {c.lower(): c for c in df_ii.columns}
        feat1_col = col_map.get('feature_1', 'Feature_1')
        feat2_col = col_map.get('feature_2', 'Feature_2')
        gain_col = col_map.get('gain', 'Gain')
        
        if feat1_col in df_ii.columns and feat2_col in df_ii.columns and gain_col in df_ii.columns:
            fig_heat = px.density_heatmap(
                df_ii, x=feat1_col, y=feat2_col, z=gain_col, histfunc="sum",
                title="Interaction Network",
                color_continuous_scale="Viridis" # Good for dark
            )
            apply_chart_style(fig_heat)

    # Charts Layout
    charts_row = dbc.Row([
        dbc.Col(html.Div(dcc.Graph(figure=fig_fi), className="glass-card"), md=12, lg=6),
        dbc.Col(html.Div(dcc.Graph(figure=fig_heat), className="glass-card"), md=12, lg=6)
    ])

    return html.Div([kpi_row, charts_row])

def render_deepdive_tab(result, selected_mode, selected_feature, theme, meta=None, target_col=None, selected_interaction_pair=None):
    if not result:
        return dbc.Alert("Run analysis first.", color="warning", className="glass-card")
    
    # Match Streamlit logic: Use keys from feature_details for dropdown options
    feature_details = result.get('feature_details', {})
    fi_data = result.get('feature_importances', [])
    dropdown_features = sorted(list(feature_details.keys()))
    
    # Exclude target_col if present
    if target_col and target_col in dropdown_features:
        dropdown_features.remove(target_col)
    
    print(f"[DEBUG] Render DeepDive Mode: {selected_mode}, Feature Count: {len(dropdown_features)}")
    
    # Feature Selector
    selector_card = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Selection Mode"),
                dbc.RadioItems(
                    id={'type': 'deepdive-mode', 'index': 0},
                    options=[
                        {'label': 'Top 5 Drivers', 'value': 'top5'},
                        {'label': 'Select Feature', 'value': 'select'}
                    ],
                    value=selected_mode or 'top5', inline=True,
                    className="mb-2"
                )
            ], md=4),
            dbc.Col([
                dbc.Label("Select Feature"),
                dbc.Select(
                    id={'type': 'deepdive-feat-select', 'index': 0},
                    options=[{'label': f, 'value': f} for f in dropdown_features],
                    value=selected_feature or (dropdown_features[0] if dropdown_features else None),
                    placeholder="Select feature...",
                    disabled=(selected_mode != 'select'),
                    # className handled by css overrides
                )
            ], md=8)
        ], className="align-items-center")
    ], className="glass-card p-3 mb-3")

    content_area = []
    features_to_show = []
    feature_details = result.get('feature_details', {})
    
    if selected_mode == 'top5':
         if fi_data:
            df_fi = pd.DataFrame(fi_data)
            col_map = {c.lower(): c for c in df_fi.columns}
            feat_col = col_map.get('feature', 'Feature')
            delta_col = col_map.get('delta_score', 'Delta_Score')
            if feat_col in df_fi.columns and delta_col in df_fi.columns:
                features_to_show = df_fi.nlargest(5, delta_col)[feat_col].tolist()
    else:
        # Select Mode
        if selected_feature:
            features_to_show = [selected_feature]
        elif dropdown_features:
            # Fallback to first feature if None selected
            features_to_show = [dropdown_features[0]]
        
    for feat in features_to_show:
        detail = feature_details.get(feat, {})
        if not detail: continue
        
        bin_counts = detail.get('bin_counts', [])
        bin_means = detail.get('bin_means', [])
        # Bin labeling logic
        bin_labels = detail.get('bin_labels', [])
        bin_edges = detail.get('bin_edges', [])
        if bin_counts and not bin_labels and bin_edges and len(bin_edges) == len(bin_counts) + 1:
            bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(bin_counts))]
        elif bin_counts and not bin_labels:
             bin_labels = [f"Bin {i}" for i in range(len(bin_counts))]
        
        # Determine labels
        mode_val = result.get('mode', 'auto').upper()
        target_label = "Avg Value" if mode_val == 'REGRESSION' else "Target Rate"

        # Chart
        fig = go.Figure()
        # Bar (Count)
        fig.add_trace(go.Bar(
            x=bin_labels, y=bin_counts, name='Count',
            marker_color='rgba(255,255,255,0.2)',
            yaxis='y'
        ))
        # Line (Target)
        if bin_means:
            fig.add_trace(go.Scatter(
                x=bin_labels, y=bin_means, name=target_label,
                yaxis='y2', mode='lines+markers',
                line=dict(color=NEON_MAGENTA, width=3)
            ))
            
        fig.update_layout(
            title=f"Analysis: {feat}",
            xaxis_title="Bins",
            yaxis=dict(title="Sample Count"),
            yaxis2=dict(title=target_label, overlaying='y', side='right', showgrid=False),
            legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center')
        )
        apply_chart_style(fig)
        
        # Table DF
        df_table = pd.DataFrame({'Bin': bin_labels, 'Count': bin_counts})
        if bin_means: df_table[target_label] = bin_means
        
        # Chart + Table Row
        content_area.append(html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig), md=8),
                dbc.Col([
                    html.H6("Stats Data", className="text-secondary mb-2"),
                    dbc.Table.from_dataframe(df_table, striped=False, bordered=False, hover=True,
                        className="table-glass small")
                ], md=4, className="d-flex flex-column justify-content-center")
            ])
        ], className="glass-card mb-3"))

    # Interaction Detail
    interaction_area = []
    interaction_details = result.get('interaction_details', {})
    if interaction_details:
        int_keys = list(interaction_details.keys())
        current_pair = selected_interaction_pair or int_keys[0]
        
        interaction_area.append(html.H4("Interaction Detail", className="text-info mt-4 mb-3"))
        interaction_area.append(dcc.Dropdown(
            id={'type': 'deepdive-interaction-select', 'index': 0},
            options=[{'label': k, 'value': k} for k in int_keys],
            value=current_pair,
            clearable=False,
            className="mb-3"
        ))
        
        det = interaction_details.get(current_pair)
        if det:
            fig_int = go.Figure(data=go.Heatmap(
                z=det['means'], x=det['bin_labels_2'], y=det['bin_labels_1'],
                colorscale='Viridis', colorbar=dict(title="Target")
            ))
            apply_chart_style(fig_int)
            fig_int.update_layout(title=f"{det['feature_1']} vs {det['feature_2']}")
            interaction_area.append(html.Div(dcc.Graph(figure=fig_int), className="glass-card"))

            # Interaction Tables
            df_counts = pd.DataFrame(det['counts'], index=det['bin_labels_1'], columns=det['bin_labels_2'])
            df_means = pd.DataFrame(det['means'], index=det['bin_labels_1'], columns=det['bin_labels_2'])
            
            # Format means
            df_means = df_means.map(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

            # Reset index for display
            df_counts_disp = df_counts.reset_index().rename(columns={'index': det['feature_1']})
            df_means_disp = df_means.reset_index().rename(columns={'index': det['feature_1']})

            interaction_area.append(dbc.Row([
                dbc.Col([
                    html.H6("Sample Count Matrix", className="text-secondary mt-3"),
                    dbc.Table.from_dataframe(df_counts_disp, striped=False, bordered=True, hover=True, className="table-glass small")
                ], md=6),
                dbc.Col([
                    html.H6("Target Mean Matrix", className="text-secondary mt-3"),
                    dbc.Table.from_dataframe(df_means_disp, striped=False, bordered=True, hover=True, className="table-glass small")
                ], md=6)
            ]))



    return html.Div([selector_card] + content_area + interaction_area)

# ============================================================
# Main Layout
# ============================================================

app.layout = dbc.Container([
    # Store Components
    dcc.Store(id='store-dataset-meta', storage_type='memory'),
    dcc.Store(id='store-analysis-result', storage_type='memory'),
    dcc.Store(id='store-analysis-params', storage_type='memory'),
    dcc.Store(id='store-job-id', storage_type='memory'),
    dcc.Store(id='store-deepdive-state', data={'mode': 'top5', 'feature': None, 'interaction': None}, storage_type='memory'),
    dcc.Store(id='theme-store', data='dark', storage_type='local'), # Force Dark
    dcc.Interval(id='job-poll-interval', interval=2000, disabled=True),
    dcc.Download(id="download-html-report"),
    
    # 1. Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="bi bi-cpu-fill fs-3 text-info me-2"),
                html.H3("AdvancedCATDAP", className="d-inline align-middle m-0 text-white")
            ], className="d-flex align-items-center py-3")
        ])
    ], className="mb-2"),

    # 2. Main Grid
    dbc.Row([
        # Left Sidebar (3/12)
        dbc.Col([
            html.Div(create_sidebar_content(), className="glass-card sticky-sidebar"),
            # Placeholder for dynamic content or export area (if we want it sticky too, move inside)
            # Actually, let's keep it simple. create_sidebar_content returns the structure.
            # We need to ensure 'report-filename-input' is in the initial layout if possible, 
            # OR we ensure the callback handles its absence (but State requires it presence).
            # BEST FIX: Add it to the initial layout in create_sidebar_content or here hidden.
        ], width=12, md=3, className="mb-4"),
        
        # Right Main Content (9/12)
        dbc.Col([
            html.Div(id='global-status-message'),
            
            # Use 'main-tabs' ID as requested
            dbc.Tabs([
                dbc.Tab(label="Dashboard", tab_id="tab-dashboard", label_class_name="text-uppercase"),
                dbc.Tab(label="Deep Dive", tab_id="tab-deepdive", label_class_name="text-uppercase"),
            ], id="main-tabs", active_tab="tab-dashboard", className="mb-3"),
            
            html.Div(id='page-content')
            
        ], width=12, md=9)
    ])

], fluid=True, style={'minHeight': '100vh'})


# ============================================================
# Callbacks
# ============================================================

@callback(
    Output('upload-status', 'children'),
    Output('store-dataset-meta', 'data'),
    Output('sidebar-dynamic-content', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    if not contents: return dash.no_update, dash.no_update, dash.no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        file_obj = io.BytesIO(decoded)
        meta = client.upload_dataset(file_obj, filename)
        meta_dict = meta.model_dump()
        col_names = [c['name'] for c in meta_dict['columns']]
        
        settings = html.Div([
            html.Hr(className="border-secondary"), 
            html.H6("Analysis Settings", className="text-info"),
            
            html.Label("Target Variable", className="small text-muted"),
            dcc.Dropdown(
                id='target-col', 
                options=[{'label': c, 'value': c} for c in col_names], 
                value=col_names[0] if col_names else None, 
                className="mb-3"
            ),
            
            html.Label("Task Type", className="small text-muted"),
                dcc.Dropdown(
                    id='task-type',
                    options=[
                        {'label': 'Auto', 'value': 'auto'},
                        {'label': 'Classification', 'value': 'classification'},
                        {'label': 'Regression', 'value': 'regression'}
                    ],
                    value='auto',
                    clearable=False,
                    className="mb-3"
                ),
                
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Label("Max Bins (2-20)"),
                        dcc.Slider(id='max-bins', min=2, max=20, step=1, value=5, marks={i: str(i) for i in range(2, 21, 2)}),
                        html.Br(),
                        dbc.Label("Top K Features (1-50)"),
                        dcc.Slider(id='top-k', min=1, max=50, step=1, value=10, marks={i: str(i) for i in [1, 10, 20, 30, 40, 50]}),
                        html.Br(),
                        dbc.Switch(id='use-aicc', label="Use AICc (Corrected)", value=True),
                    ], title="Advanced Settings")
                ], start_collapsed=True, className="mb-3"),

                dbc.Button("Run Analysis", id='run-btn', color="primary", className="w-100 neon-button")
        ], className="mt-3")
        
        return dbc.Alert(f"Loaded: {filename}", color="success", className="py-2 small"), meta_dict, settings
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger"), None, None

@callback(
    Output('store-job-id', 'data'),
    Output('store-analysis-params', 'data'),
    Output('job-poll-interval', 'disabled'),
    Output('global-status-message', 'children'),
    Input('run-btn', 'n_clicks'),
    State('store-dataset-meta', 'data'),
    State('target-col', 'value'),
    State('task-type', 'value'),
    State('max-bins', 'value'),
    State('top-k', 'value'),
    State('use-aicc', 'value'),
    prevent_initial_call=True
)
def submit_job(n_clicks, meta, target_col, task_type, max_bins, top_k, use_aicc):
    if not n_clicks or not meta: return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    print(f"[DEBUG] Submit Job Clicked. Target: {target_col}")
    try:
        # Default params
        col_names = [c['name'] for c in meta['columns']]
        candidates = [c for c in col_names if c != target_col]
        params = AnalysisParams(
            target_col=target_col, 
            candidate_features=candidates, 
            task_type=task_type or "auto",
            max_bins=max_bins,
            top_k=top_k,
            use_aicc=use_aicc
        )
        
        job_id = client.submit_job(meta['dataset_id'], params)
        print(f"[DEBUG] Job Submitted: {job_id}")
        msg = dbc.Alert([
            dbc.Spinner(size="sm", spinner_class_name="me-2"),
            f"Analysis started... (Job: {job_id[:6]})"
        ], color="info", className="glass-card border-info")
        
        # Save params
        params_dict = {'target_col': target_col, 'task_type': task_type}
        return job_id, params_dict, False, msg
    except Exception as e:
        print(f"[ERROR] Job Submission Failed: {e}")
        return None, None, True, dbc.Alert(f"Failed: {e}", color="danger", className="glass-card")

@callback(
    Output('store-analysis-result', 'data'),
    Output('job-poll-interval', 'disabled', allow_duplicate=True),
    Output('global-status-message', 'children', allow_duplicate=True),
    Input('job-poll-interval', 'n_intervals'),
    State('store-job-id', 'data'),
    prevent_initial_call=True
)
def poll_job(n, job_id):
    if not job_id: return dash.no_update, True, dash.no_update
    try:
        info = client.get_job_status(job_id)
        status = info.get('status')
        
        if status in ['completed', 'SUCCESS']:
            return info.get('result'), True, dbc.Alert("✅ Analysis Completed Successfully", color="success", className="glass-card")
        elif status in ['failed', 'FAILURE']:
             return None, True, dbc.Alert(f"❌ Job Failed: {info.get('error')}", color="danger", className="glass-card")
        else:
            stage = info.get('progress', {}).get('stage', 'Processing...')
            return dash.no_update, False, dbc.Alert([dbc.Spinner(size="sm", spinner_class_name="me-2"), stage], color="info", className="glass-card border-info")
    except Exception as e:
         return None, True, dbc.Alert(f"polling Error: {e}", color="danger")

@callback(
    Output('sidebar-export-area', 'children'),
    Input('store-analysis-result', 'data'),
    prevent_initial_call=True
)
def show_export_button(result):
    if not result: return ""
    return html.Div([
        dbc.Button([
            html.I(className="bi bi-filetype-html me-2"),
            "Export Interactive Report"
        ], id='btn-export-html', color="info", className="w-100 neon-button")
    ])



@callback(
    Output("download-html-report", "data"),
    Input("btn-export-html", "n_clicks"),
    State("store-analysis-result", "data"),
    State("store-dataset-meta", "data"),
    State("report-filename-input", "value"),
    prevent_initial_call=True
)
def download_html_report(n_clicks, result, meta, custom_filename):
    if not n_clicks or not result: return dash.no_update
    
    # Generate filename
    if custom_filename and custom_filename.strip():
        filename = custom_filename.strip()
        if not filename.lower().endswith(".html"):
            filename += ".html"
    else:
        # Default logic
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        original_name = "AdvancedCATDAP"
        if meta:
            if 'filename' in meta and meta['filename']:
                 base = os.path.basename(meta['filename'])
                 original_name, _ = os.path.splitext(base)
            elif 'dataset_id' in meta:
                base = os.path.basename(meta['dataset_id'])
                original_name, _ = os.path.splitext(base)
        
        filename = f"{original_name}_Report_{timestamp}.html"
    
    try:
        html_io = ResultExporter.generate_html_report(result, meta)
        return dcc.send_bytes(html_io.getvalue(), filename)
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        # In a real app, we might want to show a notification to the user
        return dash.no_update

# Removed Simulator Callbacks


@callback(
    Output('page-content', 'children'),
    Input('main-tabs', 'active_tab'),
    Input('store-analysis-result', 'data'),
    Input('store-deepdive-state', 'data'),
    Input('theme-store', 'data'),
    State('store-dataset-meta', 'data'),
    State('store-analysis-params', 'data')
)
def render_content(active_tab, result, deepdive_state, theme, meta, params): # theme unused but kept in sig
    try:
        if active_tab == 'tab-dashboard':
            return render_dashboard_tab(result, meta)
        elif active_tab == 'tab-deepdive':
            mode = deepdive_state.get('mode', 'top5') if deepdive_state else 'top5'
            feat = deepdive_state.get('feature') if deepdive_state else None
            interaction = deepdive_state.get('interaction') if deepdive_state else None
            target_col = params.get('target_col') if params else None
            return render_deepdive_tab(result, mode, feat, theme, meta, target_col, interaction)
        return html.Div("Tab Not Found")
    except Exception as e:
        import traceback; traceback.print_exc()
        return dbc.Alert(f"Render Error: {e}", color="danger")

@callback(
    Output('store-deepdive-state', 'data'),
    Input({'type': 'deepdive-mode', 'index': ALL}, 'value'),
    Input({'type': 'deepdive-feat-select', 'index': ALL}, 'value'),
    Input({'type': 'deepdive-interaction-select', 'index': ALL}, 'value'),
    State('store-deepdive-state', 'data'),
    prevent_initial_call=True
)
def update_deepdive_state(mode_vals, feat_vals, int_vals, current_state):
    ctx_id = ctx.triggered_id
    if not ctx_id: return dash.no_update
    print(f"[DEBUG] Update DeepDive State: Trigger={ctx_id}, ModeVals={mode_vals}")
    
    new_state = current_state.copy()
    # Handle dictionary ID pattern matching
    trigger_type = ctx_id.get('type') if isinstance(ctx_id, dict) else str(ctx_id)
    
    if trigger_type == 'deepdive-mode' and mode_vals:
         new_state['mode'] = mode_vals[0]
    elif trigger_type == 'deepdive-feat-select' and feat_vals:
         new_state['feature'] = feat_vals[0]
    elif trigger_type == 'deepdive-interaction-select' and int_vals:
         new_state['interaction'] = int_vals[0]
    
    return new_state

server = app.server
if __name__ == '__main__':
    app.run(debug=True, port=8050)
