"""
AdvancedCATDAP Dashboard - Dash Version (Final UX & Dark Mode)
Full migration from Streamlit to Dash.
Fixes: Deep Dive data structure, Visual Parity, Dark Mode, Interaction Analysis.
"""
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, MATCH, clientside_callback
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

# Initialize API client
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
client = APIClient(base_url=API_URL)

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    title="AdvancedCATDAP Dashboard",
    suppress_callback_exceptions=True
)

# ============================================================
# CSS & Theme Management
# ============================================================

# Styles are loaded from assets/style.css

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "overflowY": "auto",
    "transition": "background-color 0.3s"
}

CONTENT_STYLE = {
    "marginLeft": "21rem",
    "marginRight": "1rem",
    "padding": "2rem 1rem",
    "transition": "background-color 0.3s"
}

KPI_CARD_STYLE = {
    'textAlign': 'center', 'padding': '1rem', 'borderRadius': '8px', 
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
}
KPI_VALUE_STYLE = {'fontSize': '1.8rem', 'fontWeight': 'bold'}
KPI_LABEL_STYLE = {'fontSize': '0.9rem', 'opacity': 0.8}

# ============================================================
# Helper Functions
# ============================================================

def get_plotly_template(theme):
    return "plotly_dark" if theme == "dark" else "plotly_white"

def create_kpi_card(value, label, subvalue=None):
    content = [
        html.Div(value, style=KPI_VALUE_STYLE, className="kpi-value"),
        html.Div(label, style=KPI_LABEL_STYLE)
    ]
    if subvalue:
        content.append(html.Div(subvalue, style={'fontSize': '0.8rem', 'marginTop': '4px', 'opacity': 0.7}))
        
    return html.Div(content, style=KPI_CARD_STYLE, className="kpi-card")

# ============================================================
# Components Generators
# ============================================================

def create_sidebar_content():
    return html.Div([
        html.Div([
            html.H3("AdvancedCATDAP", className="display-6 fs-4"),
            dbc.Switch(id="theme-switch", label="🌙 Dark Mode", value=False, className="float-end"),
        ], className="d-flex justify-content-between align-items-center"),
        html.Hr(),
        html.Label("Data Upload", className="fw-bold mt-2"),
        dcc.Upload(
            id='upload-data',
            children=dbc.Button("📂 Upload CSV/Parquet", color="secondary", outline=True, className="w-100"),
            multiple=False
        ),
        html.Div(id='upload-status', className="mt-2 small"),
        html.Div(id='sidebar-dynamic-content')
    ], className="sidebar-content")

def render_dashboard_tab(result, meta, theme):
    if not result:
        return dbc.Alert("👈 Please upload data and run analysis from the sidebar.", color="info", className="mt-4")
    
    template = get_plotly_template(theme)

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
    
    delta = best_aic - baseline
    n_selected = len(fi_data)
    n_total = meta['n_columns'] - 1 if meta else 0

    kpi_row = dbc.Row([
        dbc.Col(create_kpi_card(f"{baseline:,.1f}", "Baseline AIC"), md=4),
        dbc.Col(create_kpi_card(f"{best_aic:,.1f}", "Best AIC", f"({delta:+,.1f})"), md=4),
        dbc.Col(create_kpi_card(f"{n_selected} / {n_total}", "Selected Features"), md=4),
    ], className="mb-4")

    # Feature Importance
    fig_fi = go.Figure()
    if fi_data:
        df_fi = pd.DataFrame(fi_data)
        col_map = {c.lower(): c for c in df_fi.columns}
        feat_col = col_map.get('feature', 'Feature')
        delta_col = col_map.get('delta_score', col_map.get('deltascore', 'Delta_Score'))
        
        if feat_col in df_fi.columns and delta_col in df_fi.columns:
            df_top = df_fi.nlargest(15, delta_col).sort_values(delta_col, ascending=True)
            fig_fi = px.bar(
                df_top, 
                x=delta_col, 
                y=feat_col, 
                orientation='h',
                color=delta_col,
                color_continuous_scale="Blues",
                template=template
            )
            fig_fi.update_layout(
                title="Feature Importance (Top 15)",
                height=400,
                margin=dict(l=0, r=20, t=40, b=0),
                coloraxis_showscale=False
            )

    # Interactions High Level
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
                df_ii,
                x=feat1_col,
                y=feat2_col,
                z=gain_col,
                histfunc="sum",
                color_continuous_scale="Blues",
                template=template
            )
            fig_heat.update_layout(
                title="Top Interactions Overview",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )

    return html.Div([kpi_row, dbc.Row([dbc.Col(dcc.Graph(figure=fig_fi), md=6), dbc.Col(dcc.Graph(figure=fig_heat), md=6)])])

def render_deepdive_tab(result, selected_mode, selected_feature, theme, selected_interaction_pair=None):
    if not result:
        return dbc.Alert("Run analysis first.", color="warning")
    
    template = get_plotly_template(theme)
    fi_data = result.get('feature_importances', [])
    features = []
    if fi_data:
        df_fi = pd.DataFrame(fi_data)
        col_map = {c.lower(): c for c in df_fi.columns}
        feat_col = col_map.get('feature', 'Feature')
        if feat_col in df_fi.columns: features = df_fi[feat_col].tolist()
    
    # 1. Feature Selector Area
    selector_area = dbc.Row([
        dbc.Col([
            dbc.RadioItems(
                id={'type': 'deepdive-mode', 'index': 0},
                options=[{'label': 'Top 5 Drivers', 'value': 'top5'}, {'label': 'Select Feature', 'value': 'select'}],
                value=selected_mode or 'top5', inline=True, className="mb-2"
            )
        ], md=4),
        dbc.Col([
            dcc.Dropdown(
                id={'type': 'deepdive-feat-select', 'index': 0},
                options=[{'label': f, 'value': f} for f in features],
                value=selected_feature or (features[0] if features else None),
                placeholder="Select feature...", disabled=(selected_mode != 'select'),
                className="dropdown-dark-mode-compatible" # Use custom class
            )
        ], md=8)
    ], className="mb-4 align-items-center p-2 rounded", style={'backgroundColor': 'var(--card-bg)', 'border': '1px solid var(--border-color)'})

    # 2. detailed Charts Area
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
    elif selected_feature:
        features_to_show = [selected_feature]
        
    for feat in features_to_show:
        detail = feature_details.get(feat, {})
        if not detail: continue
        
        bin_counts = detail.get('bin_counts', [])
        bin_means = detail.get('bin_means', [])
        bin_labels = detail.get('bin_labels', [])
        bin_edges = detail.get('bin_edges', [])
        
        if bin_counts:
            # Better label logic (from edges)
            if not bin_labels and bin_edges and len(bin_edges) == len(bin_counts) + 1:
                bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(bin_counts))]
            elif not bin_labels:
                 bin_labels = [f"Bin {i}" for i in range(len(bin_counts))]
            
            # Determine logic
            mode_val = result.get('mode', 'auto').upper()
            is_regression = mode_val == 'REGRESSION'
            target_label = "Avg Value" if is_regression else "Target Rate"

            # Create DF
            plot_data = {
                'Value Range': bin_labels,
                'Sample Count': bin_counts,
            }
            if bin_means:
                plot_data[target_label] = bin_means
                
            df_plot = pd.DataFrame(plot_data)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_plot['Value Range'], 
                y=df_plot['Sample Count'], 
                name='Count', 
                marker_color='#bdc3c7',
                yaxis='y'
            ))
            
            if target_label in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot['Value Range'], 
                    y=df_plot[target_label], 
                    name=target_label, 
                    yaxis='y2', 
                    line=dict(color='#e74c3c', width=3),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title=f"<b>{feat}</b> Impact",
                template=template,
                height=350,
                xaxis=dict(title="Value Range"),
                yaxis=dict(title="Count"),
                yaxis2=dict(
                    title=target_label, 
                    overlaying='y', 
                    side='right', 
                    showgrid=False
                ),
                legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # Layout: Chart Left, Table Right
            card_content = dbc.Row([
                dbc.Col(dcc.Graph(figure=fig), md=7),
                dbc.Col([
                    html.H6("Statistics", className="text-center mb-2"),
                    dbc.Table.from_dataframe(
                        df_plot, 
                        striped=True, 
                        bordered=True, 
                        hover=True, 
                        responsive=True,
                        style={'fontSize': '0.8rem'},
                        className="text-center"
                    )
                ], md=5)
            ])
            
            content_area.append(dbc.Card(dbc.CardBody(card_content), className="mb-3", style={'backgroundColor': 'var(--card-bg)', 'borderColor': 'var(--border-color)'}))

    # 3. Interaction Detail Section
    interaction_area = []
    interaction_details = result.get('interaction_details', {})
    if interaction_details:
        int_keys = list(interaction_details.keys())
        # Add Interaction Selector
        interaction_area.append(html.Hr())
        interaction_area.append(html.H4("Interaction Analysis", className="mb-3"))
        
        current_pair = selected_interaction_pair or int_keys[0]
        
        interaction_area.append(dcc.Dropdown(
            id={'type': 'deepdive-interaction-select', 'index': 0},
            options=[{'label': k, 'value': k} for k in int_keys],
            value=current_pair,
            clearable=False,
            className="mb-3 dropdown-dark-mode-compatible"
        ))
        
        # Detail Heatmap
        det = interaction_details.get(current_pair)
        if det:
            fig_int = go.Figure(data=go.Heatmap(
                z=det['means'],
                x=det['bin_labels_2'],
                y=det['bin_labels_1'],
                colorscale='Blues',
                colorbar=dict(title="Target")
            ))
            fig_int.update_layout(
                title=f"Interaction: {det['feature_1']} vs {det['feature_2']}",
                xaxis_title=det['feature_2'],
                yaxis_title=det['feature_1'],
                template=template,
                height=400
            )
            interaction_area.append(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_int)), style={'backgroundColor': 'var(--card-bg)', 'borderColor': 'var(--border-color)'}))
            
    return html.Div([selector_area] + (content_area if content_area else [dbc.Alert("No detail data available for selected feature(s).", color="secondary")]) + interaction_area)

# ============================================================
# Main Layout
# ============================================================
app.layout = html.Div([

    dcc.Store(id='store-dataset-meta', storage_type='memory'),
    dcc.Store(id='store-analysis-result', storage_type='memory'),
    dcc.Store(id='store-job-id', storage_type='memory'),
    dcc.Store(id='store-deepdive-state', data={'mode': 'top5', 'feature': None, 'interaction': None}, storage_type='memory'),
    dcc.Store(id='theme-store', data='light', storage_type='local'), # Persist theme
    dcc.Interval(id='job-poll-interval', interval=2000, disabled=True),
    
    html.Div(id='main-container', className="main-content", children=[
        html.Div(create_sidebar_content(), style=SIDEBAR_STYLE, className="sidebar"),
        html.Div([
            html.Div(id='global-status-message'),
            dbc.Tabs(id="main-tabs", active_tab="tab-dashboard", children=[
                dbc.Tab(label="Dashboard", tab_id="tab-dashboard"),
                dbc.Tab(label="Deep Dive", tab_id="tab-deepdive"),
                dbc.Tab(label="Simulator", tab_id="tab-simulator"),
            ], className="mb-3"),
            html.Div(id='page-content')
        ], style=CONTENT_STYLE)
    ])
])

# ============================================================
# Callbacks
# ============================================================

# Theme Switcher (Clientside)
clientside_callback(
    """
    function(is_dark) {
        if (is_dark) {
            document.documentElement.setAttribute('data-theme', 'dark');
            return 'dark';
        } else {
            document.documentElement.setAttribute('data-theme', 'light');
            return 'light';
        }
    }
    """,
    Output('theme-store', 'data'),
    Input('theme-switch', 'value')
)

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
            html.Hr(), html.H6("Settings"),
            html.Label("Target Column", className="small"),
            dcc.Dropdown(id='target-col', options=[{'label': c, 'value': c} for c in col_names], value=col_names[0] if col_names else None, className="mb-2 text-dark"),
             html.Label("Task Type", className="small"),
             dcc.Dropdown(id='task-type', options=[{'label': 'Auto', 'value': 'auto'}, {'label': 'Classification', 'value': 'classification'}, {'label': 'Regression', 'value': 'regression'}], value='auto', className="mb-2 text-dark"),
             dbc.Button("🚀 Run Analysis", id='run-btn', color="primary", className="w-100 mt-3")
        ], className="mt-3")
        return dbc.Alert(f"✅ {filename}", color="success", className="p-1"), meta_dict, settings
    except Exception as e:
        return dbc.Alert(f"❌ Error: {str(e)}", color="danger"), None, None

@callback(
    Output('store-job-id', 'data'),
    Output('job-poll-interval', 'disabled'),
    Output('global-status-message', 'children'),
    Input('run-btn', 'n_clicks'),
    State('store-dataset-meta', 'data'),
    State('target-col', 'value'),
    State('task-type', 'value'),
    prevent_initial_call=True
)
def submit_job(n_clicks, meta, target_col, task_type):
    if not n_clicks or not meta: return dash.no_update, dash.no_update, dash.no_update
    try:
        col_names = [c['name'] for c in meta['columns']]
        candidates = [c for c in col_names if c != target_col]
        params = AnalysisParams(target_col=target_col, candidate_features=candidates, task_type=task_type or "auto", max_bins=5, min_samples_per_bin=10, use_aicc=True)
        job_id = client.submit_job(meta['dataset_id'], params)
        return job_id, False, dbc.Alert(f"Processing... (Job: {job_id[:6]})", color="info")
    except Exception as e:
        return None, True, dbc.Alert(f"Failed: {e}", color="danger")

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
            return info.get('result'), True, dbc.Alert("✅ Analysis Completed!", color="success")
        elif status in ['failed', 'FAILURE']:

             return None, True, dbc.Alert(f"❌ Job Failed: {info.get('error')}", color="danger")
        else:
            return dash.no_update, False, dbc.Alert(f"⏳ {info.get('progress', {}).get('stage', 'Running...')}", color="info")
    except Exception as e:
         return None, True, dbc.Alert(f"❌ Error during polling: {e}", color="danger")

@callback(
    Output('page-content', 'children'),
    Input('main-tabs', 'active_tab'),
    Input('store-analysis-result', 'data'),
    Input('store-deepdive-state', 'data'),
    Input('theme-store', 'data'),
    State('store-dataset-meta', 'data')
)
def render_content(active_tab, result, deepdive_state, theme, meta):

    try:
        if active_tab == 'tab-dashboard':
            return render_dashboard_tab(result, meta, theme)
        elif active_tab == 'tab-deepdive':
            mode = deepdive_state.get('mode') if deepdive_state else 'top5'
            feat = deepdive_state.get('feature') if deepdive_state else None
            interaction = deepdive_state.get('interaction') if deepdive_state else None
            return render_deepdive_tab(result, mode, feat, theme, interaction)
        elif active_tab == 'tab-simulator':
            return dbc.Alert("Simulator module not yet implemented.", color="secondary")
        return html.Div(f"Unknown tab: {active_tab}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"❌ Error rendering content: {e}", color="danger")

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
    new_state = current_state.copy()
    
    if 'deepdive-mode' in str(ctx_id) and mode_vals: 
        new_state['mode'] = mode_vals[0]
    elif 'deepdive-feat-select' in str(ctx_id) and feat_vals: 
        new_state['feature'] = feat_vals[0]
    elif 'deepdive-interaction-select' in str(ctx_id) and int_vals:
        new_state['interaction'] = int_vals[0]
        
    return new_state

server = app.server
if __name__ == '__main__':
    app.run(debug=True, port=8050)
