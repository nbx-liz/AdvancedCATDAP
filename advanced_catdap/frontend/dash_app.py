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
import numpy as np
import base64
import io
import os
import time
import json
import re
import logging

from advanced_catdap.frontend.api_client import APIClient
from advanced_catdap.service.schema import AnalysisParams
from advanced_catdap.service.exporter import ResultExporter

# Initialize API client
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
client = APIClient(base_url=API_URL)
logger = logging.getLogger(__name__)

def configure_api_client(api_url):
    global client
    logger.info("Configuring API client to %s", api_url)
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
UNIFIED_BAR_COLORSCALE = [
    [0.00, "#1a2230"],
    [0.45, "#24506a"],
    [0.75, "#2f7f9a"],
    [1.00, "#78f3ff"],
]
UNIFIED_HEATMAP_COLORSCALE = [
    [0.00, "#0f131b"],
    [0.25, "#1a2433"],
    [0.50, "#24485f"],
    [0.75, "#2f7f9a"],
    [1.00, "#78f3ff"],
]
TEMPLATE = "plotly_dark"

def create_kpi_card(value, label, subvalue=None, color=None, class_name=""):
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
        
    return html.Div(
        content,
        className=f"glass-card text-center h-100 d-flex flex-column justify-content-center {class_name}".strip()
    )

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


def _sorted_indices_by_keys(length, sort_keys):
    if not sort_keys or len(sort_keys) != length:
        return list(range(length))
    pairs = [(i, str(sort_keys[i])) for i in range(length)]
    pairs.sort(key=lambda x: x[1])
    return [idx for idx, _ in pairs]


def _reorder_2d(values, row_idx, col_idx):
    if values is None:
        return values
    out = []
    for r in row_idx:
        row = values[r] if r < len(values) else []
        out.append([row[c] if c < len(row) else None for c in col_idx])
    return out

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
                html.Div("Supports: CSV, Parquet", className="small text-secondary")
            ], className="upload-box"),
            multiple=False,
            className="mb-3"
        ),
        
        html.Div(id='upload-status', className="mb-2"),
        
        # Dynamic Content Area (Target/Task/Run)
        html.Div(id='sidebar-dynamic-content', className="mb-1"),
        
        # Expert Area (Always Visible for now to fix callback ID error)
        html.Div([
            html.Hr(className="border-secondary"),
            html.H6("Export Report", className="text-info"),
            html.Label("Export Filename", className="mt-2 text-secondary small"),
            dbc.Input(id="report-filename-input", placeholder="Default: [Dataset]_[Date]", size="sm", className="mb-2"),
            dbc.Button([
                html.I(className="bi bi-download me-2"), "Download HTML Report"
            ], id='btn-export-html', color="info", className="w-100 neon-button")
        ], id='sidebar-export-area', className="sidebar-export-area mt-3")
    ], className="sidebar-stack")


def build_interaction_empty_reason(result, params=None, meta=None):
    ii_data = result.get("interaction_importances", []) if isinstance(result, dict) else []
    interaction_details = result.get("interaction_details", {}) if isinstance(result, dict) else {}
    mode = str((result or {}).get("mode", "")).strip().lower()
    requested_task = str((params or {}).get("task_type", "auto")).strip().lower()
    checked_pairs = (result or {}).get("checked_interaction_pairs")

    lines = [
        "No interaction pair passed the gain threshold.",
    ]
    if requested_task == "auto" and mode:
        lines.append(f"Auto task detection selected: {mode.capitalize()}.")
    lines.append("interaction_importances and interaction_details are empty.")
    if isinstance(checked_pairs, int):
        lines.append(f"Checked pairs: {checked_pairs}")
    lines.append("Tip: If target is continuous, set Task Type=Regression and rerun.")

    return html.Div(
        [
            html.Div("Interaction Network is not available", className="kpi-label mb-2"),
            html.Ul([html.Li(line) for line in lines], className="small text-muted mb-0"),
        ]
    )

def render_dashboard_tab(result, meta, params=None, theme=None): # theme arg kept for compatibility but unused
    if not result:
        return dbc.Alert([html.I(className="bi bi-info-circle me-2"), "Please upload data and run analysis."], color="dark", className="glass-card border-info")
    
    # KPI Calculation
    baseline_raw = result.get('baseline_score', 0)
    baseline_num = pd.to_numeric(pd.Series([baseline_raw]), errors="coerce").iloc[0]
    baseline = (
        float(baseline_num)
        if pd.notna(baseline_num) and np.isfinite(baseline_num)
        else None
    )
    fi_data = result.get('feature_importances', [])
    df_fi_norm = ResultExporter.normalize_feature_importances(fi_data)
    best_aic = baseline
    if not df_fi_norm.empty and "Score" in df_fi_norm.columns:
        score_series = pd.to_numeric(df_fi_norm["Score"], errors="coerce").dropna()
        if not score_series.empty:
            min_score = float(score_series.min())
            best_aic = min_score if np.isfinite(min_score) else best_aic

    if baseline is not None and best_aic is not None and np.isfinite(best_aic):
        delta = baseline - best_aic  # Improvement
        pct_change = ((best_aic - baseline) / baseline * 100.0) if baseline else None
    else:
        delta = None
        pct_change = None

    baseline_text = f"{baseline:,.0f}" if baseline is not None else "N/A"
    best_aic_text = f"{best_aic:,.0f}" if best_aic is not None else "N/A"
    if delta is not None and pct_change is not None and np.isfinite(delta) and np.isfinite(pct_change):
        delta_text = f"Delta {delta:,.0f} ({pct_change:.1f}%)"
    else:
        delta_text = "Delta N/A"
    has_transform_rules = isinstance(result, dict) and "transform_rules" in result
    transform_rules = result.get("transform_rules") if has_transform_rules else None
    if isinstance(transform_rules, dict):
        n_selected = len(transform_rules)
    else:
        n_selected = len(fi_data)
    n_total = max((meta['n_columns'] - 1), 0) if meta else 0
    selected_summary = f"{n_selected} / {n_total} features" if n_total else f"{n_selected} features"

    final_mode = str(result.get('mode', 'N/A')).upper()
    mode_title = final_mode.capitalize() if final_mode != "N/A" else "N/A"
    requested_task = str((params or {}).get('task_type', 'auto')).lower()
    requested_task_title = requested_task.capitalize() if requested_task else "Auto"
    metric_name = "AICc" if bool((params or {}).get('use_aicc', True)) else "AIC"
    estimator_name = "DecisionTreeRegressor bins" if final_mode == "REGRESSION" else "DecisionTreeClassifier bins"
    # KPI Row
    kpi_row = dbc.Row([
        dbc.Col(
            html.Div([
                html.Div("AIC Comparison", className="kpi-label"),
                html.H2(
                    [
                        f"{baseline_text} ",
                        html.Span("->", className="mx-1 text-secondary"),
                        f"{best_aic_text}",
                    ],
                    className="kpi-main-value mb-1"
                ),
                html.Div(
                    [delta_text],
                    className="kpi-delta fw-semibold"
                ),
                html.Small("AIC is better when lower.", className="kpi-note"),
            ], className="glass-card dashboard-kpi-card text-center h-100 d-flex flex-column justify-content-center"),
            width=12, lg=4
        ),
        dbc.Col(
            create_kpi_card(selected_summary, "Selected Features", None, class_name="dashboard-kpi-card"),
            width=12, lg=4
        ),
        dbc.Col(
            html.Div([
                html.Div("Model Type", className="kpi-label"),
                html.H2(f"{mode_title} ({requested_task_title})", className="kpi-main-value mb-1"),
                html.Div(f"Estimator: {estimator_name}", className="kpi-meta"),
                html.Div(f"Metric: {metric_name}", className="kpi-meta"),
            ], className="glass-card dashboard-kpi-card text-center h-100 d-flex flex-column justify-content-center"),
            width=12, lg=4
        ),
    ], className="mb-4 g-3")
    # Feature Importance Chart
    fig_fi = go.Figure()
    if not df_fi_norm.empty:
            logger.debug("normalized feature_importances head:\n%s", df_fi_norm.head())
            df_top = df_fi_norm.nlargest(15, "Delta_Score").sort_values("Delta_Score", ascending=True)
            x_vals = [float(v) for v in df_top["Delta_Score"].tolist()]
            y_vals = df_top["Feature"].astype(str).tolist()
            text_vals = [ResultExporter._format_compact(v) for v in x_vals]
            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                orientation='h',
                text=text_vals,
                texttemplate="%{text}",
                textposition='outside',
                marker=dict(
                    color=x_vals,
                    colorscale=UNIFIED_BAR_COLORSCALE,
                    line=dict(width=0),
                    showscale=False
                ),
                hovertemplate="Feature=%{y}<br>Delta_Score=%{x:.6g}<extra></extra>"
            ))
            fig_fi.update_layout(title="Top Features by Impact (Delta AIC)")
            fig_fi.update_layout(coloraxis_showscale=False, margin=dict(r=50)) # Add margin for text
            apply_chart_style(fig_fi)

    # Interactions Heatmap
    fig_heat = go.Figure()
    data_ii = result.get('interaction_importances', [])
    df_ii_norm = ResultExporter.normalize_interaction_importances(data_ii)
    mat = ResultExporter.build_interaction_matrix(df_ii_norm)
    if mat.empty:
        mat = ResultExporter.build_interaction_matrix_from_details(result.get("interaction_details", {}))
    if not mat.empty:
        fig_heat = go.Figure(data=go.Heatmap(
            z=mat.values,
            x=mat.columns.tolist(),
            y=mat.index.tolist(),
            colorscale=UNIFIED_HEATMAP_COLORSCALE,
            colorbar=dict(title="Gain"),
            hovertemplate="Feature_1=%{x}<br>Feature_2=%{y}<br>Gain=%{z:.6g}<extra></extra>"
        ))
        fig_heat.update_layout(title="Interaction Network")
        apply_chart_style(fig_heat)
        interaction_panel = dcc.Graph(figure=fig_heat)
    else:
        interaction_panel = dbc.Alert(
            build_interaction_empty_reason(result, params=params, meta=meta),
            color="secondary",
            className="mb-0",
        )

    # Charts Layout
    charts_row = dbc.Row([
        dbc.Col(html.Div(dcc.Graph(figure=fig_fi), className="glass-card"), md=12, lg=6),
        dbc.Col(html.Div(interaction_panel, className="glass-card"), md=12, lg=6)
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
    
    logger.debug("Render DeepDive Mode=%s Feature Count=%d", selected_mode, len(dropdown_features))
    
    # Feature Selector
    selector_card = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.RadioItems(
                    id={'type': 'deepdive-mode', 'index': 0},
                    options=[
                        {'label': 'Top 5 Features', 'value': 'top5'},
                        {'label': 'Select Feature', 'value': 'select'}
                    ],
                    value=selected_mode or 'top5',
                    className="btn-group",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-info",
                    labelCheckedClassName="active",
                )
            ], md=4),
            dbc.Col([
                html.Div([
                    dbc.Select(
                        id={'type': 'deepdive-feat-select', 'index': 0},
                        options=[{'label': f, 'value': f} for f in dropdown_features],
                        value=selected_feature or (dropdown_features[0] if dropdown_features else None),
                        placeholder="Select feature...",
                        # disabled handled by callback, but visibility is better
                        className="mb-0"
                    )
                ], id={'type': 'feature-select-container', 'index': 0}, style={'display': 'none' if selected_mode == 'top5' else 'block'})
            ], md=8)
        ], className="align-items-center g-3")
    ], className="mb-2")

    content_area = []
    features_to_show = []
    feature_details = result.get('feature_details', {})
    
    if selected_mode == 'top5':
         if fi_data:
            df_fi_norm = ResultExporter.normalize_feature_importances(fi_data)
            if not df_fi_norm.empty:
                features_to_show = df_fi_norm.nlargest(5, "Delta_Score")["Feature"].tolist()
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
        bin_labels = detail.get('bin_display_labels') or detail.get('bin_labels', [])
        bin_sort_keys = detail.get('bin_sort_keys', [])
        bin_edges = detail.get('bin_edges', [])
        if bin_counts and not bin_labels and bin_edges and len(bin_edges) == len(bin_counts) + 1:
            bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(bin_counts))]
        elif bin_counts and not bin_labels:
             bin_labels = [f"Bin {i}" for i in range(len(bin_counts))]

        order_idx = _sorted_indices_by_keys(len(bin_labels), bin_sort_keys)
        if order_idx:
            bin_labels = [bin_labels[i] for i in order_idx]
            if bin_counts:
                bin_counts = [bin_counts[i] for i in order_idx]
            if bin_means:
                bin_means = [bin_means[i] for i in order_idx]
        
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
                line=dict(color=NEON_CYAN, width=3)
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
            ], className="g-3")
        ], className="feature-detail-panel mb-2"))

    # Interaction Detail
    interaction_area = []
    interaction_details = result.get('interaction_details', {})
    if interaction_details:
        int_keys = list(interaction_details.keys())
        current_pair = selected_interaction_pair or int_keys[0]
        
        interaction_area.append(dbc.Select(
            id={'type': 'deepdive-interaction-select', 'index': 0},
            options=[{'label': k, 'value': k} for k in int_keys],
            value=current_pair,
            className="mb-3"
        ))
        
        det = interaction_details.get(current_pair)
        if det:
            feature_1 = det.get('feature_1', 'Feature 1')
            feature_2 = det.get('feature_2', 'Feature 2')
            metric_name = str(det.get("metric_name") or "Target Mean")
            row_labels = det.get('bin_display_labels_1') or det.get('bin_labels_1', [])
            col_labels = det.get('bin_display_labels_2') or det.get('bin_labels_2', [])
            row_sort_keys = det.get('bin_sort_keys_1') or []
            col_sort_keys = det.get('bin_sort_keys_2') or []
            row_idx = _sorted_indices_by_keys(len(row_labels), row_sort_keys)
            col_idx = _sorted_indices_by_keys(len(col_labels), col_sort_keys)
            means_matrix = _reorder_2d(det.get('means'), row_idx, col_idx)
            counts_matrix = _reorder_2d(det.get('counts'), row_idx, col_idx)
            dominant_labels = det.get("dominant_labels")
            dominant_matrix = _reorder_2d(dominant_labels, row_idx, col_idx) if dominant_labels else None
            row_labels = [row_labels[i] for i in row_idx]
            col_labels = [col_labels[i] for i in col_idx]
            fig_int = go.Figure(data=go.Heatmap(
                z=means_matrix, x=col_labels, y=row_labels,
                colorscale=UNIFIED_HEATMAP_COLORSCALE, colorbar=dict(title=metric_name)
            ))
            apply_chart_style(fig_int)
            fig_int.update_layout(
                title=f"{feature_1} vs {feature_2}",
                xaxis_title=feature_2,
                yaxis_title=feature_1,
                xaxis=dict(
                    tickfont=dict(size=11),
                    title_font=dict(size=12, color="#cfd8dc"),
                    title_standoff=10
                ),
                yaxis=dict(
                    tickfont=dict(size=11),
                    title_font=dict(size=12, color="#cfd8dc"),
                    title_standoff=10
                ),
                margin=dict(l=70, r=28, t=56, b=60),
            )
            interaction_area.append(html.Div(dcc.Graph(figure=fig_int), className="glass-card"))

            # Interaction Tables
            if counts_matrix is None:
                counts_matrix = [[pd.NA for _ in col_labels] for _ in row_labels]
            df_counts = pd.DataFrame(counts_matrix, index=row_labels, columns=col_labels)
            df_means = pd.DataFrame(means_matrix, index=row_labels, columns=col_labels)
            
            # Format means
            df_means = df_means.map(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

            # Reset index for display
            df_counts_disp = df_counts.reset_index().rename(columns={'index': det['feature_1']})
            df_means_disp = df_means.reset_index().rename(columns={'index': det['feature_1']})
            table_cols = [
                dbc.Col([
                    html.H6("Sample Count Matrix", className="text-secondary mt-3"),
                    dbc.Table.from_dataframe(df_counts_disp, striped=False, bordered=True, hover=True, className="table-glass small")
                ], md=6),
                dbc.Col([
                    html.H6(f"{metric_name} Matrix", className="text-secondary mt-3"),
                    dbc.Table.from_dataframe(df_means_disp, striped=False, bordered=True, hover=True, className="table-glass small")
                ], md=6)
            ]

            if dominant_matrix:
                df_dominant = pd.DataFrame(dominant_matrix, index=row_labels, columns=col_labels)
                df_dominant_disp = df_dominant.reset_index().rename(columns={'index': det['feature_1']})
                table_cols.append(
                    dbc.Col([
                        html.H6("Dominant Class Matrix", className="text-secondary mt-3"),
                        dbc.Table.from_dataframe(df_dominant_disp, striped=False, bordered=True, hover=True, className="table-glass small")
                    ], md=12)
                )

            interaction_area.append(dbc.Row(table_cols))



    feature_section = html.Div([
        html.Div(
            [
                html.H4("Feature Analysis", className="section-title mb-3"),
                selector_card,
                html.Div(
                    content_area if content_area else [dbc.Alert("No feature detail data available.", color="secondary", className="mb-0")]
                ),
            ],
            className="glass-card section-card p-3 mb-3"
        ),
    ])

    interaction_section = html.Div([
        html.Div(
            [
                html.H4("Bivariate Interaction Analysis", className="section-title mb-3"),
                *(interaction_area if interaction_area else [dbc.Alert("No interaction detail data available.", color="secondary", className="mb-0")]),
            ],
            className="glass-card section-card p-3 mb-3"
        ),
    ])

    return html.Div([feature_section, interaction_section])

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
            dbc.Select(
                id='target-col', 
                options=[{'label': c, 'value': c} for c in col_names], 
                value=col_names[0] if col_names else None, 
                className="mb-3"
            ),
            
            html.Label("Task Type", className="small text-muted"),
                dbc.Select(
                    id='task-type',
                    options=[
                        {'label': 'Auto', 'value': 'auto'},
                        {'label': 'Classification', 'value': 'classification'},
                        {'label': 'Regression', 'value': 'regression'}
                    ],
                    value='auto',
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
        ], className="sidebar-settings-panel mt-2")
        
        return dbc.Alert(f"Loaded: {filename}", color="dark", className="upload-status-alert py-2 small"), meta_dict, settings
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
    logger.debug("Submit Job clicked target=%s task=%s", target_col, task_type)
    try:
        # Default params
        if not target_col or target_col == "":
             return dash.no_update, dash.no_update, dash.no_update, dbc.Alert("Please select a Target Variable", color="warning", className="glass-card")
             
        col_names = [c['name'] for c in meta['columns']]
        candidates = [c for c in col_names if c != target_col]
        
        # Ensure task_type is valid
        eff_task = task_type
        if not eff_task or eff_task == "":
            eff_task = "auto"
            
        params = AnalysisParams(
            target_col=target_col, 
            candidate_features=candidates, 
            task_type=eff_task,
            max_bins=max_bins,
            top_k=top_k,
            use_aicc=use_aicc
        )
        
        job_id = client.submit_job(meta['dataset_id'], params)
        logger.debug("Job Submitted: %s", job_id)
        msg = dbc.Alert([
            dbc.Spinner(size="sm", spinner_class_name="me-2"),
            f"Analysis started... (Job: {job_id[:6]})"
        ], color="info", className="glass-card border-info")
        
        # Save params
        params_dict = {'target_col': target_col, 'task_type': task_type}
        return job_id, params_dict, False, msg
    except Exception as e:
        logger.exception("Job submission failed")
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
            return info.get('result'), True, dbc.Alert('Analysis Completed Successfully', color='success', className='glass-card')
        elif status in ['failed', 'FAILURE']:
             return None, True, dbc.Alert(f"Job Failed: {info.get('error')}", color="danger", className="glass-card")
        else:
            stage = info.get('progress', {}).get('stage', 'Processing...')
            return dash.no_update, False, dbc.Alert([dbc.Spinner(size="sm", spinner_class_name="me-2"), stage], color="info", className="glass-card border-info")
    except Exception as e:
         return None, True, dbc.Alert(f"polling Error: {e}", color="danger")

# @callback(
#     Output('sidebar-export-area', 'children'),
#     Input('store-analysis-result', 'data'),
#     prevent_initial_call=True
# )
# def show_export_button(result):
#     if not result: return ""
#     return html.Div([
#         dbc.Button([
#             html.I(className="bi bi-filetype-html me-2"),
#             "Export Interactive Report"
#         ], id='btn-export-html', color="info", className="w-100 neon-button")
#     ])



@callback(
    Output("download-html-report", "data"),
    Output("global-status-message", "children", allow_duplicate=True),
    Input("btn-export-html", "n_clicks"),
    State("store-analysis-result", "data"),
    State("store-job-id", "data"),
    State("store-dataset-meta", "data"),
    State("report-filename-input", "value"),
    State("store-analysis-params", "data"),
    State("theme-store", "data"),
    prevent_initial_call=True
)
def download_html_report(n_clicks, result, job_id, meta, custom_filename, analysis_params, theme):
    if not n_clicks or not result:
        return dash.no_update, dash.no_update
    
    # Generate filename
    logger.debug("Export requested filename=%s theme=%s", custom_filename, theme)
    if custom_filename and custom_filename.strip():
        filename = custom_filename.strip()
        if not filename.lower().endswith(".html"):
            filename += ".html"
    else:
        # Default logic
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        original_name = "AdvancedCATDAP"
        target_variable = None
        task_type = None
        if meta:
            if 'filename' in meta and meta['filename']:
                 base = os.path.basename(meta['filename'])
                 original_name, _ = os.path.splitext(base)
            elif 'dataset_id' in meta:
                base = os.path.basename(meta['dataset_id'])
                original_name, _ = os.path.splitext(base)
        if isinstance(analysis_params, dict):
            target_variable = analysis_params.get("target_col")
            task_type = analysis_params.get("task_type")
        def _safe_name_part(value, default="NA"):
            text = str(value or "").strip()
            if not text:
                return default
            text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-_.")
            return text or default
        safe_target = _safe_name_part(target_variable, "UnknownTarget")
        safe_task = _safe_name_part(task_type, "UnknownTask")
        filename = f"{original_name}_Target-{safe_target}_Task-{safe_task}_Report_{timestamp}.html"
    
    try:
        report_result = dict(result) if isinstance(result, dict) else result
        requested_task_type = None
        if isinstance(analysis_params, dict):
            requested_task_type = analysis_params.get("task_type")

        # Export should use the freshest backend payload to avoid stale/partial store state.
        if job_id:
            try:
                job_info = client.get_job_status(job_id)
                latest = job_info.get("result") if isinstance(job_info, dict) else None
                if isinstance(latest, dict) and latest:
                    report_result = dict(latest)
                if isinstance(job_info, dict):
                    params_from_job = job_info.get("params")
                    if isinstance(params_from_job, dict):
                        if params_from_job.get("task_type"):
                            requested_task_type = params_from_job.get("task_type")
                        if "Target-UnknownTarget" in filename and params_from_job.get("target_col"):
                            filename = filename.replace(
                                "Target-UnknownTarget",
                                f"Target-{re.sub(r'[^A-Za-z0-9._-]+', '-', str(params_from_job.get('target_col'))).strip('-_.') or 'UnknownTarget'}"
                            )
                        if "Task-UnknownTask" in filename and params_from_job.get("task_type"):
                            filename = filename.replace(
                                "Task-UnknownTask",
                                f"Task-{re.sub(r'[^A-Za-z0-9._-]+', '-', str(params_from_job.get('task_type'))).strip('-_.') or 'UnknownTask'}"
                            )
            except Exception:
                logger.warning("Failed to refresh job result for export", exc_info=True)

        if isinstance(report_result, dict) and requested_task_type:
            report_result["requested_task_type"] = requested_task_type

        desktop_mode = str(os.environ.get("CATDAP_DESKTOP_MODE", "")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        if desktop_mode:
            resp = client.export_html_report(
                result=report_result,
                meta=meta,
                filename=filename,
                theme=theme or "dark",
            )
            if resp.get("saved"):
                saved_path = resp.get("path", "")
                message = dbc.Alert(
                    f"HTML report saved: {saved_path}",
                    color="success",
                    className="glass-card",
                )
                return dash.no_update, message
            reason = resp.get("reason") or "cancelled"
            if reason == "cancelled":
                message = dbc.Alert(
                    "Export cancelled.",
                    color="secondary",
                    className="glass-card",
                )
                return dash.no_update, message
            message = dbc.Alert(
                f"Export not saved: {reason}",
                color="warning",
                className="glass-card",
            )
            return dash.no_update, message

        html_io = ResultExporter.generate_html_report(report_result, meta, theme=theme)
        return dcc.send_bytes(html_io.getvalue(), filename), dash.no_update
    except Exception:
        logger.exception("Export failed")
        # In a real app, we might want to show a notification to the user
        return dash.no_update, dbc.Alert("Export failed.", color="danger", className="glass-card")

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
            return render_dashboard_tab(result, meta, params)
        elif active_tab == 'tab-deepdive':
            mode = deepdive_state.get('mode', 'top5') if deepdive_state else 'top5'
            feat = deepdive_state.get('feature') if deepdive_state else None
            interaction = deepdive_state.get('interaction') if deepdive_state else None
            target_col = params.get('target_col') if params else None
            return render_deepdive_tab(result, mode, feat, theme, meta, target_col, interaction)
        return html.Div("Tab Not Found")
    except Exception as e:
        logger.exception("Render error")
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
    logger.debug("Update DeepDive State Trigger=%s ModeVals=%s", ctx_id, mode_vals)
    
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


