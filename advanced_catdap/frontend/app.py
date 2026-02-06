import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from advanced_catdap.frontend.api_client import APIClient
from advanced_catdap.service.schema import AnalysisParams
import os
import logging
from datetime import datetime

NEON_CYAN = "#00f3ff"
NEON_MAGENTA = "#bc13fe"
NEON_GREEN = "#0aff99"

def apply_chart_style(fig):
    """Apply standard dark/neon theme to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
        title_font=dict(color=NEON_CYAN),
        margin=dict(l=20, r=20, t=50, b=20)
    )

def plotly_html(fig, height=400):
    """
    Render Plotly figure as HTML component instead of st.plotly_chart.
    This bypasses potential issues with Streamlit's Plotly integration.
    """
    html = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False})
    components.html(html, height=height, scrolling=False)

# Setup debug logging to file
LOG_FILE = os.path.join(os.path.dirname(__file__), "debug.log")
MARKER_FILE = os.path.join(os.path.dirname(__file__), ".log_cleared")

def debug_log(msg):
    """Log debug message to file."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_msg = f"[{timestamp}] {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
            f.flush()  # Force write
    except Exception as e:
        pass  # Silently ignore logging errors

# Clear log file only once per app launch (not on every rerun)
# Use a marker file to track if we already cleared
if not os.path.exists(MARKER_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== App started at {datetime.now()} ===\n")
    with open(MARKER_FILE, "w") as f:
        f.write("cleared")

debug_log("Module loaded - this should appear on every rerun")

# ============================================================
# Page Config (Wide Layout)
# ============================================================
st.set_page_config(
    page_title="AdvancedCATDAP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

debug_log("After page config")

# ============================================================
# Custom CSS for KPI Cards (Dark/Light Mode Compatible)
# ============================================================
st.markdown("""
<style>
/* ============================================
   Neon & Glass Theme (Dash Unification)
   ============================================ */
:root {
    --neon-cyan: #00f3ff;
    --neon-magenta: #bc13fe;
    --neon-green: #0aff99;
    --glass-bg: rgba(20, 20, 30, 0.7);
    --glass-border: rgba(255, 255, 255, 0.1);
    --text-primary: #e0e0e0;
    --text-secondary: #b0b0b0;
}

/* Main App Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--glass-border);
}

/* KPI Cards (Metrics) */
div[data-testid="stMetric"] {
    background-color: var(--glass-bg);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 243, 255, 0.1);
    border-color: var(--neon-cyan);
}

div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--neon-cyan) !important;
    font-family: 'Inter', sans-serif;
    text-shadow: 0 0 10px rgba(0, 243, 255, 0.3);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    color: var(--text-secondary);
    border-radius: 4px;
    transition: all 0.3s;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(188, 19, 254, 0.1);
    color: var(--neon-magenta) !important;
    border-bottom: 2px solid var(--neon-magenta);
    font-weight: bold;
}

/* Headings */
h1, h2, h3, .main-header {
    font-family: 'Inter', sans-serif;
    color: white !important;
}

.sub-header {
    color: var(--text-secondary) !important;
}

/* Buttons */
.stButton button {
    background: transparent;
    border: 1px solid var(--neon-cyan);
    color: var(--neon-cyan);
    border-radius: 4px;
    transition: all 0.3s;
}
.stButton button:hover {
    background: rgba(0, 243, 255, 0.1);
    box-shadow: 0 0 15px rgba(0, 243, 255, 0.4);
    color: white;
    border-color: white;
}
.stButton button:active { 
    background: var(--neon-cyan);
    color: black;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    color: var(--text-primary);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Initialize Client
# ============================================================
client = APIClient(base_url=os.environ.get("API_URL", "http://localhost:8000"))

# ============================================================
# Session State
# ============================================================
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "dataset_meta" not in st.session_state:
    st.session_state.dataset_meta = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

debug_log("After session state init")

# ============================================================
# Sidebar - Configuration
# ============================================================
with st.sidebar:
    debug_log("Starting sidebar")
    st.markdown("## 📊 AdvancedCATDAP")
    st.caption("AIC-based Feature Analysis")
    
    st.divider()
    
    # Data Upload Section
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV/Parquet)",
        type=["csv", "parquet"],
        help="Upload your dataset to begin analysis"
    )
    
    # Auto-register logic
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        
        if (st.session_state.dataset_meta is None) or \
           (st.session_state.get("last_uploaded_name") != current_file_name):
            
            with st.spinner(f"Loading {current_file_name}..."):
                try:
                    meta = client.upload_dataset(uploaded_file, current_file_name)
                    st.session_state.dataset_id = meta.dataset_id
                    st.session_state.dataset_meta = meta
                    st.session_state.last_uploaded_name = current_file_name
                    st.session_state.job_id = None
                    st.session_state.analysis_result = None
                    st.success(f"✅ Loaded: {current_file_name}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Dataset Info
    if st.session_state.dataset_meta:
        st.divider()
        st.markdown("### 📋 Dataset Info")
        info_cols = st.columns(2)
        info_cols[0].metric("Rows", f"{st.session_state.dataset_meta.n_rows:,}")
        info_cols[1].metric("Columns", st.session_state.dataset_meta.n_columns)
        
        st.divider()
        
        # ========================================
        # Basic Settings (Always Visible)
        # ========================================
        st.markdown("### ⚙️ Basic Settings")
        
        col_names = [c.name for c in st.session_state.dataset_meta.columns]
        target_col = st.selectbox(
            "Target Column",
            col_names,
            help="The variable you want to predict/analyze",
            key="sidebar_target_col"
        )
        
        task_type = st.selectbox(
            "Task Type",
            ["auto", "classification", "regression"],
            help="Auto-detect or specify the task type",
            key="sidebar_task_type"
        )
        
        # ========================================
        # Advanced Settings (Collapsed)
        # ========================================
        with st.expander("🔧 Advanced Settings", expanded=False):
            max_bins = st.slider(
                "Max Bins",
                min_value=2,
                max_value=20,
                value=5,
                help="Maximum bins for discretizing continuous variables"
            )
            
            top_k = st.slider(
                "Top K Features",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of top features to select"
            )
            
            use_aicc = st.checkbox(
                "Use AICc (Corrected)",
                value=True,
                help="Use corrected AIC for small samples"
            )
        
        st.divider()
        
        # Run Button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            default_candidates = [c for c in col_names if c != target_col]
            
            params = AnalysisParams(
                target_col=target_col,
                candidates=default_candidates,
                max_bins=max_bins,
                top_k=top_k
            )
            
            try:
                job_id = client.submit_job(st.session_state.dataset_id, params)
                st.session_state.job_id = job_id
                st.session_state.analysis_result = None
                st.success(f"Started: {job_id[:8]}...")
            except Exception as e:
                st.error(f"Failed: {e}")

# ============================================================
# Main Area
# ============================================================

# Header
st.markdown('<p class="main-header">📊 Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore feature importances and interactions</p>', unsafe_allow_html=True)

# ============================================================
# No Data State
# ============================================================
if not st.session_state.dataset_meta:
    st.info("👈 **Upload a dataset** in the sidebar to get started.")
    st.stop()

# ============================================================
# Polling for Results
# ============================================================
if st.session_state.job_id and st.session_state.analysis_result is None:
    with st.spinner("🔄 Analysis in progress..."):
        info = client.get_job_status(st.session_state.job_id)
        status = info.get("status")
        
        if status == "SUCCESS":
            st.session_state.analysis_result = info.get("result")
            st.rerun()
        elif status == "FAILURE":
            st.error(f"❌ Job Failed: {info.get('error')}")
        else:
            progress = info.get("progress") or {}
            st.info(f"Processing... {progress.get('stage', '')}")
            time.sleep(2)
            st.rerun()

# ============================================================
# KPI Area (Top Row) - Only show when results available
# ============================================================
if st.session_state.analysis_result:
    debug_log("In KPI section")
    res = st.session_state.analysis_result
    
    # Calculate KPIs
    baseline_score = res.get("baseline_score", 0)
    fi_data = res.get("feature_importances", [])
    
    # Best model AIC (lowest score among selected features)
    best_model_aic = baseline_score
    if fi_data:
        df_fi = pd.DataFrame(fi_data)
        if "score" in df_fi.columns:
            best_model_aic = df_fi["score"].min()
        elif "Score" in df_fi.columns:
            best_model_aic = df_fi["Score"].min()
    
    delta_aic = best_model_aic - baseline_score
    n_selected = len(fi_data)
    n_total = st.session_state.dataset_meta.n_columns - 1  # Exclude target
    
    # KPI Row
    st.markdown("---")
    kpi_cols = st.columns(3)
    
    with kpi_cols[0]:
        st.metric(
            label="📏 Baseline AIC",
            value=f"{baseline_score:,.1f}",
            help="AIC score of the null model (no features)"
        )
    
    with kpi_cols[1]:
        st.metric(
            label="🎯 Best Model AIC",
            value=f"{best_model_aic:,.1f}",
            delta=f"{delta_aic:,.1f}",
            delta_color="inverse",  # Negative is good for AIC
            help="AIC score of the best feature model (lower is better)"
        )
    
    with kpi_cols[2]:
        st.metric(
            label="✅ Selected Features",
            value=f"{n_selected} / {n_total}",
            help="Number of selected features vs total candidates"
        )
    
    st.markdown("---")

# ============================================================
# Tabs: Dashboard / Deep Dive / Simulator
# ============================================================
debug_log("Before tabs creation")
tab_dashboard, tab_deepdive, tab_simulator = st.tabs([
    "📈 Dashboard",
    "🔍 Deep Dive", 
    "⚡ Simulator"
])
debug_log("After tabs creation")

# ============================================================
# Fragment for Dashboard content (prevents rerun when other tabs change)
# ============================================================
# Dashboard content function (no fragment - full rerun is fine for stability)
def dashboard_content():
    """Dashboard content - renders charts for feature importance and interactions."""
    debug_log("Dashboard content started")
    if not st.session_state.analysis_result:
        st.info("🚀 Click **Run Analysis** in the sidebar to see results.")
        debug_log("Dashboard: No results, showing info message")
    else:
        debug_log("Dashboard: Has results, rendering charts")
        res = st.session_state.analysis_result
        fi_data = res.get("feature_importances", [])
        
        if fi_data:
            debug_log(f"Dashboard: Feature importance data has {len(fi_data)} items")
            df_fi = pd.DataFrame(fi_data)
            
            # Handle key casing
            if "feature" in df_fi.columns and "Feature" not in df_fi.columns:
                df_fi.rename(columns={
                    "feature": "Feature",
                    "delta_score": "Delta_Score",
                    "score": "Score"
                }, inplace=True)
            
            # Feature Importance Chart
            st.subheader("Feature Importance")
            debug_log("Dashboard: About to render feature importance chart")
            
            if "Delta_Score" in df_fi.columns and "Feature" in df_fi.columns:
                chart_df = df_fi.sort_values("Delta_Score", ascending=True).head(15)
                
                fig_bar = px.bar(
                    chart_df,
                    x="Delta_Score",
                    y="Feature",
                    orientation='h',
                    color="Delta_Score",
                    color_continuous_scale="Blues",
                )
                fig_bar.update_layout(
                    font_family="Inter, sans-serif",
                    height=400,
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=20, t=20, b=0)
                )
                apply_chart_style(fig_bar)
                debug_log("Dashboard: Calling plotly_html for bar chart")
                plotly_html(fig_bar, height=420)
                debug_log("Dashboard: Bar chart rendered successfully")
            else:
                st.warning("Missing required columns for visualization.")
        
        # Interaction Heatmap
        ii_data = res.get("interaction_importances", [])
        if ii_data:
            debug_log(f"Dashboard: Interaction data has {len(ii_data)} items")
            st.subheader("Top Interactions")
            
            df_ii = pd.DataFrame(ii_data)
            col_map = {
                "feature_1": "Feature_1",
                "feature_2": "Feature_2",
                "gain": "Gain"
            }
            df_ii.rename(columns={k: v for k, v in col_map.items() if k in df_ii.columns}, inplace=True)
            
            if "Feature_1" in df_ii.columns and "Feature_2" in df_ii.columns and "Gain" in df_ii.columns:
                features = list(set(df_ii["Feature_1"]).union(set(df_ii["Feature_2"])))
                
                debug_log("Dashboard: About to render heatmap")
                fig_heat = px.density_heatmap(
                    df_ii,
                    x="Feature_1",
                    y="Feature_2",
                    z="Gain",
                    nbinsx=len(features),
                    nbinsy=len(features),
                    nbinsy=len(features),
                    color_continuous_scale="Blues",
                )
                fig_heat.update_layout(
                    font_family="Inter, sans-serif",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                apply_chart_style(fig_heat)
                debug_log("Dashboard: Calling plotly_html for heatmap")
                plotly_html(fig_heat, height=420)
                debug_log("Dashboard: Heatmap rendered successfully")
    
    debug_log("Dashboard content completed")

# ============================================================
# Tab 1: Dashboard
# ============================================================
with tab_dashboard:
    debug_log("Dashboard tab started")
    dashboard_content()
    debug_log("Dashboard tab completed")

# ============================================================
# Tab 2: Deep Dive
# ============================================================
# Deep Dive content function (no fragment - full rerun handles DOM cleanup properly)
def deepdive_content():
    """Deep Dive content - detailed feature analysis with interactive charts."""
    debug_log("Deep Dive content started")
    
    if not st.session_state.analysis_result:
        st.info("🚀 Click **Run Analysis** in the sidebar to see results.")
        return
    
    res = st.session_state.analysis_result
    
    st.subheader("Detailed Feature Analysis")
    
    fi_data = res.get("feature_importances", [])
    df_fi = pd.DataFrame(fi_data) if fi_data else pd.DataFrame()
    
    # Rename columns for consistency
    if not df_fi.empty and "feature" in df_fi.columns:
        df_fi.rename(columns={
            "feature": "Feature",
            "delta_score": "Delta_Score",
            "score": "Score",
            "actual_bins": "Actual_Bins",
            "method": "Method"
        }, inplace=True)
    
    if not df_fi.empty and "feature_details" in res:
        debug_log("Deep Dive: Starting feature details section")
        
        # Build available features list
        avail_feats = sorted(list(res["feature_details"].keys())) if res.get("feature_details") else []
        debug_log(f"Deep Dive: Found {len(avail_feats)} features")
        
        # Use session state for view mode tracking
        if "deepdive_mode" not in st.session_state:
            st.session_state.deepdive_mode = "top5"
        
        debug_log(f"Deep Dive: Current mode = {st.session_state.deepdive_mode}")
        
        # Simple radio buttons for mode selection (no callbacks needed without fragments)
        st.write("**View Mode:**")
        col_mode, col_feat = st.columns([0.3, 0.7])
        
        with col_mode:
            mode = st.radio(
                "Select view mode",
                ["Top 5 Drivers", "Select Feature"],
                index=0 if st.session_state.deepdive_mode == "top5" else 1,
                horizontal=True,
                label_visibility="collapsed",
                key="deepdive_mode_radio"
            )
            # Only update session state when value ACTUALLY changes (prevents rerun cascade)
            new_mode = "top5" if mode == "Top 5 Drivers" else "select"
            if st.session_state.deepdive_mode != new_mode:
                st.session_state.deepdive_mode = new_mode
        
        debug_log(f"Deep Dive: After mode selection, mode = {st.session_state.deepdive_mode}")
        
        with col_feat:
            if st.session_state.deepdive_mode == "select" and avail_feats:
                debug_log("Deep Dive: Rendering selectbox")
                selected_feat = st.selectbox(
                    "Feature",
                    avail_feats,
                    key="deepdive_feature_select",
                    label_visibility="collapsed"
                )
                debug_log(f"Deep Dive: Selected feature = {selected_feat}")
            else:
                selected_feat = None
                debug_log("Deep Dive: Selectbox not rendered (mode is top5 or no features)")
        
        # Determine features to show
        debug_log("Deep Dive: Determining features to show")
        if st.session_state.deepdive_mode == "top5":
            if "Delta_Score" in df_fi.columns and "Feature" in df_fi.columns:
                features_to_show = df_fi.sort_values("Delta_Score", ascending=False).head(5)["Feature"].tolist()
            else:
                features_to_show = avail_feats[:5] if avail_feats else []
        elif avail_feats and selected_feat:
            features_to_show = [selected_feat]
        else:
            features_to_show = []
        
        debug_log(f"Deep Dive: Features to show = {features_to_show}")
        
        # Display feature details with simplified charts
        for feat in features_to_show:
            debug_log(f"Deep Dive: Processing feature '{feat}'")
            detail = res["feature_details"].get(feat)
            if not detail:
                debug_log(f"Deep Dive: No detail for '{feat}', skipping")
                continue
            
            debug_log(f"Deep Dive: Creating expander for '{feat}'")
            with st.expander(f"📊 {feat}", expanded=True):
                debug_log(f"Deep Dive: Inside expander for '{feat}'")
                if detail.get("bin_counts") and detail.get("bin_means"):
                    debug_log(f"Deep Dive: Has bin data for '{feat}'")
                    # Prepare labels
                    if detail.get("bin_labels"):
                        bin_labels = detail["bin_labels"]
                    else:
                        bin_labels = [f"Bin {i}" for i in range(len(detail["bin_counts"]))]
                        edges = detail.get("bin_edges")
                        if edges and len(edges) == len(detail["bin_counts"]) + 1:
                            bin_labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(detail["bin_counts"]))]
                    
                    mode_val = res.get("mode", "").upper()
                    is_regression = mode_val == "REGRESSION"
                    target_label = "Average Value" if is_regression else "Target Rate"
                    
                    df_plot = pd.DataFrame({
                        "Value Range": bin_labels,
                        "Sample Count": detail["bin_counts"],
                        target_label: detail["bin_means"]
                    })
                    
                    debug_log(f"Deep Dive: Creating columns for '{feat}'")
                    # Two-column layout
                    c_plot, c_table = st.columns([0.6, 0.4])
                    
                    with c_plot:
                        debug_log(f"Deep Dive: Creating Plotly figure for '{feat}'")
                        # Dual-axis chart: bars for counts, line for target rate
                        fig_dual = go.Figure()
                        fig_dual.add_trace(go.Bar(
                            x=df_plot["Value Range"],
                            y=df_plot["Sample Count"],
                            name="Sample Count",
                            marker_color=NEON_CYAN
                        ))
                        fig_dual.add_trace(go.Scatter(
                            x=df_plot["Value Range"],
                            y=df_plot[target_label],
                            name=target_label,
                            yaxis="y2",
                            mode="lines+markers",
                            line=dict(color=NEON_MAGENTA, width=3)
                        ))
                        fig_dual.update_layout(
                            title=f"<b>{feat}</b> Impact",
                            title_font_family="Inter",
                            hovermode="x unified",
                            yaxis=dict(title="Sample Count"),
                            yaxis2=dict(title=target_label, overlaying="y", side="right", showgrid=False),
                            legend=dict(orientation="h", y=1.1),
                            height=350,
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        apply_chart_style(fig_dual)
                        debug_log(f"Deep Dive: Calling plotly_html for '{feat}'")
                        plotly_html(fig_dual, height=370)
                        debug_log(f"Deep Dive: Chart rendered for '{feat}'")
                    
                    with c_table:
                        st.caption("Statistics")
                        fmt = "{:.2f}" if is_regression else "{:.2%}"
                        st.dataframe(
                            df_plot.style.format({target_label: fmt}),
                            height=300,
                            use_container_width=True
                        )
        
        # Interaction Analysis - always shown (no mode-based filtering)
        st.divider()
        st.subheader("Interaction Analysis")
        
        if "interaction_details" in res and res["interaction_details"]:
            int_details = res["interaction_details"]
            int_keys = list(int_details.keys())
            selected_int_key = st.selectbox("Select Interaction Pair", int_keys, key="deepdive_interaction_select")
            
            if selected_int_key:
                i_det = int_details[selected_int_key]
                
                mode_val = res.get("mode", "").upper()
                is_regression = mode_val == "REGRESSION"
                metric_label = "Average Value" if is_regression else "Target Rate"
                
                st.caption(f"Impact of **{i_det['feature_1']}** × **{i_det['feature_2']}** on {metric_label}")
                
                # Plotly heatmap for interaction visualization
                debug_log("Deep Dive: Creating Plotly heatmap for interaction")
                fig_heat = go.Figure(data=go.Heatmap(
                    z=i_det['means'],
                    x=i_det['bin_labels_2'],
                    y=i_det['bin_labels_1'],
                    colorscale='Blues',
                    colorbar=dict(title=metric_label)
                ))
                fig_heat.update_layout(
                    title=f"{metric_label} by {i_det['feature_1']} vs {i_det['feature_2']}",
                    xaxis_title=i_det['feature_2'],
                    yaxis_title=i_det['feature_1'],
                    height=400
                )
                apply_chart_style(fig_heat)
                debug_log("Deep Dive: Calling plotly_html for interaction heatmap")
                plotly_html(fig_heat, height=420)
                debug_log("Deep Dive: Interaction heatmap rendered")
                
                with st.expander("📋 Detailed Statistics"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Counts**")
                        st.dataframe(pd.DataFrame(
                            i_det['counts'],
                            index=i_det['bin_labels_1'],
                            columns=i_det['bin_labels_2']
                        ))
                    with c2:
                        st.write(f"**{metric_label}s**")
                        st.dataframe(pd.DataFrame(
                            i_det['means'],
                            index=i_det['bin_labels_1'],
                            columns=i_det['bin_labels_2']
                        ))
        else:
            ii_data = res.get("interaction_importances", [])
            if not ii_data:
                st.info("No significant interactions found.")
    
    debug_log("Deep Dive content completed")

# Deep Dive tab
with tab_deepdive:
    debug_log("Deep Dive tab started")
    deepdive_content()
    debug_log("Deep Dive tab completed")

# ============================================================
# Tab 3: Simulator (Coming Soon)
# ============================================================
with tab_simulator:
    st.markdown("### ⚡ What-If Simulator")
    st.info("🚧 **Coming Soon**\n\nThis feature will allow you to simulate predictions by adjusting feature values.")
    
    # Placeholder illustration
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        **Planned Features:**
        - 🎚️ Interactive sliders for each feature
        - 📊 Real-time prediction updates
        - 📈 Sensitivity analysis charts
        - 💾 Export simulation scenarios
        """)

# ============================================================
# Data Preview (Bottom)
# ============================================================
if st.session_state.dataset_id:
    with st.expander("📋 Data Preview", expanded=False):
        try:
            preview_data = client.get_preview(st.session_state.dataset_id)
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load preview: {e}")
