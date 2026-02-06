import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from advanced_catdap.frontend.api_client import APIClient
from advanced_catdap.service.schema import AnalysisParams
import os

# ============================================================
# Page Config (Wide Layout)
# ============================================================
st.set_page_config(
    page_title="AdvancedCATDAP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for KPI Cards (Dark/Light Mode Compatible)
# ============================================================
st.markdown("""
<style>
/* ============================================
   Theme-Aware CSS Variables
   ============================================ */

/* Light Mode (Default) */
:root {
    --bg-card: #f8f9fa;
    --border-card: #e0e0e0;
    --text-primary: #1a1a2e;
    --text-secondary: #666666;
    --bg-sidebar: #fafafa;
    --shadow-card: rgba(0, 0, 0, 0.05);
}

/* Dark Mode Detection via Streamlit's theme */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-card: #262730;
        --border-card: #3d3d4d;
        --text-primary: #fafafa;
        --text-secondary: #b0b0b0;
        --bg-sidebar: #1e1e28;
        --shadow-card: rgba(0, 0, 0, 0.3);
    }
}

/* Streamlit Dark Theme Detection (Class-based) */
[data-theme="dark"], 
.stApp[data-theme="dark"],
html[data-theme="dark"] {
    --bg-card: #262730;
    --border-card: #3d3d4d;
    --text-primary: #fafafa;
    --text-secondary: #b0b0b0;
    --bg-sidebar: #1e1e28;
    --shadow-card: rgba(0, 0, 0, 0.3);
}

/* ============================================
   KPI Card Styling
   ============================================ */
div[data-testid="stMetric"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px var(--shadow-card);
}

div[data-testid="stMetric"] > div {
    padding: 0;
}

div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
    font-weight: 500;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary) !important;
}

/* ============================================
   Tab Styling
   ============================================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    font-weight: 500;
}

/* ============================================
   Header Styling
   ============================================ */
.main-header {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0;
}

.sub-header {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0;
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

# ============================================================
# Sidebar - Configuration
# ============================================================
with st.sidebar:
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
tab_dashboard, tab_deepdive, tab_simulator = st.tabs([
    "📈 Dashboard",
    "🔍 Deep Dive", 
    "⚡ Simulator"
])

# ============================================================
# Tab 1: Dashboard
# ============================================================
with tab_dashboard:
    if not st.session_state.analysis_result:
        st.info("🚀 Click **Run Analysis** in the sidebar to see results.")
    else:
        res = st.session_state.analysis_result
        fi_data = res.get("feature_importances", [])
        
        if fi_data:
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
            
            if "Delta_Score" in df_fi.columns and "Feature" in df_fi.columns:
                chart_df = df_fi.sort_values("Delta_Score", ascending=True).head(15)
                
                fig_bar = px.bar(
                    chart_df,
                    x="Delta_Score",
                    y="Feature",
                    orientation='h',
                    color="Delta_Score",
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                fig_bar.update_layout(
                    font_family="Inter, sans-serif",
                    height=400,
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=20, t=20, b=0)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Missing required columns for visualization.")
        
        # Interaction Heatmap
        ii_data = res.get("interaction_importances", [])
        if ii_data:
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
                
                fig_heat = px.density_heatmap(
                    df_ii,
                    x="Feature_1",
                    y="Feature_2",
                    z="Gain",
                    nbinsx=len(features),
                    nbinsy=len(features),
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                fig_heat.update_layout(
                    font_family="Inter, sans-serif",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================
# Tab 2: Deep Dive
# ============================================================
with tab_deepdive:
    if not st.session_state.analysis_result:
        st.info("🚀 Click **Run Analysis** in the sidebar to see results.")
    else:
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
            # Build available features list
            avail_feats = sorted(list(res["feature_details"].keys())) if res.get("feature_details") else []
            
            # Use session state for view mode tracking
            if "deepdive_mode" not in st.session_state:
                st.session_state.deepdive_mode = "top5"
            
            # Simple buttons for mode selection (more stable than radio)
            st.write("**View Mode:**")
            col_btn1, col_btn2, col_feat = st.columns([0.15, 0.15, 0.7])
            
            with col_btn1:
                if st.button("Top 5 Drivers", key="btn_top5", 
                             type="primary" if st.session_state.deepdive_mode == "top5" else "secondary"):
                    st.session_state.deepdive_mode = "top5"
            
            with col_btn2:
                if st.button("Select Feature", key="btn_select",
                             type="primary" if st.session_state.deepdive_mode == "select" else "secondary"):
                    st.session_state.deepdive_mode = "select"
            
            with col_feat:
                if st.session_state.deepdive_mode == "select" and avail_feats:
                    selected_feat = st.selectbox(
                        "Feature",
                        avail_feats,
                        key="deepdive_feature_select",
                        label_visibility="collapsed"
                    )
                else:
                    selected_feat = None
            
            # Determine features to show
            if st.session_state.deepdive_mode == "top5":
                if "Delta_Score" in df_fi.columns and "Feature" in df_fi.columns:
                    features_to_show = df_fi.sort_values("Delta_Score", ascending=False).head(5)["Feature"].tolist()
                else:
                    features_to_show = avail_feats[:5] if avail_feats else []
            elif avail_feats and selected_feat:
                features_to_show = [selected_feat]
            else:
                features_to_show = []
            
            # Display feature details with simplified charts
            for feat in features_to_show:
                detail = res["feature_details"].get(feat)
                if not detail:
                    continue
                
                with st.expander(f"📊 {feat}", expanded=True):
                    if detail.get("bin_counts") and detail.get("bin_means"):
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
                        
                        # Two-column layout
                        c_plot, c_table = st.columns([0.6, 0.4])
                        
                        with c_plot:
                            # Use Plotly Express for simpler rendering
                            fig_dual = go.Figure()
                            fig_dual.add_trace(go.Bar(
                                x=df_plot["Value Range"],
                                y=df_plot["Sample Count"],
                                name="Sample Count",
                                marker_color="#E0E0E0"
                            ))
                            fig_dual.add_trace(go.Scatter(
                                x=df_plot["Value Range"],
                                y=df_plot[target_label],
                                name=target_label,
                                yaxis="y2",
                                mode="lines+markers",
                                line=dict(color="#FF4B4B", width=3)
                            ))
                            fig_dual.update_layout(
                                title=f"<b>{feat}</b> Impact",
                                title_font_family="Inter",
                                hovermode="x unified",
                                yaxis=dict(title="Sample Count"),
                                yaxis2=dict(title=target_label, overlaying="y", side="right", showgrid=False),
                                legend=dict(orientation="h", y=1.1),
                                template="plotly_white",
                                height=350,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            st.plotly_chart(fig_dual, use_container_width=True)
                        
                        with c_table:
                            st.caption("Statistics")
                            fmt = "{:.2f}" if is_regression else "{:.2%}"
                            st.dataframe(
                                df_plot.style.format({target_label: fmt}),
                                height=300,
                                use_container_width=True
                            )
        
        # Interaction Details
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
                
                fig_int_heat = go.Figure(data=go.Heatmap(
                    z=i_det['means'],
                    x=i_det['bin_labels_2'],
                    y=i_det['bin_labels_1'],
                    colorscale='Blues',
                    colorbar=dict(title=metric_label)
                ))
                fig_int_heat.update_layout(
                    title=f"{metric_label} by {i_det['feature_1']} vs {i_det['feature_2']}",
                    xaxis_title=i_det['feature_2'],
                    yaxis_title=i_det['feature_1'],
                    height=400
                )
                st.plotly_chart(fig_int_heat, use_container_width=True)
                
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
