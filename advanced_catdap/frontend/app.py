import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from advanced_catdap.frontend.api_client import APIClient
from advanced_catdap.service.schema import AnalysisParams

# Page Config
st.set_page_config(page_title="AdvancedCATDAP", layout="wide")

import os
# Initialize Client
# In a real app, URL might come from env var
client = APIClient(base_url=os.environ.get("API_URL", "http://localhost:8000"))

st.title("AdvancedCATDAP: Exploratory Analysis")

# Session State
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "dataset_meta" not in st.session_state:
    st.session_state.dataset_meta = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Sidebar
# Sidebar
with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Dataset (CSV/Parquet)", type=["csv", "parquet"])
    
    # Auto-register logic
    if uploaded_file is not None:
        # Check if we need to register (new file or first time)
        # We use a crude check by filename to avoid re-uploading the same file object repeatedly on reruns
        # if the session state is already set for this file.
        current_file_name = uploaded_file.name
        
        # If no dataset registered OR registered dataset name differs from current
        if (st.session_state.dataset_meta is None) or \
           (st.session_state.get("last_uploaded_name") != current_file_name):
            
            with st.spinner(f"Auto-registering {current_file_name}..."):
                try:
                    meta = client.upload_dataset(uploaded_file, current_file_name)
                    st.session_state.dataset_id = meta.dataset_id
                    st.session_state.dataset_meta = meta
                    st.session_state.last_uploaded_name = current_file_name
                    # Reset analysis state on new data
                    st.session_state.job_id = None
                    st.session_state.analysis_result = None
                    st.success(f"Loaded: {current_file_name}")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    if st.session_state.dataset_meta:
        st.divider()
        st.caption(f"**ID**: {st.session_state.dataset_id}")
        st.caption(f"**Rows**: {st.session_state.dataset_meta.n_rows:,}")
        st.caption(f"**Cols**: {st.session_state.dataset_meta.n_columns}")

# Main Layout
# Use tabs for Data Preview vs Analysis (Config + Results)
tab1, tab2 = st.tabs(["Analysis & Results", "Data Preview"])

with tab2:
    if st.session_state.dataset_id:
        try:
            preview_data = client.get_preview(st.session_state.dataset_id)
            # Deprecation fix: use_container_width -> width='stretch'
            st.dataframe(pd.DataFrame(preview_data), width="stretch") # Replaced use_container_width=True
        except Exception as e:
            st.warning(f"Could not load preview: {e}")
    else:
        st.info("Upload a dataset to see preview.")

with tab1:
    if st.session_state.dataset_meta:
        col_names = [c.name for c in st.session_state.dataset_meta.columns]
        
        st.subheader("Configuration")
        
        # Smart defaults
        target_col = st.selectbox("Target Column (Objective Variable)", col_names)
        
        # Auto-exclude target from candidates
        default_candidates = [c for c in col_names if c != target_col]
        
        # We use a multiselect but default to everything else
        # We use a dynamic key to reset selection when target changes
        
        # --- Candidate Columns (Improved UI) ---
        st.write("Candidate Columns (Features)")
        
        # Prepare data for editor
        candidate_rows = []
        for col in st.session_state.dataset_meta.columns:
            if col.name == target_col:
                continue
            candidate_rows.append({
                "Select": True, # Default selected
                "Column": col.name,
                "Type": col.dtype,
                "Missing": getattr(col, "missing_count", 0),
                "Unique": getattr(col, "unique_approx", 0)
            })
        
        df_candidates = pd.DataFrame(candidate_rows)
        
        # Add helper buttons
        col_sel1, col_sel2 = st.columns([0.2, 0.8])
        if col_sel1.button("Select All"):
            df_candidates["Select"] = True
        if col_sel2.button("Deselect All"):
            df_candidates["Select"] = False
            
        # Render editor
        edited_candidates = st.data_editor(
            df_candidates,
            column_config={
                "Select": st.column_config.CheckboxColumn("Include?", help="Check to include this feature in analysis"),
                "Column": st.column_config.TextColumn("Feature Name", disabled=True),
                "Type": st.column_config.TextColumn("Dtype", disabled=True),
                "Missing": st.column_config.NumberColumn("Missing Vals", disabled=True),
                "Unique": st.column_config.NumberColumn("Unique Vals", disabled=True),
            },
            disabled=["Column", "Type", "Missing", "Unique"],
            hide_index=True,
            key=f"editor_{target_col}", # Reset on target change
            # use_container_width=True, # Deprecated
            height=300
        )
        
        # Extract selected
        selected_features = edited_candidates[edited_candidates["Select"]]["Column"].tolist()
        candidates = selected_features
        
        c1, c2 = st.columns(2)
        max_bins = c1.number_input(
            "Discretization Bins (Max)", 
            min_value=2, max_value=20, value=5,
            help="Maximum number of bins to use when discretizing continuous variables."
        )
        top_k = c2.number_input(
            "Max Results to Display", 
            min_value=1, max_value=50, value=10,
            help="Number of top features/interactions to show in the results."
        )
        
        submitted = st.button("Run Analysis", type="primary")
            
        if submitted:
            # If user cleared candidates, handle it (though API might handle None as All, explicit is safer)
            final_candidates = candidates if candidates else None
            
            params = AnalysisParams(
                target_col=target_col,
                candidates=final_candidates,
                max_bins=max_bins,
                top_k=top_k
            )
            try:
                job_id = client.submit_job(st.session_state.dataset_id, params)
                st.session_state.job_id = job_id
                st.session_state.analysis_result = None # Reset previous result
                st.success(f"Analysis started! Job ID: {job_id}")
            except Exception as e:
                st.error(f"Submission failed: {e}")
    else:
        st.info("👈 Please upload a dataset in the sidebar to start.")

    # Results Section (Inline)
    if st.session_state.job_id:
        st.divider()
        st.subheader("Analysis Results")
        st.write(f"Job ID: {st.session_state.job_id}")
        
        # Polling/Refresh
        status_container = st.empty()
        
        if st.session_state.analysis_result is None:
            # Auto-poll
            with st.spinner("Analysis in progress..."):
                info = client.get_job_status(st.session_state.job_id)
                status = info.get("status")
                
                if status == "SUCCESS":
                    st.session_state.analysis_result = info.get("result")
                    st.success("Analysis Complete!")
                    st.rerun()
                elif status == "FAILURE":
                    st.error(f"Job Failed: {info.get('error')}")
                else:
                    # Still running
                    progress = info.get("progress") or {}
                    st.info(f"Processing... {progress.get('stage', '')}")
                    time.sleep(2)
                    st.rerun()
        
        # Display Results
        if st.session_state.analysis_result:
            res = st.session_state.analysis_result  # Dict form
            
            # 1. Feature Importance Plot (Top Priority)
            st.subheader("Feature Importance")
            fi_data = res.get("feature_importances", [])
            
            # Need to define df_fi for later use in details
            df_fi = pd.DataFrame() 
            
            if fi_data:
                df_fi = pd.DataFrame(fi_data)
                
                # Handle possible key casing (API vs Mock compatibility)
                if "feature" in df_fi.columns and "Feature" not in df_fi.columns:
                    df_fi.rename(columns={"feature": "Feature", "delta_score": "Delta_Score", "score": "Score"}, inplace=True)
                
                # Check required columns
                if "Delta_Score" in df_fi.columns and "Feature" in df_fi.columns:
                    fig_bar = px.bar(
                        df_fi.sort_values("Delta_Score", ascending=True).head(st.session_state.dataset_meta.n_columns),
                        x="Delta_Score",
                        y="Feature",
                        orientation='h',
                        title="<b>Feature Importance</b> (Delta AIC)", 
                        color="Delta_Score",
                        color_continuous_scale="Blues",
                        template="plotly_white"
                    )
                    fig_bar.update_layout(
                        font_family="Inter, sans-serif",
                        title_font_size=20
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.error("Feature importance data missing required columns (Feature, Delta_Score).")
                    st.write(df_fi.head())
            else:
                st.warning("No feature importances found.")

            # 2. Detailed Feature Analysis (Impact)
            st.subheader("Top Driver Analysis (Impact on Target)")
            
            # Use df_fi from above
            if not df_fi.empty and "feature_details" in res and "Delta_Score" in df_fi.columns:
                # Picker for "All Features" vs "Top 5"
                view_mode = st.radio("View Mode", ["Top 5 Drivers", "Select Feature"], horizontal=True)
                
                features_to_show = []
                if view_mode == "Top 5 Drivers":
                    feat_col = "Feature"
                    score_col = "Delta_Score"
                    features_to_show = df_fi.sort_values(score_col, ascending=False).head(5)[feat_col].tolist()
                else:
                    # Select from all features that have details
                    avail_feats = sorted(list(res["feature_details"].keys()))
                    selected_feat = st.selectbox("Select Feature to Inspect", avail_feats)
                    if selected_feat:
                        features_to_show = [selected_feat]

                for feat in features_to_show:
                    detail = res["feature_details"].get(feat)
                    if not detail:
                        continue
                        
                    with st.expander(f"Details: {feat}", expanded=True):
                        if detail.get("bin_counts") and detail.get("bin_means"):
                            # Prepare Data
                            # Prioritize explicit labels (e.g. from analyzer)
                            if detail.get("bin_labels"):
                                bin_labels = detail["bin_labels"]
                            else:
                                bin_labels = [f"Bin {i}" for i in range(len(detail["bin_counts"]))]
                                edges = detail.get("bin_edges")
                                if edges and len(edges) == len(detail["bin_counts"]) + 1:
                                    bin_labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(detail["bin_counts"]))]
                            
                            # Dynamic Label based on Mode
                            # Handle case sensitivity (backend returns 'regression')
                            mode_val = res.get("mode", "").upper()
                            is_regression = mode_val == "REGRESSION"
                            target_label = "Average Value" if is_regression else "Target Rate"
                            
                            df_plot = pd.DataFrame({
                                "Value Range": bin_labels,
                                "Sample Count": detail["bin_counts"],
                                target_label: detail["bin_means"]
                            })
                            
                            # Layout: Chart LEFT, Table RIGHT
                            c_plot, c_table = st.columns([0.6, 0.4])
                            
                            with c_plot:
                                fig_dual = go.Figure()
                                fig_dual.add_trace(go.Bar(
                                    x=df_plot["Value Range"], y=df_plot["Sample Count"],
                                    name="Sample Count", marker_color="#E0E0E0"
                                ))
                                fig_dual.add_trace(go.Scatter(
                                    x=df_plot["Value Range"], y=df_plot[target_label],
                                    name=target_label, yaxis="y2", mode="lines+markers",
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
                                st.caption("Detailed Statistics")
                                fmt = "{:.2f}" if is_regression else "{:.2%}"
                                st.dataframe(
                                    df_plot.style.format({target_label: fmt}), 
                                    width="stretch", # Replaced use_container_width=True
                                    height=300
                                )

            # 3. Model Baseline (Deprioritized)
            with st.expander("Model Baseline Details", expanded=False):
                 st.metric("Baseline Score (AIC)", f"{res.get('baseline_score', 0):.2f}")

            # --- Interaction Heatmap ---
            st.subheader("Top Interactions")
            ii_data = res.get("interaction_importances", [])
            if ii_data:
                df_ii = pd.DataFrame(ii_data)
                
                # Handle aliases: Feature_1, Feature_2, Gain
                col_map = {
                    "feature_1": "Feature_1", 
                    "feature_2": "Feature_2", 
                    "gain": "Gain"
                }
                df_ii.rename(columns={k:v for k,v in col_map.items() if k in df_ii.columns}, inplace=True)
                
                # Create matrix for heatmap
                if "Feature_1" in df_ii.columns and "Feature_2" in df_ii.columns:
                    features = list(set(df_ii["Feature_1"]).union(set(df_ii["Feature_2"])))
                    if features:
                        # Plotly Heatmap
                        # Construct pivot
                        pivot = df_ii.pivot(index="Feature_1", columns="Feature_2", values="Gain")
                        
                        st.write(df_ii.head(20))
                        
                        # Basic heatmap of raw gain values
                        fig_heat = px.density_heatmap(
                            df_ii, 
                            x="Feature_1", 
                            y="Feature_2", 
                            z="Gain", 
                            title="<b>Interaction Gain Heatmap</b>",
                            nbinsx=len(features),
                            nbinsy=len(features),
                            color_continuous_scale="Blues", # Unified with other plots
                            template="plotly_white"
                        )
                        fig_heat.update_layout(
                            font_family="Inter, sans-serif",
                            title_font_size=20
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)

            # --- Detailed Interaction Plots (New) ---
            if "interaction_details" in res and res["interaction_details"]:
                st.subheader("Interaction Detail Analysis")
                int_details = res["interaction_details"]
                # Create labels for selectbox
                int_keys = list(int_details.keys())
                selected_int_key = st.selectbox("Select Interaction Pair", int_keys)
                
                if selected_int_key:
                    i_det = int_details[selected_int_key]
                    
                    start_mode = res.get("mode", "").upper()
                    is_regression = start_mode == "REGRESSION"
                    metric_label = "Average Value" if is_regression else "Target Rate"
                    
                    st.caption(f"Impact of {i_det['feature_1']} x {i_det['feature_2']} on {metric_label}")
                    
                    # Heatmap of Means
                    # Row: Feature 1 (bin_labels_1), Col: Feature 2 (bin_labels_2), Z: means
                    fig_int_heat = go.Figure(data=go.Heatmap(
                        z=i_det['means'],
                        x=i_det['bin_labels_2'], # Col
                        y=i_det['bin_labels_1'], # Row
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
                    
                    with st.expander("Detailed Interaction Statistics"):
                        # Show raw counts and means?
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("**Counts**")
                            st.write(pd.DataFrame(i_det['counts'], index=i_det['bin_labels_1'], columns=i_det['bin_labels_2']))
                        with c2:
                            st.write(f"**{metric_label}s**")
                            st.write(pd.DataFrame(i_det['means'], index=i_det['bin_labels_1'], columns=i_det['bin_labels_2']))

            else:
                if not ii_data:
                    st.write("No significant interactions found.")

    else:
        st.info("No job running.")
