import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from advanced_catdap.frontend.api_client import APIClient
from advanced_catdap.service.schema import AnalysisParams

# Page Config
st.set_page_config(page_title="AdvancedCATDAP", layout="wide")

# Initialize Client
# In a real app, URL might come from env var
client = APIClient(base_url="http://localhost:8000")

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
with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Dataset (CSV/Parquet)", type=["csv", "parquet"])
    
    if uploaded_file is not None:
        if st.button("Register Dataset"):
            with st.spinner("Uploading..."):
                try:
                    meta = client.upload_dataset(uploaded_file, uploaded_file.name)
                    st.session_state.dataset_id = meta.dataset_id
                    st.session_state.dataset_meta = meta
                    st.success(f"Registered: {meta.dataset_id}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.dataset_meta:
        st.info(f"Active ID: {st.session_state.dataset_id}")
        st.write(f"Rows: {st.session_state.dataset_meta.n_rows}")
        st.write(f"Cols: {st.session_state.dataset_meta.n_columns}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Data Preview", "Configuration", "Results"])

with tab1:
    if st.session_state.dataset_id:
        try:
            preview_data = client.get_preview(st.session_state.dataset_id)
            st.dataframe(pd.DataFrame(preview_data))
        except Exception as e:
            st.error(f"Could not load preview: {e}")
    else:
        st.info("Upload a dataset to see preview.")

with tab2:
    if st.session_state.dataset_meta:
        col_names = [c.name for c in st.session_state.dataset_meta.columns]
        
        with st.form("analysis_config"):
            target_col = st.selectbox("Target Column", col_names)
            candidates = st.multiselect("Candidate Columns (Empty=All)", col_names, default=[])
            
            c1, c2 = st.columns(2)
            max_bins = c1.number_input("Max Bins", min_value=2, max_value=20, value=5)
            top_k = c2.number_input("Top K Features", min_value=1, max_value=50, value=10)
            
            submitted = st.form_submit_button("Run Analysis")
            
            if submitted:
                params = AnalysisParams(
                    target_col=target_col,
                    candidates=candidates if candidates else None,
                    max_bins=max_bins,
                    top_k=top_k
                )
                try:
                    job_id = client.submit_job(st.session_state.dataset_id, params)
                    st.session_state.job_id = job_id
                    st.session_state.analysis_result = None # Reset previous result
                    st.success(f"Job submitted: {job_id}")
                except Exception as e:
                    st.error(f"Submission failed: {e}")
    else:
        st.info("Please upload a dataset first.")

with tab3:
    if st.session_state.job_id:
        st.write(f"Job ID: {st.session_state.job_id}")
        
        # Polling/Refresh
        status_container = st.empty()
        
        if st.session_state.analysis_result is None:
            # Need to poll
            if st.button("Refresh Status"):
                info = client.get_job_status(st.session_state.job_id)
                status = info.get("status")
                
                if status == "SUCCESS":
                    st.session_state.analysis_result = info.get("result")
                    st.success("Analysis Complete!")
                    st.rerun() # Rerun to show results immediately
                elif status == "FAILURE":
                    st.error(f"Job Failed: {info.get('error')}")
                elif status == "PROGRESS":
                    progress = info.get("progress", {})
                    st.info(f"Running... Stage: {progress.get('stage')} {progress.get('data')}")
                else:
                    st.info(f"Status: {status}")
        
        # Display Results
        if st.session_state.analysis_result:
            res = st.session_state.analysis_result  # Dict form
            
            st.metric("Baseline Score", f"{res.get('baseline_score', 0):.2f}")
            
            # --- Feature Importance Plot ---
            st.subheader("Feature Importance")
            fi_data = res.get("feature_importances", [])
            if fi_data:
                df_fi = pd.DataFrame(fi_data)
                fig_bar = px.bar(
                    df_fi.sort_values("delta_score", ascending=True).head(st.session_state.dataset_meta.n_columns),
                    x="delta_score",
                    y="feature",
                    orientation='h',
                    title="<b>Feature Importance</b> (Delta AIC)", # Bold title
                    color="delta_score",
                    color_continuous_scale="Viridis", # Premium palette
                    template="plotly_white" # Clean template
                )
                fig_bar.update_layout(
                    font_family="Inter, sans-serif",
                    title_font_size=20
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No feature importances found.")

            # --- Interaction Heatmap ---
            st.subheader("Top Interactions")
            ii_data = res.get("interaction_importances", [])
            if ii_data:
                df_ii = pd.DataFrame(ii_data)
                # Create matrix for heatmap
                features = list(set(df_ii["feature_1"]).union(set(df_ii["feature_2"])))
                if features:
                    # Plotly Heatmap
                    # Construct pivot
                    pivot = df_ii.pivot(index="feature_1", columns="feature_2", values="gain")
                    
                    st.write(df_ii.head(20))
                    
                    # Basic heatmap of raw gain values
                    fig_heat = px.density_heatmap(
                        df_ii, 
                        x="feature_1", 
                        y="feature_2", 
                        z="gain", 
                        title="<b>Interaction Gain Heatmap</b>",
                        nbinsx=len(features),
                        nbinsy=len(features),
                        color_continuous_scale="Magma", # Distinct from bar chart
                        template="plotly_white"
                    )
                    fig_heat.update_layout(
                        font_family="Inter, sans-serif",
                        title_font_size=20
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

            else:
                st.write("No significant interactions found.")

    else:
        st.info("No job running.")
