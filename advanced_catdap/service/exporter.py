import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Any
import re

# Ensure template availability
pio.templates.default = "plotly_white" # Use white for report readability or dark if preferred

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

class ResultExporter:
    """Service to export analysis results to various formats."""

    @staticmethod
    def _resolve_column(df: pd.DataFrame, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        lower_map = {}
        for col in df.columns:
            lower_map.setdefault(str(col).lower(), col)
        for col in candidates:
            hit = lower_map.get(str(col).lower())
            if hit is not None:
                return hit
        return None

    @staticmethod
    def _coalesce_columns(df: pd.DataFrame, candidates) -> pd.Series:
        resolved_cols = []
        for cand in candidates:
            if cand in df.columns and cand not in resolved_cols:
                resolved_cols.append(cand)
                continue
            for col in df.columns:
                if str(col).lower() == str(cand).lower() and col not in resolved_cols:
                    resolved_cols.append(col)
                    break
        if not resolved_cols:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        out = df[resolved_cols[0]]
        for col in resolved_cols[1:]:
            out = out.combine_first(df[col])
        return out

    @staticmethod
    def normalize_feature_importances(fi_data) -> pd.DataFrame:
        if not fi_data:
            return pd.DataFrame(columns=["Feature", "Score", "Delta_Score"])
        df = pd.DataFrame(fi_data).copy()
        feat_series = ResultExporter._coalesce_columns(df, ["Feature", "feature"])
        score_series = ResultExporter._coalesce_columns(df, ["Score", "score"])
        delta_series = ResultExporter._coalesce_columns(df, ["Delta_Score", "delta_score", "deltascore"])
        if feat_series.isna().all() or delta_series.isna().all():
            return pd.DataFrame(columns=["Feature", "Score", "Delta_Score"])
        out = pd.DataFrame({
            "Feature": feat_series.astype(str),
            "Delta_Score": pd.to_numeric(delta_series, errors="coerce"),
        })
        out["Score"] = pd.to_numeric(score_series, errors="coerce")
        out = out.dropna(subset=["Delta_Score"])
        return out

    @staticmethod
    def normalize_interaction_importances(ii_data) -> pd.DataFrame:
        if not ii_data:
            return pd.DataFrame(columns=["Feature_1", "Feature_2", "Gain"])
        df = pd.DataFrame(ii_data).copy()
        f1_series = ResultExporter._coalesce_columns(df, ["Feature_1", "feature_1"])
        f2_series = ResultExporter._coalesce_columns(df, ["Feature_2", "feature_2"])
        # Keep Gain as canonical output while accepting legacy/alternate score keys.
        gain_series = ResultExporter._coalesce_columns(df, ["Gain", "gain", "Pair_Score", "pair_score"])
        if f1_series.isna().all() or f2_series.isna().all() or gain_series.isna().all():
            return pd.DataFrame(columns=["Feature_1", "Feature_2", "Gain"])
        out = pd.DataFrame({
            "Feature_1": f1_series.astype(str),
            "Feature_2": f2_series.astype(str),
            "Gain": pd.to_numeric(gain_series, errors="coerce"),
        })
        out = out.dropna(subset=["Gain"])
        return out

    @staticmethod
    def build_interaction_matrix(df_ii_norm: pd.DataFrame) -> pd.DataFrame:
        if df_ii_norm is None or df_ii_norm.empty:
            return pd.DataFrame()
        pivot = df_ii_norm.pivot_table(
            index="Feature_2",
            columns="Feature_1",
            values="Gain",
            aggfunc="sum",
            fill_value=0.0
        )
        if pivot.empty:
            return pivot
        row_order = pivot.max(axis=1).sort_values(ascending=False).index
        col_order = pivot.max(axis=0).sort_values(ascending=False).index
        return pivot.loc[row_order, col_order]

    @staticmethod
    def build_interaction_matrix_from_details(interaction_details: Dict[str, Any]) -> pd.DataFrame:
        if not interaction_details:
            return pd.DataFrame()
        rows = []
        for _, det in interaction_details.items():
            if not isinstance(det, dict):
                continue
            f1 = det.get("feature_1")
            f2 = det.get("feature_2")
            counts = det.get("counts")
            means = det.get("means")
            if not f1 or not f2 or counts is None or means is None:
                continue
            try:
                df_counts = pd.DataFrame(counts)
                df_means = pd.DataFrame(means)
                if df_counts.empty or df_means.empty:
                    continue
                nrows = min(len(df_counts), len(df_means))
                ncols = min(len(df_counts.columns), len(df_means.columns))
                df_counts = df_counts.iloc[:nrows, :ncols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                df_means = df_means.iloc[:nrows, :ncols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                strength = float((df_means.abs() * df_counts).to_numpy().sum())
            except Exception:
                continue
            rows.append({"Feature_1": str(f1), "Feature_2": str(f2), "Gain": strength})
        if not rows:
            return pd.DataFrame()
        return ResultExporter.build_interaction_matrix(pd.DataFrame(rows))

    @staticmethod
    def _format_compact(value: float) -> str:
        try:
            v = float(value)
        except Exception:
            return str(value)
        abs_v = abs(v)
        if abs_v >= 1_000_000:
            return f"{v/1_000_000:.3g}M"
        if abs_v >= 1_000:
            return f"{v/1_000:.4g}k"
        if abs_v >= 100:
            return f"{v:.1f}"
        if abs_v >= 1:
            return f"{v:.3g}"
        return f"{v:.4f}"

    @staticmethod
    def _safe_dom_id(text: str, index: int, prefix: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_")
        if not cleaned:
            cleaned = "item"
        return f"{prefix}_{index}_{cleaned}"

    @staticmethod
    def _sorted_indices_by_keys(length: int, sort_keys):
        if not sort_keys or len(sort_keys) != length:
            return list(range(length))
        pairs = [(i, str(sort_keys[i])) for i in range(length)]
        pairs.sort(key=lambda x: x[1])
        return [idx for idx, _ in pairs]

    @staticmethod
    def _reorder_2d(values, row_idx, col_idx):
        if values is None:
            return values
        out = []
        for r in row_idx:
            row = values[r] if r < len(values) else []
            out.append([row[c] if c < len(row) else None for c in col_idx])
        return out

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
        df_fi_norm = ResultExporter.normalize_feature_importances(fi_data)
        if not df_fi_norm.empty and "Score" in df_fi_norm.columns:
            score_series = pd.to_numeric(df_fi_norm["Score"], errors="coerce").dropna()
            if not score_series.empty:
                best_aic = float(score_series.min())
        delta = baseline - best_aic
        pct_change = ((best_aic - baseline) / baseline * 100.0) if baseline else 0.0
        n_total = max((meta.get("n_columns", 0) - 1), 0) if isinstance(meta, dict) else 0
        selected_summary = f"{len(fi_data)} / {n_total} features" if n_total else f"{len(fi_data)} features"
        final_mode = str(result.get("mode", "N/A")).upper()
        mode_title = final_mode.capitalize() if final_mode != "N/A" else "N/A"
        requested_task = result.get("requested_task_type", result.get("task_type", "auto"))
        task_hint = str(requested_task).capitalize()
        metric_name = "AICc" if bool(result.get("use_aicc", True)) else "AIC"
        estimator_name = "DecisionTreeRegressor bins" if final_mode == "REGRESSION" else "DecisionTreeClassifier bins"
        
        # 2. Global Charts
        
        # Feature Importance
        fig_fi = go.Figure()
        if not df_fi_norm.empty:
            # MATCH GUI LOGIC: Top 15 by Delta AIC, then sorted
            df_top = df_fi_norm.nlargest(15, "Delta_Score").sort_values("Delta_Score", ascending=True)
            x_vals = [float(v) for v in df_top["Delta_Score"].tolist()]
            y_vals = df_top["Feature"].astype(str).tolist()
            text_vals = [ResultExporter._format_compact(v) for v in x_vals]
            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                orientation="h",
                text=text_vals,
                texttemplate="%{text}",
                textposition="outside",
                marker=dict(
                    color=x_vals,
                    colorscale=UNIFIED_BAR_COLORSCALE,
                    line=dict(width=0),
                    colorbar=dict(title="Delta_Score"),
                    showscale=False
                ),
                hovertemplate="Feature=%{y}<br>Delta_Score=%{x:.6g}<extra></extra>"
            ))
            fig_fi.update_layout(title="Top Features by Impact (Delta AIC)")
            # Apply standardized style
            ResultExporter.apply_chart_style(fig_fi, is_dark)
            # Add specific margins for bar chart text
            fig_fi.update_layout(margin=dict(r=50))

        # Interactions Heatmap (Global)
        fig_heat = go.Figure()
        ii_data = result.get('interaction_importances', [])
        df_ii_norm = ResultExporter.normalize_interaction_importances(ii_data)
        mat = ResultExporter.build_interaction_matrix(df_ii_norm)
        if mat.empty:
            mat = ResultExporter.build_interaction_matrix_from_details(result.get("interaction_details", {}))
        if not mat.empty:
            fig_heat = go.Figure(data=go.Heatmap(
                # Use plain JSON arrays to avoid browser-dependent typed-array decoding in exported HTML.
                z=mat.values.tolist(),
                x=mat.columns.tolist(),
                y=mat.index.tolist(),
                colorscale=UNIFIED_HEATMAP_COLORSCALE,
                colorbar=dict(title="Gain"),
                hovertemplate="Feature_1=%{x}<br>Feature_2=%{y}<br>Gain=%{z:.6g}<extra></extra>"
            ))
            fig_heat.update_layout(title="Global Interaction Network")
            ResultExporter.apply_chart_style(fig_heat, is_dark)
        else:
            fig_heat.add_annotation(
                text="No interaction data available (Gain/Pair_Score not found)",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
            fig_heat.update_layout(title="Global Interaction Network")
            fig_heat.update_xaxes(visible=False)
            fig_heat.update_yaxes(visible=False)
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
            bin_labels = detail.get('bin_display_labels') or detail.get('bin_labels', [])
            bin_sort_keys = detail.get('bin_sort_keys', [])
            if not bin_labels and detail.get('bin_edges'):
                edges = detail['bin_edges']
                bin_labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges)-1)]
            elif not bin_labels:
                bin_labels = [f"Bin {i}" for i in range(len(bin_counts))]
            order_idx = ResultExporter._sorted_indices_by_keys(len(bin_labels), bin_sort_keys)
            if order_idx:
                bin_labels = [bin_labels[i] for i in order_idx]
                if bin_counts:
                    bin_counts = [bin_counts[i] for i in order_idx]
                if bin_means:
                    bin_means = [bin_means[i] for i in order_idx]
            
            # Chart
            sub_fig = go.Figure()
            sub_fig.add_trace(go.Bar(x=bin_labels, y=bin_counts, name='Count', marker_color='rgba(100, 100, 100, 0.6)', yaxis='y'))
            if bin_means:
               sub_fig.add_trace(go.Scatter(x=bin_labels, y=bin_means, name='Target', yaxis='y2', mode='lines+markers', line=dict(color=NEON_CYAN, width=3)))
            
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
            feature_stats[feat] = df_stat.to_html(classes="table table-glass table-sm table-bordered table-hover", index=False, float_format="%.4f")

        # 4. Interaction Details (All Pairs)
        interaction_charts = {} # pair_key -> html_div
        interaction_stats = {} # pair_key -> html_table_str
        interaction_details = result.get('interaction_details', {})
        all_interactions = sorted(list(interaction_details.keys()))
        
        for pair_key in all_interactions:
            det = interaction_details.get(pair_key)
            if det:
                bin_labels_1 = det.get('bin_display_labels_1') or det.get('bin_labels_1', [])
                bin_labels_2 = det.get('bin_display_labels_2') or det.get('bin_labels_2', [])
                sort_keys_1 = det.get('bin_sort_keys_1') or []
                sort_keys_2 = det.get('bin_sort_keys_2') or []
                row_idx = ResultExporter._sorted_indices_by_keys(len(bin_labels_1), sort_keys_1)
                col_idx = ResultExporter._sorted_indices_by_keys(len(bin_labels_2), sort_keys_2)
                bin_labels_1 = [bin_labels_1[i] for i in row_idx]
                bin_labels_2 = [bin_labels_2[i] for i in col_idx]
                means = ResultExporter._reorder_2d(det.get('means'), row_idx, col_idx)
                if means is None or not bin_labels_1 or not bin_labels_2:
                    continue
                feature_1 = det.get('feature_1') or pair_key
                feature_2 = det.get('feature_2') or ""
                metric_name = str(det.get("metric_name") or "Target Mean")
                fig_int = go.Figure(data=go.Heatmap(
                    z=means, x=bin_labels_2, y=bin_labels_1,
                    colorscale=UNIFIED_HEATMAP_COLORSCALE, colorbar=dict(title=metric_name)
                ))
                fig_int.update_layout(
                    title=f"Interaction: {feature_1} vs {feature_2}",
                    xaxis_title=feature_2,
                    yaxis_title=feature_1,
                    xaxis=dict(
                        tickfont=dict(size=11),
                        title_font=dict(size=12, color="#cfd8dc"),
                        title_standoff=10,
                    ),
                    yaxis=dict(
                        tickfont=dict(size=11),
                        title_font=dict(size=12, color="#cfd8dc"),
                        title_standoff=10,
                    ),
                    height=500,
                )
                ResultExporter.apply_chart_style(fig_int, is_dark)
                # Add specific margins
                fig_int.update_layout(margin=dict(l=40, r=40, t=80, b=100))
                interaction_charts[pair_key] = pio.to_html(fig_int, full_html=False, include_plotlyjs=False)
                
                # Interaction Tables (Sample Count & Target Mean)
                counts = ResultExporter._reorder_2d(det.get('counts'), row_idx, col_idx)
                if counts is None:
                    counts = [[pd.NA for _ in bin_labels_2] for _ in bin_labels_1]
                df_counts = pd.DataFrame(counts, index=bin_labels_1, columns=bin_labels_2)
                df_means = pd.DataFrame(means, index=bin_labels_1, columns=bin_labels_2)
                dominant_labels = ResultExporter._reorder_2d(det.get("dominant_labels"), row_idx, col_idx)
                
                # Format means
                df_means = df_means.map(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)

                # Reset index for display
                df_counts_disp = df_counts.reset_index().rename(columns={'index': feature_1})
                df_means_disp = df_means.reset_index().rename(columns={'index': feature_1})

                # Two tables side-by-side
                table_counts_html = df_counts_disp.to_html(classes="table table-glass table-sm table-bordered table-hover", index=False)
                table_means_html = df_means_disp.to_html(classes="table table-glass table-sm table-bordered table-hover", index=False)

                dominant_html = ""
                if dominant_labels:
                    df_dominant = pd.DataFrame(dominant_labels, index=bin_labels_1, columns=bin_labels_2)
                    df_dominant_disp = df_dominant.reset_index().rename(columns={'index': feature_1})
                    table_dominant_html = df_dominant_disp.to_html(
                        classes="table table-glass table-sm table-bordered table-hover",
                        index=False,
                    )
                    dominant_html = f"""
                    <div class="col-md-12">
                        <h6 class="text-secondary mt-3">Dominant Class Matrix</h6>
                        <div class="table-responsive">{table_dominant_html}</div>
                    </div>
                    """

                # Combine into grid
                combined_html = f"""
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="text-secondary mt-3">Sample Count Matrix</h6>
                        <div class="table-responsive">{table_counts_html}</div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-secondary mt-3">{metric_name} Matrix</h6>
                        <div class="table-responsive">{table_means_html}</div>
                    </div>
                    {dominant_html}
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
    --accent-primary: #00f3ff;
    --surface-border: rgba(255, 255, 255, 0.14);
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

.header-section {{
    padding: 1.1rem 0 1.2rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}}

.section-card {{
    border: 1px solid var(--surface-border);
}}

.section-title {{
    color: var(--text-primary);
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    border-left: 3px solid var(--accent-primary);
    padding-left: 0.6rem;
    margin-bottom: 0.9rem;
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

.dashboard-kpi-card {{
    padding: 0.95rem 1.05rem;
    min-height: 144px;
}}

.dashboard-kpi-card .kpi-label {{
    font-size: 0.74rem;
    letter-spacing: 0.07em;
    margin-bottom: 0.35rem;
    line-height: 1.2;
}}

.dashboard-kpi-card .kpi-value {{
    font-size: 1.42rem;
    margin-bottom: 0.18rem;
    line-height: 1.22;
}}

.kpi-main-value {{
    color: var(--text-primary);
    font-size: 1.38rem;
    font-weight: 700;
    line-height: 1.22;
    letter-spacing: 0.01em;
}}

.kpi-delta {{
    color: var(--accent-primary);
    font-size: 0.93rem;
    margin-bottom: 0.18rem;
    line-height: 1.2;
}}

.kpi-note {{
    color: var(--text-secondary);
    font-size: 0.74rem;
    letter-spacing: 0.01em;
    line-height: 1.2;
}}

.kpi-meta {{
    color: var(--text-secondary);
    font-size: 0.78rem;
    line-height: 1.25;
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
    text-align: center;
    vertical-align: middle;
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
    background-color: rgba(40, 40, 50, 0.9) !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    -webkit-text-fill-color: #e0e0e0 !important;
}}

select.form-select:focus {{
    background-color: rgba(60, 60, 70, 0.92) !important;
    color: #ffffff !important;
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 10px rgba(0, 243, 255, 0.3) !important;
}}

select.form-select option {{
    background-color: #1f232c !important;
    color: #e0e0e0 !important;
}}

select.form-select option:checked {{
    background-color: #2a3240 !important;
    color: #ffffff !important;
}}

/* Helper for Dropdown Logic */
.hidden {{
    display: none !important;
}}

.detail-card {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.8rem;
}}

.detail-card .plotly-graph-div {{
    margin-top: 0.15rem;
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
                <div class="row mb-4 g-3">
                    <div class="col-12 col-lg-4">
                        <div class="glass-card dashboard-kpi-card text-center h-100 d-flex flex-column justify-content-center">
                            <div class="kpi-label">AIC Comparison</div>
                            <h2 class="kpi-main-value mb-1">{baseline:,.0f} <span class="mx-1 text-secondary">-></span> {best_aic:,.0f}</h2>
                            <div class="kpi-delta fw-semibold">Delta {delta:,.0f} ({pct_change:.1f}%)</div>
                            <small class="kpi-note">AIC is better when lower.</small>
                        </div>
                    </div>
                    <div class="col-12 col-lg-4">
                        <div class="glass-card dashboard-kpi-card text-center h-100 d-flex flex-column justify-content-center">
                            <div class="kpi-value">{selected_summary}</div>
                            <div class="kpi-label">Selected Features</div>
                        </div>
                    </div>
                    <div class="col-12 col-lg-4">
                        <div class="glass-card dashboard-kpi-card text-center h-100 d-flex flex-column justify-content-center">
                            <div class="kpi-label">Model Type</div>
                            <h2 class="kpi-main-value mb-1">{mode_title} ({task_hint})</h2>
                            <div class="kpi-meta">Estimator: {estimator_name}</div>
                            <div class="kpi-meta">Metric: {metric_name}</div>
                        </div>
                    </div>
                </div>

                <!-- Global Charts -->
                <div class="row mb-5">
                    <div class="col-md-6">
                        <div class="glass-card section-card">
                             <!-- Feature Importance -->
                             {pio.to_html(fig_fi, full_html=False, include_plotlyjs=False)}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="glass-card section-card">
                             <!-- Interaction Network -->
                             {pio.to_html(fig_heat, full_html=False, include_plotlyjs=False)}
                        </div>
                    </div>
                </div>
        """
        
        # Feature Selection Dropdown
        feature_id_map = {f: ResultExporter._safe_dom_id(f, i, "feat") for i, f in enumerate(all_features)}
        feature_options = "".join([f'<option value="{feature_id_map[f]}">{f}</option>' for f in all_features])
        
        html_content += f"""
                <div class="glass-card section-card p-3 mb-5">
                    <h3 class="section-title"><i class="bi bi-bar-chart-fill me-2"></i>Feature Analysis</h3>
                    <div class="mb-3">
                        <div class="row align-items-center g-3">
                            <div class="col-md-4">
                                <label class="form-label text-muted mb-0">Select Feature to Inspect:</label>
                            </div>
                            <div class="col-md-8">
                                <select class="form-select" onchange="showFeature(this.value)">
                                    {feature_options}
                                </select>
                            </div>
                        </div>
                    </div>
                    <div id="feature-details-container">
        """
        
        # Loop features - Generate Hidden Divs
        for i, feat in enumerate(all_features):
            chart_html = feature_plots.get(feat, "")
            table_html = feature_stats.get(feat, "")
            dom_id = feature_id_map[feat]
            # Show first feature by default, hide others
            is_hidden = 'hidden' if i > 0 else '' 
            
            html_content += f"""
                    <div id="chart_{dom_id}" class="feature-chart {is_hidden}">
                        <div class="glass-card detail-card mb-3">
                            <div>
                                {chart_html}
                            </div>
                        </div>
                    </div>
                    
                    <div id="table_{dom_id}" class="feature-table {is_hidden}">
                        <div class="glass-card detail-card mb-4">
                            <div>
                                <h6 class="text-muted mb-3">Binning Statistics</h6>
                                <div class="table-responsive">
                                    {table_html}
                                </div>
                            </div>
                        </div>
                    </div>
            """
            
        html_content += "</div></div>"

        # Interaction Selection Dropdown
        interaction_id_map = {p: ResultExporter._safe_dom_id(p, i, "int") for i, p in enumerate(all_interactions)}
        int_options = "".join([f'<option value="{interaction_id_map[p]}">{p}</option>' for p in all_interactions])
        
        html_content += f"""
                <div class="glass-card section-card p-3 mb-5">
                    <h3 class="section-title"><i class="bi bi-diagram-3-fill me-2"></i>Bivariate Interaction Analysis</h3>
                    <div class="mb-3">
                        <div class="row align-items-center g-3">
                             <div class="col-md-4">
                                <label class="form-label text-muted mb-0">Select Interaction Pair:</label>
                            </div>
                            <div class="col-md-8">
                                <select class="form-select" onchange="showInteraction(this.value)">
                                    {int_options}
                                </select>
                            </div>
                        </div>
                    </div>
                    <div id="interaction-details-container">
        """
        
        for i, pair_key in enumerate(all_interactions):
            chart_html = interaction_charts.get(pair_key, "")
            table_html = interaction_stats.get(pair_key, "")
            dom_id = interaction_id_map[pair_key]
            is_hidden = 'hidden' if i > 0 else ''
            
            html_content += f"""
                    <div id="intchart_{dom_id}" class="int-chart {is_hidden}">
                        <div class="glass-card detail-card mb-3">
                            <div>
                                {chart_html}
                            </div>
                        </div>
                    </div>
                    
                    <div id="inttable_{dom_id}" class="int-table {is_hidden}">
                         <div class="glass-card detail-card mb-4">
                            <div>
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
