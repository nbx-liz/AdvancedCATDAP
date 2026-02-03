import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from advanced_catdap.visualizer import plot_importance, plot_interaction_heatmap
from advanced_catdap.core import AdvancedCATDAP

# Set backend to avoid window popup
plt.switch_backend('Agg')

@pytest.fixture
def analysis_results(realistic_data_factory):
    """Generates real results from realistic data."""
    df = realistic_data_factory(n_samples=500, task_type='classification', seed=42)
    # Lower threshold to ensure features are kept for interaction search
    # extra_penalty=-100 ensures almost any gain is accepted (> 2.0 - 100)
    model = AdvancedCATDAP(verbose=False, min_cat_fraction=0.0, max_categories=1000, extra_penalty=-100.0) 
    # force delta_threshold negative to keep features even if weak
    results, combo_results = model.analyze(df, target_col='ChurnStatus', delta_threshold=-100.0, top_k=10)
    
    if combo_results.empty:
        # Debug why empty
        # If this happens, coverage will fail. We want to fail test instead.
        raise ValueError("No interactions found in test setup! Visualizer coverage will fail.")
        
    return results, combo_results

def test_plot_importance_realistic(analysis_results):
    """Test plot_importance with real analysis results."""
    results, _ = analysis_results
    try:
        fig = plot_importance(results, top_k=5)
        assert fig is not None, "plot_importance should return a Figure"
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"plot_importance raised exception on real data: {e}")

def test_plot_interaction_heatmap_realistic(analysis_results):
    """Test plot_interaction_heatmap with real analysis results."""
    _, combo_results = analysis_results
    try:
        fig = plot_interaction_heatmap(combo_results, top_k=5)
        assert fig is not None, "plot_interaction_heatmap should return a Figure"
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"plot_interaction_heatmap raised exception on real data: {e}")

def test_plot_importance_empty():
    """Test that plot_importance handles empty/None input gracefully."""
    try:
        plot_importance(pd.DataFrame(), top_k=2)
        plot_importance(None, top_k=2)
    except Exception as e:
        pytest.fail(f"plot_importance raised exception on empty input: {e}")

def test_plot_interaction_heatmap_empty():
    """Test that plot_interaction_heatmap handles empty/None input gracefully."""
    try:
        plot_interaction_heatmap(pd.DataFrame(), top_k=2)
        plot_interaction_heatmap(None, top_k=2)
    except Exception as e:
        pytest.fail(f"plot_interaction_heatmap raised exception on empty input: {e}")
