import io
import pandas as pd
from advanced_catdap.service.exporter import ResultExporter

def test_excel_export_minimal():
    result = {
        'mode': 'REGRESSION',
        'baseline_score': 1000.0,
        'feature_importances': [
            {'Feature': 'A', 'Delta_Score': 50.0},
            {'Feature': 'B', 'Delta_Score': 30.0}
        ],
        'interaction_importances': [
            {'Feature_1': 'A', 'Feature_2': 'B', 'Gain': 10.0}
        ],
        'feature_details': {
            'A': {
                'bin_counts': [10, 20],
                'bin_means': [0.5, 0.6],
                'bin_labels': ['Low', 'High']
            }
        }
    }
    
    excel_io = ResultExporter.generate_excel_report(result)
    assert isinstance(excel_io, io.BytesIO)
    
    # Verify we can read it back
    with pd.ExcelFile(excel_io, engine='openpyxl') as xls:
        assert 'Summary' in xls.sheet_names
        assert 'Feature Importances' in xls.sheet_names
        assert 'Interactions' in xls.sheet_names
        assert 'Det_A' in xls.sheet_names
        
        df_summary = pd.read_excel(xls, 'Summary')
        assert 'Baseline AIC' in df_summary['Item'].values
        
        df_fi = pd.read_excel(xls, 'Feature Importances')
        assert len(df_fi) == 2

def test_html_export_minimal():
    result = {
        'mode': 'CLASSIFICATION',
        'baseline_score': 500.0,
        'feature_importances': [
            {'Feature': 'ColA', 'Delta_Score': 100.0, 'Score': 400.0},
            {'Feature': 'ColB', 'Delta_Score': 50.0, 'Score': 450.0}
        ],
        'interaction_importances': [],
        'feature_details': {}
    }
    meta = {'dataset_id': 'test_data', 'n_rows': 100, 'n_columns': 5}
    
    html_io = ResultExporter.generate_html_report(result, meta)
    assert isinstance(html_io, io.BytesIO)
    
    content = html_io.getvalue().decode('utf-8')
    assert "<!DOCTYPE html>" in content
    assert "Analysis Report" in content
    assert "ColA" in content
    assert "plotly" in content # Check for Plotly embed
