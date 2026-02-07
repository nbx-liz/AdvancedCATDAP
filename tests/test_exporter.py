import io
import pandas as pd
from advanced_catdap.service.exporter import ResultExporter

def test_html_export_interactive():
    result = {
        'mode': 'CLASSIFICATION',
        'baseline_score': 500.0,
        'feature_importances': [
            {'Feature': 'ColA', 'Delta_Score': 100.0, 'Score': 400.0},
            {'Feature': 'ColB', 'Delta_Score': 50.0, 'Score': 450.0}
        ],
        'interaction_importances': [
             {'Feature_1': 'ColA', 'Feature_2': 'ColB', 'Gain': 10.0}
        ],
        'feature_details': {
            'ColA': {
                'bin_counts': [10, 20],
                'bin_means': [0.5, 0.6],
                'bin_labels': ['Low', 'High']
            }
        },
        'interaction_details': {
            'ColA - ColB': {
                'feature_1': 'ColA', 'feature_2': 'ColB',
                'means': [[0.1, 0.2], [0.3, 0.4]],
                'bin_labels_1': ['L', 'H'], 'bin_labels_2': ['L', 'H']
            }
        }
    }
    meta = {'dataset_id': 'test_data', 'n_rows': 100, 'n_columns': 5}
    
    html_io = ResultExporter.generate_html_report(result, meta)
    assert isinstance(html_io, io.BytesIO)
    
    content = html_io.getvalue().decode('utf-8')
    assert "<!DOCTYPE html>" in content
    assert "Analysis Report" in content
    assert "ColA" in content
    assert "plotly" in content 
    assert "function showFeature(featName)" in content 
    assert "function toggleTheme()" in content # Theme toggle check
    assert "inttable_" in content # Interaction table check
    assert "ColA - ColB" in content
