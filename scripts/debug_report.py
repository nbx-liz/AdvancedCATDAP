import pandas as pd
import sys
import os
import io

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_catdap.service.exporter import ResultExporter

def debug_report_generation():
    print("DEBUG: Generating Report...")
    
    # Mock Result with Potentially Problematic Feature Names and Values
    result = {
        'mode': 'CLASSIFICATION',
        'baseline_score': 67743.0,
        'feature_importances': [
            {'Feature': 'Churn', 'Delta_Score': 4226.0, 'Score': 63517.0},
            {'Feature': 'Gender.9m', 'Delta_Score': 100.0, 'Score': 67600.0},
            {'Feature': 'With Space', 'Delta_Score': 50.0, 'Score': 67700.0}
        ],
        'interaction_importances': [
            {'Feature_1': 'Churn', 'Feature_2': 'Gender.9m', 'Gain': 5000.0},
            {'Feature_1': 'With Space', 'Feature_2': 'Churn', 'Gain': 0.5} # Normal vs Large
        ],
        'feature_details': {
            'Churn': {'bin_counts': [100, 200], 'bin_labels': ['A', 'B']},
            'Gender.9m': {'bin_counts': [50, 50], 'bin_labels': ['M', 'F']},
            'With Space': {'bin_counts': [1, 2], 'bin_labels': ['X', 'Y']}
        },
        'interaction_details': {
            'Churn - Gender.9m': {
                'feature_1': 'Churn', 'feature_2': 'Gender.9m',
                'means': [[0.1, 0.2], [0.3, 0.4]],
                'bin_labels_1': ['A', 'B'], 'bin_labels_2': ['M', 'F']
            }
        }
    }
    
    meta = {'dataset_id': 'debug_ds', 'n_rows': 1000, 'n_columns': 10}
    
    html_io = ResultExporter.generate_html_report(result, meta, theme='dark')
    html_content = html_io.getvalue().decode('utf-8')
    
    # Check 1: Metric Values in HTML
    # Look for "4226" in the HTML source (it should be in the javascript data for plotly)
    if "4226" in html_content:
        print("PASS: Delta Score 4226 found in HTML.")
    else:
        print("FAIL: Delta Score 4226 NOT found in HTML!")
        
    # Check 2: Interaction Gain 5000
    if "5000" in html_content:
        print("PASS: Gain 5000 found in HTML.")
    else:
        print("FAIL: Gain 5000 NOT found in HTML!")
        
    # Check 3: ID Generation for 'Gender.9m'
    # Should find id="chart_Gender.9m"
    if 'id="chart_Gender.9m"' in html_content:
        print("PASS: ID for Gender.9m found.")
    else:
        print("FAIL: ID for Gender.9m NOT found.")
        
    # Check 4: Option value
    if '<option value="Gender.9m">' in html_content:
        print("PASS: Option for Gender.9m found.")
    else:
        print("FAIL: Option for Gender.9m NOT found.")

    with open("debug_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Saved debug_report.html")

if __name__ == "__main__":
    debug_report_generation()
