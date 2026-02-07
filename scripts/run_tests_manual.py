import sys
import os
import io

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_catdap.service.exporter import ResultExporter

def test_html_export_interactive():
    print("Starting test_html_export_interactive...")
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
    
    html_io = ResultExporter.generate_html_report(result, meta, theme='dark') # Test with dark theme
    if not isinstance(html_io, io.BytesIO):
        print("FAIL: Expected BytesIO")
        return False
    
    content = html_io.getvalue().decode('utf-8')
    
    checks = [
        "<!DOCTYPE html>",
        "AdvancedCATDAP Report",
        "ColA",
        "plotly",
        "function showFeature(featName)",
        "function toggleTheme()",
        "class=\"dark-mode\"", # Check if dark mode class is applied
        "--bg-color: #060606", # Check if dark palette variable definition is present
        '<select class="form-select"', # Check for dropdown presence
        '.hidden {', # Check for hidden class definition in CSS
        'id="feature-details-container"', # Check for container
        'id="interaction-details-container"' # Check for container
    ]
    
    for check in checks:
        if check not in content:
            print(f"FAIL: '{check}' not found in HTML content.")
            # print(content[:500]) # Print first 500 chars for debugging
            return False
            
    print("PASS: test_html_export_interactive")
    return True

if __name__ == "__main__":
    if test_html_export_interactive():
        print("All manual tests passed.")
        sys.exit(0)
    else:
        print("Manual tests failed.")
        sys.exit(1)
