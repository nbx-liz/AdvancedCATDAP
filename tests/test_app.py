import pytest
from unittest.mock import MagicMock, patch, ANY
import sys
import pandas as pd
from advanced_catdap.service.schema import DatasetMetadata, ColumnInfo

class MockSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    def __setattr__(self, key, value):
        self[key] = value

@pytest.fixture
def mock_streamlit():
    # Create the mock object
    mock_st = MagicMock()
    # Setup session state dict-like behavior with attribute access
    session_state = MockSessionState()
    mock_st.session_state = session_state
    
    # Setup common layout mocks
    mock_st.sidebar = MagicMock()
    mock_st.tabs.return_value = [MagicMock(), MagicMock()]
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
    mock_st.file_uploader.return_value = None
    mock_st.form_submit_button.return_value = False
    
    # Patch sys.modules so when app imports streamlit, it gets our mock
    # We also need to mock plotly
    with patch.dict(sys.modules, {
        "streamlit": mock_st, 
        "plotly.express": MagicMock(), 
        "plotly.graph_objects": MagicMock()
    }):
        # We must reload/import here or in the test. 
        # But if we do it in test, this fixture needs to be active.
        yield mock_st

@pytest.fixture
def mock_client():
    # We also need to patch APIClient. 
    # Since app.py imports it: from advanced_catdap.frontend.api_client import APIClient
    # We can patch strictly that path, BUT since we are reloading app.py,
    # it will try to import APIClient again from the real module.
    # So we should patch the real module or sys.modules for api_client too?
    # Actually, simpler is to mock the class in the imported module AFTER reload?
    # No, app.py instantiates it at top level: client = APIClient(...)
    # So we must mock it BEFORE reload.
    
    mock_cls = MagicMock()
    client_instance = mock_cls.return_value
    
    # We mock the module where APIClient is defined, OR we mock the import in app.py logic?
    # If app.py does `from ... import APIClient`, we can mock the source module `advanced_catdap.frontend.api_client`
    
    with patch("advanced_catdap.frontend.api_client.APIClient", mock_cls):
        yield client_instance

def run_app():
    import advanced_catdap.frontend.app
    import importlib
    # Make sure we reload to trigger top-level code
    importlib.reload(advanced_catdap.frontend.app)

def test_app_initial_load(mock_streamlit, mock_client):
    """Test initial load with no data."""
    run_app()
    
    # Verify page config
    mock_streamlit.set_page_config.assert_called()
    
    # Verify title
    mock_streamlit.title.assert_called()
    
    # Verify session state init
    assert "dataset_id" in mock_streamlit.session_state
    assert mock_streamlit.session_state["dataset_id"] is None

def test_auto_registration(mock_streamlit, mock_client):
    """Test that uploading a file triggers auto-registration."""
    # Setup mock file
    mock_file = MagicMock()
    mock_file.name = "test.csv"
    mock_streamlit.file_uploader.return_value = mock_file
    
    # Setup API response
    mock_meta = DatasetMetadata(
        dataset_id="123", 
        n_rows=100, 
        n_columns=3, 
        columns=[
            ColumnInfo(name="target", dtype="int"),
            ColumnInfo(name="feat1", dtype="float"),
            ColumnInfo(name="feat2", dtype="object")
        ],
        filename="test.csv",
        file_path="/tmp/test.csv"
    )
    mock_client.upload_dataset.return_value = mock_meta
    
    run_app()
    
    # Verify upload called
    mock_client.upload_dataset.assert_called_with(mock_file, "test.csv")
    
    # Verify session state updated
    assert mock_streamlit.session_state["dataset_id"] == "123"
    assert mock_streamlit.session_state["last_uploaded_name"] == "test.csv"
    assert mock_streamlit.session_state["dataset_meta"] == mock_meta

def test_config_form_logic(mock_streamlit, mock_client):
    """Test configuration defaults and submission."""
    # Pre-populate session state
    mock_meta = DatasetMetadata(
        dataset_id="123", 
        n_rows=100, 
        n_columns=3, 
        columns=[
            ColumnInfo(name="target", dtype="int"),
            ColumnInfo(name="feat1", dtype="float"),
            ColumnInfo(name="feat2", dtype="object")
        ],
        filename="test.csv",
        file_path="/tmp/test.csv"
    )
    mock_streamlit.session_state["dataset_id"] = "123"
    mock_streamlit.session_state["dataset_meta"] = mock_meta
    mock_streamlit.session_state["job_id"] = None
    
    # Mock form interaction
    mock_streamlit.selectbox.return_value = "target" # User selects target
    mock_streamlit.multiselect.return_value = ["feat1", "feat2"] # User accepts default
    mock_streamlit.form_submit_button.return_value = True # User clicks submit
    
    mock_client.submit_job.return_value = "job_999"
    
    run_app()
    
    # Verify submit call
    mock_client.submit_job.assert_called()
    call_args = mock_client.submit_job.call_args
    assert call_args[0][0] == "123" # dataset_id
    params = call_args[0][1]
    assert params.target_col == "target"
    assert params.candidates == ["feat1", "feat2"]
    
    # Verify job id stored
    assert mock_streamlit.session_state["job_id"] == "job_999"

def test_results_display(mock_streamlit, mock_client):
    """Test results are displayed when job_id exists."""
    mock_streamlit.session_state["dataset_id"] = "123"
    mock_streamlit.session_state["job_id"] = "job_999"
    mock_streamlit.session_state["analysis_result"] = {"baseline_score": 0.5}
    
    # Client returns SUCCESS status
    mock_client.get_job_status.return_value = {"status": "SUCCESS", "result": {"baseline_score": 0.5}}
    
    run_app()
    
    # Verify results heading shown
    # We look for "Analysis Results" in calls
    found = False
    for call in mock_streamlit.subheader.mock_calls:
        if "Analysis Results" in str(call):
            found = True
            break
    assert found
