import pytest
from unittest.mock import MagicMock, patch
import sys

def test_app_import():
    """
    Smoke test to check if app.py can validly be imported (syntax check and basic setup).
    We mock streamlit to avoid starting a server.
    """
    mocks = {"streamlit": MagicMock(), "plotly.express": MagicMock(), "plotly.graph_objects": MagicMock()}
    mocks["streamlit"].tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
    
    with patch.dict(sys.modules, mocks):
        import advanced_catdap.frontend.app
