import sys
import os
import time
import socket
import subprocess
import threading
import webview
import uvicorn
import multiprocessing
from pathlib import Path

# Fix for PyInstaller multiprocessing
multiprocessing.freeze_support()

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def run_api(port):
    """Run FastAPI backend"""
    # Set PYTHONPATH to include the extract directory if frozen
    if hasattr(sys, '_MEIPASS'):
        sys.path.append(sys._MEIPASS)
    
    # Import here to avoid early loading
    from advanced_catdap.service.api import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

def run_streamlit(api_port, streamlit_port):
    """Run Streamlit frontend"""
    if hasattr(sys, '_MEIPASS'):
        sys.path.append(sys._MEIPASS)
    
    # Set env var for Frontend to know API URL
    os.environ['API_URL'] = f"http://127.0.0.1:{api_port}"
    
    # Streamlit CLI arguments
    app_path = get_resource_path(os.path.join("advanced_catdap", "frontend", "app.py"))
    
    # We use sys.executable to run streamlit as a subprocess to ensure clean environment
    # But in frozen state, we can't easily run "streamlit run". 
    # Instead, we can import streamlit.web.cli and run it in this process/thread 
    # OR dispatch to sys.executable with a flag if we want separation.
    
    # Using CLI dispatch method
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port", str(streamlit_port),
        "--server.headless", "true",
        "--global.developmentMode", "false",
        "--browser.serverAddress", "127.0.0.1",
    ]
    
    from streamlit.web import cli
    cli.main()

if __name__ == "__main__":
    # Argument dispatching for multiprocessing
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        port = int(sys.argv[2])
        run_api(port)
        sys.exit(0)
        
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        api_port = int(sys.argv[2])
        st_port = int(sys.argv[3])
        run_streamlit(api_port, st_port)
        sys.exit(0)

    # --- Main Process ---
    
    # 1. Allocate Ports
    api_port = get_free_port()
    st_port = get_free_port()
    
    print(f"Starting AdvancedCATDAP Decktop...")
    print(f"API Port: {api_port}, GUI Port: {st_port}")
    
    # 2. Start Subprocesses
    # We reuse this executable to run the subprocesses
    exe_path = sys.executable
    script_path = os.path.abspath(__file__)
    
    # If frozen, exe_path is the executable. If ufrozen, it's python.exe
    if getattr(sys, 'frozen', False):
        cmd_api = [exe_path, "--api", str(api_port)]
        cmd_st = [exe_path, "--streamlit", str(api_port), str(st_port)]
    else:
        cmd_api = [exe_path, script_path, "--api", str(api_port)]
        cmd_st = [exe_path, script_path, "--streamlit", str(api_port), str(st_port)]
        
    # Start API
    proc_api = subprocess.Popen(cmd_api, cwd=os.getcwd())
    
    # Start Streamlit
    proc_st = subprocess.Popen(cmd_st, cwd=os.getcwd())
    
    # 3. Open WebView
    # Wait a bit for servers
    # Ideally we poll /health, but sleep is simpler for MVP
    time.sleep(2) 
    
    window = webview.create_window(
        title="AdvancedCATDAP",
        url=f"http://127.0.0.1:{st_port}",
        width=1200,
        height=800,
        confirm_close=True
    )
    
    def on_closed():
        print("Closing application...")
        proc_st.terminate()
        proc_api.terminate()
        sys.exit(0)

    webview.start(on_closed)
