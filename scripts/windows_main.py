import sys
import os
import time
import socket
import subprocess
import threading
import multiprocessing
import logging
from pathlib import Path
import webview
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("startup.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def wait_for_port(port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# --- Subprocess Entry Points ---

def run_api_server(port):
    """Run FastAPI Logic"""
    import uvicorn
    from advanced_catdap.service.api import app
    # Check if we are in frozen mode to set proper reload config
    is_frozen = getattr(sys, 'frozen', False)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", reload=not is_frozen)

def run_streamlit_server(api_port, streamlit_port):
    """Run Streamlit Logic"""
    # Set environment variable for the frontend to find the API
    os.environ["API_URL"] = f"http://127.0.0.1:{api_port}"
    
    # Locate the app.py file
    # In frozen mode, we need to find where we bundled it.
    # We will assume it's in the standard location or bundled.
    # For PyInstaller, we usually bundle advanced_catdap package.
    
    if hasattr(sys, '_MEIPASS'):
        # In temporary directory
        app_path = os.path.join(sys._MEIPASS, "advanced_catdap", "frontend", "app.py")
    else:
        app_path = os.path.join(os.getcwd(), "advanced_catdap", "frontend", "app.py")
    
    if not os.path.exists(app_path):
        logger.error(f"Streamlit app not found at {app_path}")
        sys.exit(1)

    # Hack sys.argv to simulate 'streamlit run'
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port", str(streamlit_port),
        "--server.headless", "true",
        "--browser.serverAddress", "127.0.0.1",
        "--global.developmentMode", "false"
    ]
    
    from streamlit.web import cli as stcli
    sys.exit(stcli.main())

def run_worker_process(args):
    """Run Worker Logic"""
    from advanced_catdap.service.local_worker import run_worker
    run_worker(
        job_id=args.job_id,
        dataset_id=args.dataset_id,
        params_json=args.params,
        data_dir=args.data_dir,
        db_path=args.db_path
    )

# --- Main Launcher ---

def main():
    # Handle Subprocess Modes (PyInstaller multiprocess support)
    if len(sys.argv) > 1:
        if sys.argv[1] == "--api":
            port = int(sys.argv[2])
            run_api_server(port)
            return
        elif sys.argv[1] == "--streamlit":
            api_port = int(sys.argv[2])
            st_port = int(sys.argv[3])
            run_streamlit_server(api_port, st_port)
            return
        elif sys.argv[1] == "--worker":
            # Worker argument parsing
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--worker", action="store_true")
            parser.add_argument("--job-id", required=True)
            parser.add_argument("--dataset-id", required=True)
            parser.add_argument("--params", required=True)
            parser.add_argument("--data-dir", default="data")
            parser.add_argument("--db-path", default="data/jobs.db")
            args = parser.parse_args()
            run_worker_process(args)
            return

    # --- Main Process Logic ---
    multiprocessing.freeze_support() # Crucial for Windows
    
    logger.info("Starting AdvancedCATDAP Launcher...")
    
    api_port = find_free_port()
    st_port = find_free_port()
    logger.info(f"Ports assigned - API: {api_port}, Streamlit: {st_port}")
    
    # Launch API Subprocess
    if getattr(sys, 'frozen', False):
        exe = sys.executable
        api_cmd = [exe, "--api", str(api_port)]
        st_cmd = [exe, "--streamlit", str(api_port), str(st_port)]
    else:
        exe = sys.executable
        script = os.path.abspath(__file__)
        api_cmd = [exe, script, "--api", str(api_port)]
        st_cmd = [exe, script, "--streamlit", str(api_port), str(st_port)]

    # Use creationflags to hide console window of subprocesses if possible
    CREATE_NO_WINDOW = 0x08000000
    
    logger.info("Launching API Server...")
    api_proc = subprocess.Popen(
        api_cmd, 
        creationflags=CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    )
    
    logger.info("Launching Streamlit Server...")
    st_proc = subprocess.Popen(
        st_cmd, 
        creationflags=CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
        env={**os.environ, "API_URL": f"http://127.0.0.1:{api_port}"}
    )
    
    # Wait for startup
    if not wait_for_port(api_port):
        logger.error("API failed to start")
        api_proc.terminate()
        st_proc.terminate()
        sys.exit(1)
        
    if not wait_for_port(st_port):
        logger.error("Streamlit failed to start")
        api_proc.terminate()
        st_proc.terminate()
        sys.exit(1)

    # Launch GUI
    logger.info("Launching WebView...")
    
    def on_closing():
        logger.info("Window closing, killing servers...")
        api_proc.terminate()
        st_proc.terminate()
    
    window = webview.create_window(
        'AdvancedCATDAP', 
        f'http://localhost:{st_port}',
        width=1280,
        height=800,
        resizable=True
    )
    
    # Start webview (blocking)
    webview.start(func=None, debug=False)
    
    # Cleanup after window closes
    on_closing()
    sys.exit(0)

if __name__ == "__main__":
    main()
