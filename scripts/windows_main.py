"""
AdvancedCATDAP Windows Main Entry Point - Dash Version (Simple)
Uses threading instead of subprocess for simpler startup.
"""
import sys
import os
import threading
import time
import socket
from pathlib import Path
import webview

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def find_free_port():
    """Find an available port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def wait_for_server(port, timeout=30):
    """Wait for server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('127.0.0.1', port))
                return True
        except ConnectionRefusedError:
            time.sleep(0.5)
    return False

def run_api_server(port):
    """Run API server"""
    import uvicorn
    from advanced_catdap.service import api as api_module

    api_module.configure_desktop_export_hook(DesktopExportBridge().save_html_report)
    api_app = api_module.app
    uvicorn.run(
        api_app,
        host="127.0.0.1",
        port=port,
        log_level="warning"
    )

def run_dash_server(port, api_port):
    """Run Dash server"""
    os.environ['API_URL'] = f'http://127.0.0.1:{api_port}'
    from advanced_catdap.frontend.dash_app import app, configure_api_client
    configure_api_client(f'http://127.0.0.1:{api_port}')
    app.run(debug=False, port=port, use_reloader=False)


class DesktopExportBridge:
    """Bridge that shows native save dialog and writes exported HTML."""

    def save_html_report(self, filename: str, payload: bytes) -> str | None:
        default_name = filename or "AdvancedCATDAP_Report.html"
        if not default_name.lower().endswith(".html"):
            default_name += ".html"

        if not webview.windows:
            return None

        save_dialog = getattr(webview, "SAVE_DIALOG", 1)
        path = webview.windows[0].create_file_dialog(
            save_dialog,
            save_filename=default_name,
            file_types=("HTML files (*.html)", "All files (*.*)"),
        )
        if not path:
            return None

        selected = path[0] if isinstance(path, (list, tuple)) else path
        output_path = Path(str(selected))
        output_path.write_bytes(payload)
        return str(output_path)


if __name__ == "__main__":
    print("Starting AdvancedCATDAP (Dash version)...")
    os.environ["CATDAP_DESKTOP_MODE"] = "1"
    
    # Find free ports
    api_port = find_free_port()
    dash_port = find_free_port()
    print(f"Ports - API: {api_port}, Dash: {dash_port}")
    
    # Start API server in thread
    print("Starting API server...")
    api_thread = threading.Thread(target=run_api_server, args=(api_port,), daemon=True)
    api_thread.start()
    
    if not wait_for_server(api_port, timeout=30):
        print("ERROR: API server failed to start")
        sys.exit(1)
    print("API server ready")
    
    # Start Dash server in thread
    print("Starting Dash server...")
    dash_thread = threading.Thread(target=run_dash_server, args=(dash_port, api_port), daemon=True)
    dash_thread.start()
    
    if not wait_for_server(dash_port, timeout=30):
        print("ERROR: Dash server failed to start")
        sys.exit(1)
    print("Dash server ready")
    
    url = f'http://127.0.0.1:{dash_port}'
    
    # Create WebView window
    try:
        print("Opening WebView2 window...")
        window = webview.create_window(
            'AdvancedCATDAP',
            url,
            width=1400,
            height=900,
            resizable=True,
            min_size=(1024, 768)
        )
        
        webview.start(gui='edgechromium', debug=False)
        print("Window closed, exiting...")
    except Exception as e:
        print(f"WebView failed: {e}")
        import webbrowser
        webbrowser.open(url)
        
        print("\n" + "="*60)
        print("  AdvancedCATDAP is running!")
        print(f"  Open in browser: {url}")
        print("="*60)
        print("\nPress Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    print("Goodbye!")
