from __future__ import annotations

import os
import threading
import time

from advanced_catdap.runtime.desktop_export import DesktopExportBridge
from advanced_catdap.runtime.process import find_free_port, wait_for_server


def run_api_server(port: int, desktop_mode: bool) -> None:
    """Run API server."""
    import uvicorn
    from advanced_catdap.service import api as api_module

    if desktop_mode:
        api_module.configure_desktop_export_hook(DesktopExportBridge().save_html_report)
    else:
        api_module.configure_desktop_export_hook(None)
    uvicorn.run(api_module.app, host="127.0.0.1", port=port, log_level="warning")


def run_dash_server(port: int, api_port: int) -> None:
    """Run Dash server."""
    os.environ["API_URL"] = f"http://127.0.0.1:{api_port}"
    from advanced_catdap.frontend.dash_app import app, configure_api_client

    configure_api_client(f"http://127.0.0.1:{api_port}")
    app.run(debug=False, port=port, use_reloader=False)


def _start_servers(desktop_mode: bool) -> tuple[int, int] | None:
    api_port = find_free_port()
    dash_port = find_free_port()
    print(f"Ports - API: {api_port}, Dash: {dash_port}")

    print("Starting API server...")
    api_thread = threading.Thread(target=run_api_server, args=(api_port, desktop_mode), daemon=True)
    api_thread.start()
    if not wait_for_server(api_port, timeout=30):
        print("ERROR: API server failed to start")
        return None
    print("API server ready")

    print("Starting Dash server...")
    dash_thread = threading.Thread(target=run_dash_server, args=(dash_port, api_port), daemon=True)
    dash_thread.start()
    if not wait_for_server(dash_port, timeout=30):
        print("ERROR: Dash server failed to start")
        return None
    print("Dash server ready")
    return (api_port, dash_port)


def run_desktop_mode() -> int:
    """Start desktop GUI stack."""
    import webview

    print("Starting AdvancedCATDAP (Desktop mode)...")
    os.environ["CATDAP_DESKTOP_MODE"] = "1"

    ports = _start_servers(desktop_mode=True)
    if not ports:
        return 1
    _, dash_port = ports

    url = f"http://127.0.0.1:{dash_port}"
    try:
        print("Opening WebView2 window...")
        webview.create_window(
            "AdvancedCATDAP",
            url,
            width=1400,
            height=900,
            resizable=True,
            min_size=(1024, 768),
        )
        webview.start(gui="edgechromium", debug=False)
        print("Window closed, exiting...")
    except Exception as exc:
        print(f"WebView failed: {exc}")
        import webbrowser

        webbrowser.open(url)
        print("\n" + "=" * 60)
        print("  AdvancedCATDAP is running!")
        print(f"  Open in browser: {url}")
        print("=" * 60)
        print("\nPress Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    print("Goodbye!")
    return 0


def run_web_mode() -> int:
    """Start browser-only mode without native WebView."""
    import webbrowser

    print("Starting AdvancedCATDAP (Web mode)...")
    os.environ.pop("CATDAP_DESKTOP_MODE", None)
    ports = _start_servers(desktop_mode=False)
    if not ports:
        return 1
    _, dash_port = ports

    url = f"http://127.0.0.1:{dash_port}"
    webbrowser.open(url)
    print("\n" + "=" * 60)
    print("  AdvancedCATDAP is running in browser mode")
    print(f"  Open in browser: {url}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    print("Goodbye!")
    return 0


# Backward-compatible alias for legacy tests/callers.
run_gui_mode = run_desktop_mode

