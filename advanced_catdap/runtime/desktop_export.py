from __future__ import annotations

from pathlib import Path


class DesktopExportBridge:
    """Bridge that shows native save dialog and writes exported HTML."""

    def save_html_report(self, filename: str, payload: bytes) -> str | None:
        import webview

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

