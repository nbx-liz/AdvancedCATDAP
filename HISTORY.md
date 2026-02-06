# Project History: Dash Migration (February 2026)

This document records the technical decisions, challenges, and solutions encountered during the migration of AdvancedCATDAP from a Streamlit web app to a Dash-based desktop application.

## 🎯 Objective
Migrate the frontend to **Dash** to improve desktop integration (via `pywebview`), performance, and control over UI theming/layout, while retaining the existing FastAPI backend.

## 🛠 Architecture Decisions

### 1. Desktop Launcher (`windows_main.py`)
*   **Decision**: Use `threading` instead of `subprocess` to launch the API and Dash servers.
*   **Reason**: `subprocess` management on Windows caused issues with orphan processes and signal handling. Running servers in daemon threads within the main process ensures that closing the WebView window kills all related services instantly.

### 2. Theming & Dark Mode
*   **Decision**: Use **CSS Variables** and **Clientside Callbacks**.
*   **Reason**: Server-side callbacks for theming introduce latency. By using `clientside_callback` to toggle the `data-theme` attribute on the `<html>` element, we achieve instant theme switching.
*   **Detail**:
    *   Style definitions moved to `advanced_catdap/frontend/assets/style.css`.
    *   Specific overrides added for standard Bootstrap components and React-Select dropdowns (which are notoriously hard to style in dark mode).

### 3. State Management
*   **Decision**: Use `dcc.Store` (memory/local) instead of global variables.
*   **Reason**: Dash is stateless. To emulate Streamlit's `session_state`, we utilized:
    *   `store-dataset-meta`: Dataset metadata.
    *   `store-analysis-result`: Large analysis results.
    *   `store-deepdive-state`: UI state for the Deep Dive tab.
    *   `theme-store`: Persisted using `storage_type='local'`.

## 🐛 Challenges & Solutions

### 🚨 Challenge 1: Startup Crash (AttributeError: html.Style)
*   **Symptom**: Application fails to launch with `AttributeError: module 'dash.html' has no attribute 'Style'`.
*   **Cause**: The code attempted to inject CSS using `html.Style(CUSTOM_CSS)`. This component was deprecated/removed in recent Dash versions or required `dash-dangerously-set-inner-html`.
*   **Solution**: Moved all CSS to an external file `assets/style.css`. Dash automatically includes CSS files found in the `assets/` folder.

### 🚨 Challenge 2: Deep Dive Charts Missing/freezing
*   **Symptom**: The "Deep Dive" tab showed empty charts or threw errors.
*   **Cause**: Data mismatch between Backend Pydantic models (lists of floats) and Frontend expectations. Also, missing Logic for generating bin labels from `bin_edges`.
*   **Solution**:
    *   Implemented robust DataFrame reconstruction in `render_deepdive_tab`.
    *   Added logic to generate readable bin labels (e.g., `[0.00, 5.00)`) from `bin_edges` when the backend doesn't provide explicit labels.

### 🚨 Challenge 3: WebView2 & Plotly Freeze
*   **Symptom**: Plotly charts sometimes caused the WebView window to freeze or not render.
*   **Investigation**: Suspected threading or GPU issues.
*   **Final Root Cause**: Often related to Exceptions occurring during the initial callback rendering (e.g., `prevent_initial_call` preventing necessary data loading, or data parsing errors).
*   **Solution**: Fixed the underlying Python exceptions in callbacks. The WebView freeze was a side-effect of unhandled JS/Python errors blocking the renderer.

### 🚨 Challenge 4: Legacy Test Failures
*   **Symptom**: `pytest` failed on `tests/test_app.py`.
*   **Cause**: These tests targeted the old `streamlit` app and mocked Streamlit components that are no longer relevant or compatible with the new environment.
*   **Solution**: Deleted legacy tests. Validated the system using the existing backend tests (which cover the core logic) and a new startup verification script.

## 📝 Usage

### Running the App
```bash
# Production entry point
uv run python scripts/windows_main.py
```

### Development
```bash
# Run Dash server standalone (hot-reload enabled)
uv run python -m advanced_catdap.frontend.dash_app
```

## 🔮 Future Roadmap
1.  **Simulator Tab**: Implement "What-If" analysis using the `transform` method on modified inputs.
2.  **E2E Testing**: Implement Playwright tests to verify clicking through tabs and interactions automatically.
