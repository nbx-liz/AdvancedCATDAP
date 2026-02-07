# Project History: Dash Migration (February 2026)

This document records the technical decisions, challenges, and solutions encountered during the migration of AdvancedCATDAP from a Streamlit web app to a Dash-based desktop application, as well as subsequent enhancement rounds.

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

---

## ⚠️ 最近のアップデートと継続的な課題 (HTMLレポート)

Dashへの移行成功後、HTMLレポートの強化（WebGUIとの整合性向上）に取り組んできましたが、現在も多くの課題が未解決です。

### HTMLレポート改善の試み (Round 9〜13):

| Phase | 主な対応内容 | 現状と残存課題 |
| :--- | :--- | :--- |
| **Round 9-11** | インタラクティブHTML生成の導入。メトリクス（Delta AIC）の同期。 | **課題**: レポートとGUIで数値が一致しない（例: 0-10 vs 4000）。 |
| **Round 12** | チャートスタイルの統一 (`apply_chart_style`)、フォント色の修正。 | **課題**: 背景色と文字色の不一致による視認性不良の継続。 |
| **Round 13** | `cyborg`テーマの適用、`style.css`の埋め込み、欠落項目の追加。 | **課題**: KPIが見えない、統計テーブルが不鮮明、データが依然として不正確。 |

### 未解決の主な障壁:
- **データ抽出の乖離**: `exporter.py` (静的生成) と `dash_app.py` (動的制御) の間で、データのスケーリングやフィルタリング処理が同期できていない。
- **CSSの競合**: スタンドアロンHTML内での外部ライブラリ (CDN) とカスタムCSSの優先順位制御が非常に困難。
- **エスケープ処理**: f-string内部でのエスケープが不完全であり、特定のブラウザ環境でスクリプトやスタイルが崩れる。

## 📈 現状の総括
WebGUI本体は安定して動作していますが、輸出用のHTMLレポート機能については、正確性と視覚的再現性の両面で、ユーザーの要求水準に達していないのが現状の記録です。

---
*最終更新日: 2026-02-07*

---

## Test Automation Update (February 2026)
- Migrated manual HTML report script checks into `tests/test_report_manual_migration.py`.
- Converted manual SQLite lifecycle verification into `tests/test_sqlite_integration.py`.
- Added `pytest` marker config in `pyproject.toml`:
  - `integration` marker for DB/worker-based slower tests.
  - default test run excludes integration for stability/speed.
