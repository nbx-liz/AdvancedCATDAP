# プロジェクト履歴（Dash移行とUI改善）

このドキュメントは、AdvancedCATDAP のフロントエンド移行（Streamlit → Dash）および、その後の改善サイクルで行った技術判断・課題・対応内容を記録したものです。

---

## 1. 目的
- `pywebview` を含むデスクトップ利用との相性改善
- 画面レイアウト・テーマ制御の柔軟性向上
- FastAPI バックエンド資産を維持したまま、UIを段階的に進化

---

## 2. 主な設計判断

### 2.1 デスクトップランチャー
- 方針: API/Dash起動を `subprocess` ではなく `threading` 主体に整理
- 目的: Windowsでのプロセス終了漏れ（孤児プロセス）やシグナル制御問題を抑制

### 2.2 テーマ設計
- 方針: CSS変数とクライアントサイド処理を中心に統一
- 目的: テーマ切替時の遅延を抑え、即時反映を実現
- 実装拠点: `advanced_catdap/frontend/assets/style.css`

### 2.3 状態管理
- 方針: Dash標準の `dcc.Store` を利用
- 目的: 画面更新時の状態再現性確保（dataset/result/deepdive/theme など）

---

## 3. 過去の主要課題と解決

### 3.1 起動時クラッシュ（`html.Style` 関連）
- 症状: 起動時に Dash 側で属性エラー
- 原因: インラインスタイル注入方式の非互換
- 対応: スタイルを `assets/style.css` に集約

### 3.2 Deep Dive 表示欠落・固まり
- 症状: グラフ空表示や描画停止
- 原因: バックエンド返却形式とUI側期待形式の差分、binラベル生成不足
- 対応: 再構成ロジックを強化し、`bin_edges` からのラベル生成を追加

### 3.3 WebView + Plotly 描画不安定
- 症状: 描画が止まる、UIが固まる
- 原因: 初期コールバック内例外の連鎖
- 対応: 例外点を解消し、描画側は副作用を最小化

### 3.4 旧Streamlitテストの不整合
- 症状: `pytest` が旧UI向けテストで失敗
- 原因: 実装基盤がDashへ移行済み
- 対応: 旧テストを整理し、現行構成向けテストへ置換

---

## 4. 2026年2月 開発ログ（本スレッドの反映）

### 4.1 Global Interaction Network 空表示の修正
- `interaction_importances` 正規化を後方互換化
  - `Gain/gain` に加え `Pair_Score/pair_score` を受理
- 空データ時の表示を明示化
  - 「理由が分かる空表示」に改善
- `counts` 欠損時の `KeyError` を回避
- 関連テスト追加・強化
  - `tests/test_exporter.py` に回帰ケースを追加

### 4.2 WebGUIデザイン改善（Dashboard/Deep Dive）
- Dashboard上部KPIの情報設計を再構築
  - AIC比較カード統合
  - `Selected Features` 表示改善（`x / y features`）
  - `Model Type` に推定器/指標情報を明示
- Deep Diveの冗長ラベル削除と構造整理
  - 表示範囲が直感的に分かるパネル構成へ調整
- 配色・余白・タイポの統一
  - シアン基調のアクセントへ整理
  - 表ヘッダ中央揃え、メッセージ文字化け修正

### 4.3 HTMLレポートをWebGUIと整合
- カード構成・余白・見出し階層をWebGUI寄りに統一
- グラフ配色を共通パレットへ統一（棒/ヒートマップ/折れ線）
- ドロップダウン配色を黒基調で固定（白化防止）
- Bivariateヒートマップに軸名・余白・文字サイズを適正化
- 表ヘッダ中央揃え、冗長見出しの整理

### 4.4 テスト集約と自動化
- 手動スクリプトを `tests/` に移行
  - `scripts/run_tests_manual.py` → `tests/test_report_manual_migration.py`
  - `tests/manual_test_sqlite.py` → `tests/test_sqlite_integration.py`
- `integration` マーカー運用を導入（通常実行から分離）
  - `pyproject.toml` の pytest 設定を更新

### 4.5 PR作成
- ブランチ: `feature/dashboard-refactor`
- PR: `#11`
- 概要: WebGUI/HTML整合、可視化改善、テスト自動化移行

### 4.6 ドキュメント整理（README/HISTORY）
- READMEの古いフロントエンド表記を現行Dash構成に統一
  - guiオプションの説明を Dash + Plotly に修正
  - Desktop構成の説明を Dash frontend に修正
  - 旧 app_legacy.py 参照を削除
- HISTORYは履歴を保持したまま追記方式で更新（既存章は削除しない）


---

## 5. 現在のテスト運用

```bash
# 通常テスト（integration除外）
uv run pytest -q

# integrationテストのみ
uv run pytest -q -m integration

# exporterのカバレッジ確認
uv run coverage run -m pytest -q -p no:cacheprovider tests/test_exporter.py
uv run coverage report -m advanced_catdap/service/exporter.py
```

---

最終更新日: 2026-02-07
