# AGENTS.md — Antigravity / data analysis library + analysis workflow

## 0) Mission
このリポジトリでは次の2つを両立する：
1) データ分析用ライブラリ（advanced_catdap/）の開発・品質維持
2) 分析作業（analysis/）の再現性ある実行・レポート化

最優先：再現性（reproducibility）と安全性（no destructive ops）。

---

## 1) Ground rules (Must)
- 破壊的コマンドは禁止（例：`rm -rf`, `del /s`, `format`, ワイルドカード削除、パス誤りの可能性がある削除）。
- 既存データの上書きは原則禁止。必要なら `analysis/reports/` に新規出力し、入力データは不変に扱う。
- 機密や個人情報（PII）、APIキー、トークン、.env、認証情報を出力・コミットしない。
- 変更は小さく、常に `git diff` がレビュー可能な粒度にする。

---

## 2) Repo map
- `advanced_catdap/` : ライブラリ本体
- `tests/` : 単体テスト（追加・変更には原則テストを追加）
- `analysis/` : 分析（notebooks/scripts）とレポート出力（reports）
- `analysis/data/` : ローカルデータ置き場（git管理しない前提）
- `examples/` : ドキュメント（使い方・設計方針・例）

---

## 3) Environment & commands (Auto-detect)
まずリポジトリを見て、該当する仕組みのコマンドだけ使う：
- `pyproject.toml` + `uv.lock` がある → uv を優先
- `pyproject.toml` + `poetry.lock` がある → poetry を使う
- `requirements.txt` がある → venv + pip
- R がある場合：`renv.lock` があれば renv を使う

標準コマンド（存在するものだけ）：
- Install: `uv sync` / `poetry install` / `pip install -r requirements.txt`
- Test: `pytest -q`（または `make test`）
- Lint/Format: `ruff` / `black` / `isort`（または `make lint` / `make fmt`）
- Type check: `pyright` / `mypy`（採用している場合のみ）

---

## 4) Library development policy (advanced_catdap/)
- 公開APIは安定重視：破壊的変更（breaking change）は事前に明記し、影響範囲を列挙する。
- 追加する関数/クラスは docstring を付け、最小の例を `examples/` または `analysis/scripts/` に置く。
- パフォーマンスに関わる処理はベンチ/計測手段（簡易でも可）を添える。
- 新機能・バグ修正には原則テストを付ける（`tests/`）。

---

## 5) Analysis workflow policy (analysis/)
### 5.1 Outputs
- 図表：`analysis/reports/figures/`
- 集計結果：`analysis/reports/tables/`
- レポート：`analysis/reports/`（Markdown/HTML/PDF 等、採用形式に合わせる）

### 5.2 Reproducibility
- 分析は「入口（入力）→処理→出口（出力）」を明示する。
- 乱数を使う場合は seed を固定し、環境差（OS/CPU）で結果が変わり得る点を注記。
- notebook を使う場合でも、最終的な再現実行は `analysis/scripts/` に落としておく（可能な範囲で）。

### 5.3 Data handling
- 生データは原則コミットしない（必要ならサンプル/モック/統計的に匿名化した小データのみ）。
- 大きいファイルはパスと取得手順だけ記載し、出力にも生データを埋め込まない。

---

## 6) Definition of done (共通)
作業完了の条件：
- 変更点の要約（何を、なぜ）
- `tests` が通り、カバー率が90%以上であることを確認する（最低でも80%以上または「なぜ通せないか」と代替検証）
- `advanced_catdap/` の変更なら、使用例を `examples/`に追加する
- 分析タスクなら、再現手順（コマンド）と出力先（reports配下）が明記されている

---

## 7) Ask-first changes (Stop & ask)
- 依存関係の追加・更新（特に解析系パッケージは肥大化しやすい）
- CI/リリース手順の変更
- データスキーマ変更、DB変更、クラウド資源を触る操作

---

## 8) Git/GitHub rules
- Never push to main (PR only). No force-push to shared branches.
- Do not merge PRs or create releases/tags without explicit instruction.
- Keep PRs small and reviewable. One logical change per PR.
- PR description must include: summary, scope/impact, how to test, and outputs path (analysis reports).
- Ask first before: adding dependencies, changing CI/workflows, changing permissions, breaking API changes.
- Never include secrets/PII in commits, issues, PRs, logs, or artifacts.