# テスト戦略

## 方針

Protocol + DIアーキテクチャにより、LLM・ネットワーク・GPU・Streamlitに依存せずコアロジックを検証できる構成を採用しています。

## テストピラミッド

```
        ┌─────────────┐
        │  E2E (手動)  │  Streamlit画面操作
        ├─────────────┤
        │  統合テスト   │  パイプライン全体（モックanalyzer注入）
        ├─────────────┤
        │ ユニットテスト │  純粋関数・個別サービス
        └─────────────┘
```

| レベル | 対象 | 外部依存 | 実行速度 |
|--------|------|----------|----------|
| ユニット | analysis_type, topic_sentiment, insight, text_processor | なし | 高速 |
| 統合 | analysis_pipeline（モックanalyzer経由） | なし | 高速 |
| スロー | SentimentAnalyzer（実BERTモデル） | モデルファイル | 低速 |
| E2E | 画面操作 | 全依存 | 手動 |

## テスト実行方法

```bash
# 高速テスト（日常開発用）— BERTモデル不要
uv run pytest -m "not slow"

# カバレッジ付き
uv run pytest -m "not slow" --cov --cov-report=term-missing

# HTML形式のカバレッジレポート
uv run pytest -m "not slow" --cov --cov-report=html
# → htmlcov/index.html をブラウザで確認

# 全テスト（BERTモデルロード含む）
uv run pytest

# 特定テストファイルのみ
uv run pytest tests/test_analysis_type.py -v
```

## テストファイル構成

```
tests/
├── test_analysis_type.py       # 分析タイプ判定（5パターン全条件網羅）
├── test_topic_sentiment.py     # トピック別感情集約（境界値テスト）
├── test_analysis_pipeline.py   # パイプライン統合テスト（モックanalyzer）
├── test_clients.py             # データ収集クライアント（HTTPモック）
├── test_services.py            # サービス層（キーワード抽出・履歴・インサイト）
└── test_analyzer.py            # BERTアンサンブル実モデルテスト (@slow)
```

## モック戦略

### SentimentAnalyzerProtocol のモック

パイプラインテストでは、BERTモデルの代わりにルールベースのモックを注入します。

```python
class MockAnalyzer:
    """タイトルの内容で感情を決定する軽量モック."""

    def analyze(self, text: str) -> dict[str, float | str]:
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str]) -> list[dict[str, float | str]]:
        results = []
        for text in texts:
            if "良い" in text:
                results.append({"positive": 0.8, "negative": 0.1, "label": "positive"})
            elif "悪い" in text:
                results.append({"positive": 0.1, "negative": 0.8, "label": "negative"})
            else:
                results.append({"positive": 0.3, "negative": 0.3, "label": "neutral"})
        return results
```

このモックはProtocolに準拠しているため、型チェック（mypy）でも問題なく通ります。

### HTTPクライアントのモック

`unittest.mock.patch` でrequests.getをモックし、固定レスポンスを返します。

```python
@patch("src.clients.google_news.requests.get")
def test_fetch_returns_articles(self, mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = """<rss>...</rss>"""
    mock_get.return_value = mock_resp
    # ...
```

## カバレッジ方針

### 計測対象

コアロジック（テスト可能かつビジネス価値の高い部分）:

- `src/services/` — 分析パイプライン、タイプ判定、トピック感情、乖離検出
- `src/clients/` — データ収集（HTTPモックで検証）
- `src/models/` — データモデル
- `src/analyzer.py` — 感情分析（スロー実行時のみ）

### 計測対象外

外部依存が強く、モック化のコストが高い部分:

- `src/ui/` — Streamlit描画（UIテストフレームワーク未導入）
- `src/orchestrator.py` — Streamlit直接呼び出し（UI通知の抽象化が前提）
- `src/reporter.py` — LLM推論（出力の非決定性）
- `src/adapters.py` — 薄いラッパー（統合テストで間接的にカバー）
- `src/logging_config.py` — 設定のみ

### 閾値

- 計測対象範囲で **70%以上** を維持（`pyproject.toml` の `fail_under` で強制）
- CIで `pytest --cov` を実行し、閾値未満でビルド失敗とする

## テスト設計の原則

1. **純粋関数を優先的にテスト** — 入力→出力が決定的な関数は最もテストしやすく、バグの温床になりやすい
2. **DIの恩恵を最大化** — Protocolに依存する箇所はモック注入で外部依存を排除
3. **スローテストの分離** — `@pytest.mark.slow` でBERTモデル依存テストを分離し、日常開発を高速に保つ
4. **境界値を重視** — min_count、閾値判定、空入力など、エッジケースを明示的にテスト
5. **テストが仕様書になる** — テストクラス名・メソッド名で「何が保証されているか」を読み取れるようにする

## 今後の拡張候補

- **オーケストレーターのテスト** — UI通知部分を抽象化（NotifierProtocol等）すれば、フロー制御のテストが可能に
- **プロパティベーステスト** — hypothesis等で感情スコアの値域不変条件を検証
- **スナップショットテスト** — LLMプロンプト生成部分の回帰検出
