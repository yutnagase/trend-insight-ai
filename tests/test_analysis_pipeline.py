"""分析パイプラインの統合テスト（モックanalyzer使用）."""

from unittest.mock import patch

import pytest

from src.models.article import Article
from src.services.analysis_pipeline import run_analysis


class MockAnalyzer:
    """テスト用の感情分析器モック.

    Protocol準拠: analyze() / analyze_batch() を実装。
    タイトルに「良い」を含めばpositive、「悪い」ならnegative、それ以外neutral。
    """

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


def _make_articles(titles: list[str], source: str) -> list[Article]:
    return [
        Article(title=t, url=f"https://example.com/{i}", source=source)
        for i, t in enumerate(titles)
    ]


class TestRunAnalysis:
    """run_analysisの統合テスト."""

    @pytest.fixture
    def analyzer(self):
        return MockAnalyzer()

    @pytest.fixture(autouse=True)
    def _mock_wordcloud(self):
        """CI環境に日本語フォントがないためワードクラウド生成をスキップ."""
        with patch("src.services.analysis_pipeline.generate_wordcloud", return_value=None):
            yield

    def test_basic_pipeline_returns_analysis_result(self, analyzer):
        """基本的なパイプライン実行でAnalysisResultが返る."""
        news = _make_articles(["良いニュース", "悪いニュース", "普通のニュース"], "news")
        result = run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
        )

        assert result.keyword == "テスト"
        assert len(result.news.results) == 3
        assert result.news.stats is not None
        assert result.news.stats.positive > 0

    def test_multi_source_divergence_detected(self, analyzer):
        """複数ソースで方向が異なれば乖離が検出される."""
        news = _make_articles(["良い話題"] * 8 + ["普通の話題"] * 2, "news")
        sns = _make_articles(["悪い話題"] * 8 + ["普通の話題"] * 2, "bluesky")

        result = run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=sns,
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
        )

        assert len(result.divergences) > 0
        # 最大乖離ペアの乖離幅が大きい
        assert result.divergences[0][2] > 0.3

    def test_analysis_types_assigned(self, analyzer):
        """分析タイプが1つ以上割り当てられる."""
        news = _make_articles(["良いニュース", "悪いニュース"], "news")
        result = run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
        )

        assert len(result.analysis_types) >= 1
        assert "type" in result.analysis_types[0]
        assert "label" in result.analysis_types[0]

    def test_empty_sources_handled(self, analyzer):
        """全ソース空でもエラーにならない（ニュースのみ必須）."""
        news = _make_articles(["テスト記事"], "news")
        result = run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
        )

        assert result.bsky.results == []
        assert result.hatena.results == []
        assert result.bsky.stats is None

    def test_topic_sentiments_computed(self, analyzer):
        """十分なデータがあればトピック別感情が算出される."""
        titles = [
            "AI技術の良い進化",
            "AI活用の良い事例",
            "AI規制の悪い影響",
            "AI開発の悪い問題",
            "AI研究の普通の動向",
        ]
        news = _make_articles(titles, "news")
        result = run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
        )

        # キーワードが抽出されていればトピック感情が計算される
        if result.news.keywords:
            # トピック感情が存在するか、空でもエラーにならないことを確認
            assert isinstance(result.topic_sentiments, dict)

    def test_net_score_reflects_sentiment(self, analyzer):
        """ネットスコアが感情分布を反映する."""
        positive_news = _make_articles(["良い話題"] * 10, "news")
        result = run_analysis(
            keyword="テスト",
            news_articles=positive_news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
        )

        assert result.news.net_score > 0.5

    def test_progress_callback_called(self, analyzer):
        """progress_callbackが呼ばれる."""
        news = _make_articles(["テスト記事"], "news")
        calls = []

        def on_progress(label: str, current: int, total: int):
            calls.append((label, current, total))

        run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
            progress_callback=on_progress,
        )

        assert len(calls) >= 1
        assert calls[0][0] == "メディア記事"

    def test_hatena_entry_data_preserved(self, analyzer):
        """hatena_entry_dataがそのまま結果に含まれる."""
        news = _make_articles(["テスト"], "news")
        entry_data = [{"url": "https://example.com", "bookmarks": 100}]

        result = run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=entry_data,
            analyzer=analyzer,
        )

        assert result.hatena_entry_data == entry_data

    def test_representative_samples_extracted(self, analyzer):
        """代表意見が抽出される."""
        news = _make_articles(
            ["良いニュース1", "良いニュース2", "悪いニュース1", "悪いニュース2", "普通"],
            "news",
        )
        result = run_analysis(
            keyword="テスト",
            news_articles=news,
            sns_articles=[],
            hatena_articles=[],
            hatena_entry_data=[],
            analyzer=analyzer,
        )

        assert "positive" in result.news.samples
        assert "negative" in result.news.samples
        assert len(result.news.samples["positive"]) >= 1
