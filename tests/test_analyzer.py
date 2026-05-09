"""感情分析のテスト."""

import pytest

from src.analyzer import SentimentAnalyzer, compute_sentiment_stats


@pytest.fixture(scope="session")
def analyzer():
    """SentimentAnalyzerをセッション全体で共有する（モデルロード1回）."""
    return SentimentAnalyzer()


class TestSentimentAnalyzer:
    """SentimentAnalyzerのテスト."""

    def test_analyze_returns_required_keys(self, analyzer):
        result = analyzer.analyze("テスト文章")
        assert "positive" in result
        assert "negative" in result
        assert "label" in result

    def test_analyze_label_is_valid(self, analyzer):
        result = analyzer.analyze("テスト")
        assert result["label"] in ("positive", "neutral", "negative")

    def test_analyze_scores_are_bounded(self, analyzer):
        result = analyzer.analyze("何かのテキスト")
        assert 0.0 <= result["positive"] <= 1.0
        assert 0.0 <= result["negative"] <= 1.0

    def test_positive_text_detected(self, analyzer):
        result = analyzer.analyze("この製品は素晴らしい、最高の体験でした")
        assert result["label"] == "positive"

    def test_negative_text_detected(self, analyzer):
        result = analyzer.analyze("最悪のサービスで二度と利用しない")
        assert result["label"] == "negative"

    def test_analyze_batch_returns_correct_count(self, analyzer):
        texts = ["良い天気", "悲しいニュース", "普通の日"]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3

    def test_analyze_batch_empty(self, analyzer):
        assert analyzer.analyze_batch([]) == []

    def test_analyze_batch_consistency_with_single(self, analyzer):
        text = "この映画は面白かった"
        single = analyzer.analyze(text)
        batch = analyzer.analyze_batch([text])[0]
        assert single["label"] == batch["label"]
        assert abs(single["positive"] - batch["positive"]) < 1e-6


class TestComputeSentimentStats:
    """compute_sentiment_statsのテスト."""

    def test_empty_results(self):
        stats = compute_sentiment_stats([])
        assert stats == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    def test_all_positive(self):
        results = [{"label": "positive"} for _ in range(5)]
        stats = compute_sentiment_stats(results)
        assert stats["positive"] == 1.0
        assert stats["neutral"] == 0.0
        assert stats["negative"] == 0.0

    def test_mixed_results(self):
        results = [
            {"label": "positive"},
            {"label": "negative"},
            {"label": "neutral"},
        ]
        stats = compute_sentiment_stats(results)
        assert abs(stats["positive"] - 1 / 3) < 0.01
        assert abs(stats["neutral"] - 1 / 3) < 0.01
        assert abs(stats["negative"] - 1 / 3) < 0.01
