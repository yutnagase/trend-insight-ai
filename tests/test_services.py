"""サービス層のテスト."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.services.history import load_history, save_history
from src.services.insight import generate_insight
from src.services.text_processor import create_tokenizer, extract_keywords


class TestExtractKeywords:
    """extract_keywordsのテスト."""

    def test_extracts_nouns(self):
        titles = ["人工知能の技術が進化している"]
        result = extract_keywords(titles, "テスト")
        words = [w for w, _ in result]
        assert "人工" in words or "知能" in words or "技術" in words

    def test_excludes_search_keyword(self):
        titles = ["生成AIの最新動向について"]
        result = extract_keywords(titles, "生成")
        words = [w for w, _ in result]
        assert "生成" not in words

    def test_excludes_stop_words(self):
        titles = ["ニュースの記事を配信する"]
        result = extract_keywords(titles, "テスト")
        words = [w for w, _ in result]
        assert "ニュース" not in words
        assert "記事" not in words

    def test_empty_input(self):
        result = extract_keywords([], "テスト")
        assert result == []

    def test_accepts_tokenizer_param(self):
        tok = create_tokenizer()
        titles = ["技術革新が進む"]
        result = extract_keywords(titles, "テスト", tokenizer=tok)
        assert len(result) > 0


class TestGenerateInsight:
    """generate_insightのテスト."""

    def test_single_source_positive(self):
        stats = {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
        result = generate_insight(stats)
        assert "ポジティブ" in result

    def test_gap_detected(self):
        news = {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
        bsky = {"positive": 0.1, "neutral": 0.1, "negative": 0.8}
        result = generate_insight(news, bsky_stats=bsky)
        assert "温度差" in result

    def test_no_gap(self):
        stats = {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
        result = generate_insight(stats, bsky_stats=stats, hatena_stats=stats)
        assert "一致" in result


class TestHistory:
    """履歴管理のテスト."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "history.json"
            with patch("src.services.history.HISTORY_PATH", tmp_path):
                save_history(
                    keyword="テスト",
                    news_results=[{"label": "positive", "title": "test"}],
                    sns_results=[],
                    hatena_results=[],
                    wordcloud_images={},
                    ai_report="テストレポート",
                )
                history = load_history()

            assert len(history) == 1
            assert history[0]["keyword"] == "テスト"
            assert history[0]["ai_report"] == "テストレポート"

    def test_load_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "nonexistent.json"
            with patch("src.services.history.HISTORY_PATH", tmp_path):
                history = load_history()
            assert history == []
