"""クライアント層のテスト."""

from unittest.mock import MagicMock, patch

import pytest

from src.clients.base import ClientError
from src.clients.bluesky import BlueskyClient
from src.clients.google_news import GoogleNewsClient
from src.clients.hatena import HatenaClient
from src.models.article import Article


class TestGoogleNewsClient:
    """GoogleNewsClientのテスト."""

    def test_source_name(self):
        client = GoogleNewsClient()
        assert client.source_name == "news"

    @patch("src.clients.google_news.requests.get")
    def test_fetch_returns_articles(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
        <channel>
            <item>
                <title>テスト記事タイトル</title>
                <link>https://example.com/article1</link>
            </item>
            <item>
                <title>テスト記事2</title>
                <link>https://example.com/article2</link>
            </item>
        </channel>
        </rss>"""
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = GoogleNewsClient(fetch_count=5)
        articles = client.fetch("テスト")

        assert len(articles) == 2
        assert all(isinstance(a, Article) for a in articles)
        assert articles[0].title == "テスト記事タイトル"
        assert articles[0].source == "news"

    @patch("src.clients.google_news.requests.get")
    def test_fetch_timeout_raises_client_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.Timeout("timeout")

        client = GoogleNewsClient()
        with pytest.raises(ClientError):
            client.fetch("テスト")

    @patch("src.clients.google_news.requests.get")
    def test_safe_fetch_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("network error")

        client = GoogleNewsClient()
        result = client.safe_fetch("テスト")
        assert result == []


class TestBlueskyClient:
    """BlueskyClientのテスト."""

    def test_source_name(self):
        client = BlueskyClient()
        assert client.source_name == "bluesky"

    def test_is_configured_false_when_no_credentials(self):
        client = BlueskyClient()
        assert client.is_configured is False

    def test_is_configured_true_when_credentials_set(self):
        client = BlueskyClient(handle="test.bsky.social", app_password="pass")
        assert client.is_configured is True

    def test_media_account_filtering(self):
        assert BlueskyClient._is_media_account("nhk-news.bsky.social") is True
        assert BlueskyClient._is_media_account("random-user.bsky.social") is False

    def test_safe_fetch_returns_empty_on_auth_failure(self):
        client = BlueskyClient(handle="invalid", app_password="invalid")
        result = client.safe_fetch("テスト")
        assert result == []


class TestHatenaClient:
    """HatenaClientのテスト."""

    def test_source_name(self):
        client = HatenaClient()
        assert client.source_name == "hatena"

    @patch("src.clients.hatena.requests.get")
    def test_fetch_with_entries_returns_empty_on_no_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel></channel></rss>"""
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = HatenaClient()
        articles, entries = client.fetch_with_entries("存在しないキーワード")
        assert articles == []
        assert entries == []

    @patch("src.clients.hatena.requests.get")
    def test_fetch_with_entries_handles_timeout(self, mock_get):
        import requests

        mock_get.side_effect = requests.Timeout("timeout")

        client = HatenaClient()
        articles, entries = client.fetch_with_entries("テスト")
        assert articles == []
        assert entries == []
