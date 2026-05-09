"""GoogleニュースRSSクライアント."""

import urllib.parse
from datetime import datetime

import feedparser
import requests
import structlog

from src.clients.base import BaseClient, ClientError
from src.models.article import Article

logger = structlog.get_logger(__name__)

DEFAULT_FETCH_COUNT = 30
REQUEST_TIMEOUT = 15


class GoogleNewsClient(BaseClient):
    """GoogleニュースRSSから記事を取得するクライアント."""

    def __init__(self, fetch_count: int = DEFAULT_FETCH_COUNT) -> None:
        self._fetch_count = fetch_count

    @property
    def source_name(self) -> str:
        return "news"

    def fetch(self, keyword: str) -> list[Article]:
        """GoogleニュースRSSからキーワード関連記事を取得する.

        Args:
            keyword: 検索キーワード.

        Returns:
            Article のリスト.

        Raises:
            ClientError: RSS取得・パースに失敗した場合.
        """
        encoded = urllib.parse.quote(keyword)
        ts = datetime.now().timestamp()
        url = (
            f"https://news.google.com/rss/search?q={encoded}&hl=ja&gl=JP"
            f"&ceid=JP:ja&_t={ts}"
        )

        log = logger.bind(phase="collect", source=self.source_name, keyword=keyword)

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except requests.RequestException as e:
            log.error("RSSリクエスト失敗", error=str(e))
            raise ClientError(f"Google News RSS request failed: {e}") from e
        except Exception as e:
            log.error("RSSパース失敗", error=str(e))
            raise ClientError(f"Google News RSS parse failed: {e}") from e

        if feed.bozo and not feed.entries:
            log.error("RSSエラー", error=str(feed.bozo_exception))
            raise ClientError(f"Google News RSS error: {feed.bozo_exception}")

        articles: list[Article] = []
        for entry in feed.entries[: self._fetch_count]:
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6])

            articles.append(
                Article(
                    title=entry.title,
                    url=entry.link,
                    source=self.source_name,
                    published_at=published_at,
                )
            )

        log.info("記事取得完了", article_count=len(articles))
        return articles
