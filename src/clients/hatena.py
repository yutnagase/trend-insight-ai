"""はてなブックマーククライアント."""

import time
import urllib.parse

import feedparser
import requests
import structlog

from src.clients.base import BaseClient, ClientError
from src.models.article import Article

logger = structlog.get_logger(__name__)

DEFAULT_TOP_N = 5
ENTRY_API = "https://b.hatena.ne.jp/entry/json/"
SEARCH_RSS = "https://b.hatena.ne.jp/search/text"
REQUEST_TIMEOUT = 10


class HatenaClient(BaseClient):
    """はてなブックマークからコメントを収集するクライアント."""

    def __init__(self, top_n: int = DEFAULT_TOP_N, min_users: int = 3) -> None:
        self._top_n = top_n
        self._min_users = min_users

    @property
    def source_name(self) -> str:
        return "hatena"

    def fetch(self, keyword: str) -> list[Article]:
        """キーワードで検索し、上位記事のコメントをArticleとして返す.

        Args:
            keyword: 検索キーワード.

        Returns:
            各コメントを1つのArticleとして正規化したリスト.

        Raises:
            ClientError: 検索またはコメント取得に失敗した場合.
        """
        entries = self._search_entries(keyword)
        if not entries:
            return []

        entries.sort(key=lambda x: x["bookmark_count"], reverse=True)
        top_entries = entries[: self._top_n]

        articles: list[Article] = []
        for entry in top_entries:
            time.sleep(0.5)
            comments = self._get_comments(entry["url"])
            for comment in comments:
                articles.append(
                    Article(
                        title=comment["comment"],
                        url=entry["url"],
                        source=self.source_name,
                        author=comment["user"],
                        metadata={
                            "entry_title": entry["title"],
                            "bookmark_count": entry["bookmark_count"],
                            "timestamp": comment.get("timestamp", ""),
                        },
                    )
                )
        return articles

    def fetch_with_entries(self, keyword: str) -> tuple[list[Article], list[dict]]:
        """コメントArticleリストと元記事情報の両方を返す（UI表示用）.

        safe_fetch同様、例外発生時は空リストを返しログに記録する。

        Args:
            keyword: 検索キーワード.

        Returns:
            (コメントArticleリスト, 記事ごとのメタ情報リスト).
        """
        try:
            entries = self._search_entries(keyword)
        except ClientError as e:
            logger.warning(
                "fetch_with_entries失敗",
                phase="collect",
                source=self.source_name,
                keyword=keyword,
                error=str(e),
            )
            return [], []

        if not entries:
            return [], []

        entries.sort(key=lambda x: x["bookmark_count"], reverse=True)
        top_entries = entries[: self._top_n]

        articles: list[Article] = []
        entry_data: list[dict] = []

        for entry in top_entries:
            time.sleep(0.5)
            comments = self._get_comments(entry["url"])
            if comments:
                entry_data.append(entry)
                for comment in comments:
                    articles.append(
                        Article(
                            title=comment["comment"],
                            url=entry["url"],
                            source=self.source_name,
                            author=comment["user"],
                            metadata={
                                "entry_title": entry["title"],
                                "bookmark_count": entry["bookmark_count"],
                                "timestamp": comment.get("timestamp", ""),
                            },
                        )
                    )
        return articles, entry_data

    def _search_entries(self, keyword: str) -> list[dict]:
        """はてなブックマーク検索RSSでエントリを取得する."""
        quoted_keyword = f'"{keyword}"'
        search_url = (
            f"{SEARCH_RSS}?q={urllib.parse.quote(quoted_keyword)}&users={self._min_users}&mode=rss"
        )

        try:
            resp = requests.get(search_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except requests.RequestException as e:
            raise ClientError(f"Hatena search request failed: {e}") from e
        except Exception as e:
            raise ClientError(f"Hatena search RSS parse failed: {e}") from e

        entries: list[dict] = []
        for entry in feed.entries:
            if keyword not in entry.title:
                continue
            bookmark_count = int(entry.get("hatena_bookmarkcount", 0))
            entries.append(
                {
                    "title": entry.title,
                    "url": entry.link,
                    "bookmark_count": bookmark_count,
                }
            )
        return entries

    def _get_comments(self, url: str) -> list[dict[str, str]]:
        """指定URLのはてなブックマークコメントを取得する."""
        api_url = f"{ENTRY_API}?url={urllib.parse.quote(url)}"
        try:
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        if not data or "bookmarks" not in data:
            return []

        comments: list[dict[str, str]] = []
        for bookmark in data["bookmarks"]:
            comment = bookmark.get("comment", "").strip()
            if comment:
                comments.append(
                    {
                        "user": bookmark.get("user", ""),
                        "comment": comment,
                        "timestamp": bookmark.get("timestamp", ""),
                    }
                )
        return comments
