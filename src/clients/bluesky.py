"""BlueSky AT Protocolクライアント."""

from atproto import Client as BskyClient

from src.clients.base import BaseClient, ClientError
from src.models.article import Article

DEFAULT_FETCH_COUNT = 20

# メディア・法人アカウント除外キーワード
MEDIA_FILTER_KEYWORDS: set[str] = {
    "kyodonews", "yahoo", "nhk", "asahi", "mainichi", "yomiuri", "sankei",
    "nikkei", "jiji", "reuters", "afpbb", "cnn", "bbc", "tv-asahi",
    "tbs", "ntv", "fuji", "tokyonp", "chunichi", "hokkaido-np",
    "press", "news", "times", "journal", "media", "official",
}


class BlueskyClient(BaseClient):
    """BlueSkyから投稿を検索・取得するクライアント."""

    def __init__(
        self,
        handle: str | None = None,
        app_password: str | None = None,
        fetch_count: int = DEFAULT_FETCH_COUNT,
    ) -> None:
        self._handle = handle
        self._app_password = app_password
        self._fetch_count = fetch_count

    @property
    def source_name(self) -> str:
        return "bluesky"

    @property
    def is_configured(self) -> bool:
        """認証情報が設定されているか."""
        return bool(self._handle and self._app_password)

    def fetch(self, keyword: str) -> list[Article]:
        """BlueSkyから公開投稿を検索して取得する（メディアアカウント除外）.

        Args:
            keyword: 検索キーワード.

        Returns:
            Article のリスト.

        Raises:
            ClientError: API呼び出しに失敗した場合.
        """
        try:
            client = BskyClient()
            if self._handle and self._app_password:
                client.login(self._handle, self._app_password)

            response = client.app.bsky.feed.search_posts(
                params={"q": keyword, "limit": self._fetch_count, "lang": "ja"}
            )
        except Exception as e:
            raise ClientError(f"BlueSky API error: {e}") from e

        articles: list[Article] = []
        for post in response.posts:
            author = post.author.handle
            if self._is_media_account(author):
                continue

            rkey = post.uri.split("/")[-1]
            post_url = f"https://bsky.app/profile/{author}/post/{rkey}"

            articles.append(
                Article(
                    title=post.record.text,
                    url=post_url,
                    source=self.source_name,
                    author=f"@{author}",
                )
            )
        return articles

    @staticmethod
    def _is_media_account(handle: str) -> bool:
        handle_lower = handle.lower()
        return any(kw in handle_lower for kw in MEDIA_FILTER_KEYWORDS)
