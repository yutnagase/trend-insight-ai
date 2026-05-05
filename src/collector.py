"""データ収集モジュール - Googleニュース・BlueSky・はてなブックマークからのデータ取得."""

import time
import urllib.parse
from datetime import datetime

import feedparser
import requests
from atproto import Client as BskyClient

FETCH_COUNT_NEWS = 30
FETCH_COUNT_BSKY = 20
HATENA_TOP_N = 5  # コメント取得対象の上位記事数

# BlueSkyで除外するメディア・法人アカウントのキーワード
MEDIA_FILTER_KEYWORDS: set[str] = {
    "kyodonews", "yahoo", "nhk", "asahi", "mainichi", "yomiuri", "sankei",
    "nikkei", "jiji", "reuters", "afpbb", "cnn", "bbc", "tv-asahi",
    "tbs", "ntv", "fuji", "tokyonp", "chunichi", "hokkaido-np",
    "press", "news", "times", "journal", "media", "official",
}


def fetch_news_articles(keyword: str) -> list[dict[str, str]]:
    """GoogleニュースRSSからキーワード関連記事を取得する.

    Args:
        keyword: 検索キーワード.

    Returns:
        タイトルとURLを含む辞書のリスト.
    """
    encoded = urllib.parse.quote(keyword)
    ts = datetime.now().timestamp()
    url = (
        f"https://news.google.com/rss/search?q={encoded}&hl=ja&gl=JP"
        f"&ceid=JP:ja&_t={ts}"
    )
    feed = feedparser.parse(url)
    return [
        {"title": entry.title, "url": entry.link}
        for entry in feed.entries[:FETCH_COUNT_NEWS]
    ]


def fetch_bluesky_posts(
    keyword: str, handle: str | None = None, app_password: str | None = None
) -> list[dict[str, str]]:
    """BlueSkyから公開投稿を検索して取得する（メディアアカウント除外）.

    Args:
        keyword: 検索キーワード.
        handle: BlueSkyのハンドル.
        app_password: アプリパスワード.

    Returns:
        テキスト・URL・著者を含む辞書のリスト.
    """
    client = BskyClient()
    if handle and app_password:
        client.login(handle, app_password)

    response = client.app.bsky.feed.search_posts(
        params={"q": keyword, "limit": FETCH_COUNT_BSKY, "lang": "ja"}
    )

    posts: list[dict[str, str]] = []
    for post in response.posts:
        author = post.author.handle
        if _is_media_account(author):
            continue
        uri = post.uri
        parts = uri.split("/")
        rkey = parts[-1]
        post_url = f"https://bsky.app/profile/{author}/post/{rkey}"
        posts.append({
            "title": post.record.text,
            "url": post_url,
            "author": f"@{author}",
        })
    return posts


class HatenaCollector:
    """はてなブックマークAPIからコメントを収集するクラス."""

    ENTRY_API = "https://b.hatena.ne.jp/entry/json/"
    SEARCH_RSS = "https://b.hatena.ne.jp/search/text"

    def search_entries(self, keyword: str, min_users: int = 3) -> list[dict]:
        """はてなブックマーク検索RSSでキーワードに関連するエントリを取得する.

        Args:
            keyword: 検索キーワード.
            min_users: 最低ブックマーク数.

        Returns:
            タイトル・URL・ブックマーク数を含む辞書のリスト.
        """
        # フレーズ検索で正確にマッチさせる
        quoted_keyword = f'"{keyword}"'
        search_url = (
            f"{self.SEARCH_RSS}?q={urllib.parse.quote(quoted_keyword)}"
            f"&users={min_users}&mode=rss"
        )
        feed = feedparser.parse(search_url)
        entries: list[dict] = []
        for entry in feed.entries:
            # タイトルにキーワードが含まれるもののみ採用
            if keyword not in entry.title:
                continue
            bookmark_count = int(entry.get("hatena_bookmarkcount", 0))
            entries.append({
                "title": entry.title,
                "url": entry.link,
                "bookmark_count": bookmark_count,
            })
        return entries

    def get_comments(self, url: str) -> list[dict[str, str]]:
        """指定URLのはてなブックマークコメントを取得する.

        Args:
            url: 対象記事のURL.

        Returns:
            ユーザー名・コメント・タイムスタンプを含む辞書のリスト.
        """
        api_url = f"{self.ENTRY_API}?url={urllib.parse.quote(url)}"
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
                comments.append({
                    "user": bookmark.get("user", ""),
                    "comment": comment,
                    "timestamp": bookmark.get("timestamp", ""),
                })
        return comments

    def fetch_comments_by_keyword(
        self, keyword: str, top_n: int = HATENA_TOP_N
    ) -> list[dict]:
        """キーワードでブックマーク検索し、上位記事のコメントを収集する.

        Args:
            keyword: 検索キーワード.
            top_n: コメント取得対象の上位記事数.

        Returns:
            記事情報とコメントを含む辞書のリスト.
        """
        entries = self.search_entries(keyword, min_users=3)
        if not entries:
            return []

        # ブックマーク数でソートし上位N件を選定
        entries.sort(key=lambda x: x["bookmark_count"], reverse=True)
        top_entries = entries[:top_n]

        results: list[dict] = []
        for entry in top_entries:
            time.sleep(0.5)  # サーバー負荷軽減
            comments = self.get_comments(entry["url"])
            if comments:
                results.append({
                    "title": entry["title"],
                    "url": entry["url"],
                    "bookmark_count": entry["bookmark_count"],
                    "comments": comments,
                })
        return results


def _is_media_account(handle: str) -> bool:
    """ハンドルがメディア・法人アカウントか判定する.

    Args:
        handle: BlueSkyハンドル.

    Returns:
        メディアアカウントならTrue.
    """
    handle_lower = handle.lower()
    return any(kw in handle_lower for kw in MEDIA_FILTER_KEYWORDS)
