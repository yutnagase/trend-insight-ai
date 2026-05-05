"""データ収集モジュール - GoogleニュースRSSおよびBlueSkyからの記事/投稿取得."""

import urllib.parse
from datetime import datetime

import feedparser
from atproto import Client as BskyClient

FETCH_COUNT_NEWS = 30
FETCH_COUNT_BSKY = 20

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
        handle: BlueSkyのハンドル（例: user.bsky.social）.
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
        # メディアアカウントを除外
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


def _is_media_account(handle: str) -> bool:
    """ハンドルがメディア・法人アカウントか判定する.

    Args:
        handle: BlueSkyハンドル.

    Returns:
        メディアアカウントならTrue.
    """
    handle_lower = handle.lower()
    return any(kw in handle_lower for kw in MEDIA_FILTER_KEYWORDS)
