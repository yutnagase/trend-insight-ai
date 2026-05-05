"""データ収集モジュール - Googleニュース・BlueSky・Redditからのデータ取得."""

import urllib.parse
from datetime import datetime

import feedparser
import praw
from atproto import Client as BskyClient

FETCH_COUNT_NEWS = 30
FETCH_COUNT_BSKY = 20
FETCH_COUNT_REDDIT = 20

# BlueSkyで除外するメディア・法人アカウントのキーワード
MEDIA_FILTER_KEYWORDS: set[str] = {
    "kyodonews", "yahoo", "nhk", "asahi", "mainichi", "yomiuri", "sankei",
    "nikkei", "jiji", "reuters", "afpbb", "cnn", "bbc", "tv-asahi",
    "tbs", "ntv", "fuji", "tokyonp", "chunichi", "hokkaido-np",
    "press", "news", "times", "journal", "media", "official",
}

# 日本語コンテンツが多いサブレディット
JAPANESE_SUBREDDITS: list[str] = [
    "newsokur", "japan_anime", "lowlevelaware", "BakaNewsJP",
    "steamr", "jisakupc", "japanlife", "japan",
]


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


def fetch_reddit_posts(
    keyword: str,
    client_id: str,
    client_secret: str,
    user_agent: str = "TrendInsightAI/1.0",
) -> list[dict[str, str]]:
    """Redditから日本語関連の投稿を検索して取得する.

    Args:
        keyword: 検索キーワード.
        client_id: Reddit API Client ID.
        client_secret: Reddit API Client Secret.
        user_agent: User-Agent文字列.

    Returns:
        タイトル・URL・著者・サブレディットを含む辞書のリスト.
    """
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    posts: list[dict[str, str]] = []

    # 日本語サブレディットから検索
    subreddit_str = "+".join(JAPANESE_SUBREDDITS)
    results = reddit.subreddit(subreddit_str).search(
        keyword, sort="new", time_filter="month", limit=FETCH_COUNT_REDDIT
    )

    for submission in results:
        posts.append({
            "title": submission.title,
            "url": f"https://www.reddit.com{submission.permalink}",
            "author": f"u/{submission.author.name}" if submission.author else "u/[deleted]",
            "subreddit": f"r/{submission.subreddit.display_name}",
        })

    # 日本語サブレディットで不足した場合、全体検索で補完
    if len(posts) < FETCH_COUNT_REDDIT:
        remaining = FETCH_COUNT_REDDIT - len(posts)
        existing_urls = {p["url"] for p in posts}
        results = reddit.subreddit("all").search(
            keyword, sort="new", time_filter="month", limit=remaining * 2
        )
        for submission in results:
            url = f"https://www.reddit.com{submission.permalink}"
            if url in existing_urls:
                continue
            # 日本語タイトルを含む投稿のみ
            if not _contains_japanese(submission.title):
                continue
            posts.append({
                "title": submission.title,
                "url": url,
                "author": f"u/{submission.author.name}" if submission.author else "u/[deleted]",
                "subreddit": f"r/{submission.subreddit.display_name}",
            })
            if len(posts) >= FETCH_COUNT_REDDIT:
                break

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


def _contains_japanese(text: str) -> bool:
    """テキストに日本語文字が含まれるか判定する.

    Args:
        text: 判定対象テキスト.

    Returns:
        日本語文字を含むならTrue.
    """
    return any("\u3040" <= c <= "\u9fff" for c in text)
