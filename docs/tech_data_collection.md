# データ収集 — feedparser / atproto / requests

## データ収集の概要

本プロジェクトでは3つの情報源からデータを集めています。それぞれ取得方法が異なるため、用途に合ったライブラリを使い分けています。

| 情報源             | ライブラリ | プロトコル              |
| ------------------ | ---------- | ----------------------- |
| Googleニュース     | feedparser | RSS（XML）              |
| BlueSky            | atproto    | AT Protocol（REST API） |
| はてなブックマーク | requests   | HTTP（JSON API）        |

## Googleニュース — feedparser

### RSSとは

RSS（Really Simple Syndication）は、Webサイトの更新情報を配信するためのXML形式のフォーマットです。ニュースサイトやブログが新着記事の一覧をRSSフィードとして公開しており、プログラムから簡単に記事タイトルとURLを取得できます。

GoogleニュースもRSSフィードを提供しているので、APIキー不要・認証不要で最新ニュースを取得できます。

### feedparserの基本

```python
import feedparser

# GoogleニュースのRSSを取得
url = "https://news.google.com/rss/search?q=生成AI&hl=ja&gl=JP&ceid=JP:ja"
feed = feedparser.parse(url)

# 記事一覧を表示
for entry in feed.entries[:5]:
    print(f"{entry.title}")
    print(f"  {entry.link}")
```

`feedparser.parse()` にURLを渡すだけで、RSSフィードをパースしてPythonのオブジェクトとして扱えます。`feed.entries` に記事の配列が入っています。

### 本プロジェクトでの実装

```python
import urllib.parse
import feedparser

def fetch_news_articles(keyword):
    encoded = urllib.parse.quote(keyword)
    url = (
        f"https://news.google.com/rss/search?q={encoded}"
        f"&hl=ja&gl=JP&ceid=JP:ja"
    )
    feed = feedparser.parse(url)
    return [
        {"title": entry.title, "url": entry.link}
        for entry in feed.entries[:30]
    ]
```

`urllib.parse.quote()` でキーワードをURLエンコードし、日本語をそのままURLに含められるようにしています。

## BlueSky — atproto

### AT Protocolとは

AT Protocol（Authenticated Transfer Protocol）は、BlueSkyが開発した分散型SNSのためのプロトコルです。Twitterと違い、オープンな仕様で誰でもクライアントを作れます。

`atproto` ライブラリはこのプロトコルのPython SDKで、投稿の検索・取得・投稿などができます。

### 基本的な使い方

```python
from atproto import Client

client = Client()
client.login("yourname.bsky.social", "your-app-password")

# キーワードで投稿を検索
response = client.app.bsky.feed.search_posts(
    params={"q": "生成AI", "limit": 20, "lang": "ja"}
)

for post in response.posts:
    print(f"@{post.author.handle}: {post.record.text[:50]}")
```

### メディアアカウントの除外

BlueSkyで検索すると、ニュースメディアの公式アカウントの投稿も混ざります。本プロジェクトでは「個人の生の声」を拾いたいので、メディアアカウントをフィルタリングしています。

```python
MEDIA_FILTER_KEYWORDS = {
    "kyodonews", "yahoo", "nhk", "asahi", "mainichi",
    "nikkei", "reuters", "press", "news", "official",
}

def _is_media_account(handle):
    handle_lower = handle.lower()
    return any(kw in handle_lower for kw in MEDIA_FILTER_KEYWORDS)
```

ハンドル名に「news」「press」などが含まれていたらメディアとみなして除外します。完璧ではありませんが、大半のメディアアカウントはこれで弾けます。

### 投稿URLの組み立て

BlueSkyのAPIが返すのは内部URI（`at://did:plc:xxx/app.bsky.feed.post/yyy`）なので、ブラウザで開けるURLに変換しています。

```python
uri = post.uri  # at://did:plc:xxx/app.bsky.feed.post/rkey
rkey = uri.split("/")[-1]
post_url = f"https://bsky.app/profile/{post.author.handle}/post/{rkey}"
```

## はてなブックマーク — requests

### はてなブックマークAPIの特徴

はてなブックマークは認証不要の公開APIを提供しています。特定のURLに対するブックマーク数やコメントを取得できます。

本プロジェクトでは2つのAPIを組み合わせて使っています。

1. **検索RSS**

- キーワードでブックマークされた記事を検索

2. **エントリーJSON API**

- 特定記事のコメント一覧を取得

### 検索RSSで記事を探す

```python
import feedparser
import urllib.parse

keyword = "生成AI"
quoted = f'"{keyword}"'  # フレーズ検索
url = (
    f"https://b.hatena.ne.jp/search/text"
    f"?q={urllib.parse.quote(quoted)}&users=3&mode=rss"
)
feed = feedparser.parse(url)

for entry in feed.entries[:5]:
    bookmarks = entry.get("hatena_bookmarkcount", 0)
    print(f"[{bookmarks}users] {entry.title}")
```

`users=3` で最低3ブックマーク以上の記事に絞り、ノイズを減らしています。フレーズ検索（`""`で囲む）にすることで、キーワードが完全一致する記事だけを取得します。

### コメントを取得する

```python
import requests

article_url = "https://example.com/article/123"
api_url = f"https://b.hatena.ne.jp/entry/json/?url={urllib.parse.quote(article_url)}"

resp = requests.get(api_url, timeout=10)
data = resp.json()

for bookmark in data["bookmarks"]:
    comment = bookmark.get("comment", "").strip()
    if comment:  # コメントがある人だけ
        print(f"{bookmark['user']}: {comment}")
```

ブックマークした全員がコメントを書くわけではないので、コメントが空でないものだけを抽出しています。

### サーバーへの配慮

はてなのAPIは公開されていますが、短時間に大量リクエストを送るとアクセス制限される可能性があります。本プロジェクトでは以下の対策をしています。

```python
import time

for entry in top_entries[:5]:  # 上位5件に限定
    time.sleep(0.5)            # 0.5秒待つ
    comments = self.get_comments(entry["url"])
```

- 取得対象をブックマーク数上位5件に限定
- リクエスト間に0.5秒のスリープを挿入

## 3つのライブラリの使い分けまとめ

| ライブラリ | 得意なこと          | 本プロジェクトでの役割               |
| ---------- | ------------------- | ------------------------------------ |
| feedparser | RSSフィードのパース | Googleニュース・はてブ検索の記事取得 |
| atproto    | BlueSky APIとの通信 | SNS投稿の検索・取得                  |
| requests   | 汎用HTTPリクエスト  | はてブJSON APIからコメント取得       |

feedparserはRSS/Atom形式のXMLを扱うのに特化しており、requestsで取得してXMLを自分でパースするより圧倒的に楽です。atprotoはBlueSky専用のSDKで、認証やページネーションを内部で処理してくれます。

## 参考リンク

- [feedparser ドキュメント](https://feedparser.readthedocs.io/)
- [atproto（Python SDK）GitHub](https://github.com/MarshalX/atproto)
- [AT Protocol 仕様](https://atproto.com/)
- [はてなブックマーク エントリー情報取得API](https://developer.hatena.ne.jp/ja/documents/bookmark/apis/getinfo)
