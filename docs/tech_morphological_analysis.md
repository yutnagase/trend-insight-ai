# 形態素解析（Morphological Analysis） - Janome

## 形態素解析とは

形態素解析は、日本語の文を単語（形態素）に分割する処理です。英語はスペースで単語が区切られていますが、日本語は「今日はいい天気ですね」のように単語の切れ目がありません。形態素解析を使うと、これを「今日 / は / いい / 天気 / です / ね」と分割できます。

本プロジェクトでは、ワードクラウドやトレンドキーワード抽出のために形態素解析を使っています。

## なぜJanomeを選んだか

日本語の形態素解析ツールはいくつかありますが、Janomeには以下の利点があります。

- **Pure Python実装**
  - C言語のコンパイルが不要で、`pip install janome` だけで入る
- **辞書内蔵**
  - MeCabのように別途辞書をインストールする必要がない
- **軽量**
  - 本プロジェクトのように「名詞を抽出したい」程度の用途には十分な性能

MeCabの方が高速ですが、環境構築の手間を考えるとJanomeの方が手軽です。

## 基本的な使い方

```python
from janome.tokenizer import Tokenizer

tok = Tokenizer()

# 文を形態素に分割
for token in tok.tokenize("生成AIの規制法案が提出された"):
    print(f"{token.surface}\t{token.part_of_speech}")
```

出力：

```
生成    名詞,サ変接続,*,*
AI      名詞,固有名詞,組織,*
の      助詞,連体化,*,*
規制    名詞,サ変接続,*,*
法案    名詞,一般,*,*
が      助詞,格助詞,一般,*
提出    名詞,サ変接続,*,*
さ      動詞,自立,*,*
れ      動詞,接尾,*,*
た      助動詞,*,*,*
```

`token.surface` が表層形（見た目の文字列）、`token.part_of_speech` が品詞情報です。

## 本プロジェクトでの使い方

### 名詞だけを抽出する

ワードクラウドやキーワードランキングでは、助詞や動詞は不要です。名詞だけを取り出しています。

```python
from janome.tokenizer import Tokenizer
from collections import Counter

tok = Tokenizer()

def extract_keywords(titles):
    words = []
    for title in titles:
        for token in tok.tokenize(title):
            # 品詞の最初の要素が「名詞」かどうか
            part = token.part_of_speech.split(",")[0]
            if part == "名詞" and len(token.surface) > 1:
                words.append(token.surface)
    return Counter(words).most_common()
```

### ストップワードの除外

「の」「に」「は」のような助詞は品詞フィルタで除外されますが、名詞でも不要な語があります。たとえば「Yahoo」「ニュース」「記事」などはどのキーワードで検索しても出てくるので、ストップワードとして除外しています。

```python
STOP_WORDS = {
    "の", "に", "は", "が", "を", "で",  # 助詞（念のため）
    "Yahoo", "ニュース", "新聞", "速報",  # メディア共通語
    "https", "http", "www", "com", "jp",  # URL断片
}

# 検索キーワード自体も除外する
stop = STOP_WORDS | {search_keyword}
```

検索キーワードそのものも除外しています。「生成AI」で検索したら「生成」「AI」が最頻出になるのは当然なので、それを除いた上で何が話題になっているかを見たいからです。

### 数字トークンの除外

Janomeは「30」「10」などの数字を名詞（数詞）として解析します。「処理時間が30分から10分へ」のような文から「30」「10」がキーワードとして抽出されても意味がありません。そのため、数字のみのトークンは正規表現で除外しています。

```python
import re

_NUMERIC_PATTERN = re.compile(r"^[\d,.\-+%０-９]+$")

# 抽出時のフィルタに追加
if not _NUMERIC_PATTERN.match(surface):
    words.append(surface)
```

### メディア名の動的除外

GoogleニュースRSSのタイトルは「記事タイトル - メディア名」の形式です。そのため、特定メディアの記事が多いと、そのメディア名がキーワード上位に来てしまいます（例: 「ゴリミー」）。

これを防ぐため、タイトル末尾の「 - メディア名」部分からメディア名を動的に抽出し、`extra_stop_words` としてキーワード抽出時に除外しています。

```python
# タイトルからメディア名を抽出
news_media_names = set()
for title in news_titles:
    if " - " in title:
        media = title.rsplit(" - ", 1)[-1].strip()
        if media:
            news_media_names.add(media)

# キーワード抽出時に除外
extract_keywords(titles, keyword, extra_stop_words=news_media_names)
```

### 1文字の語を除外する理由

```python
if part == "名詞" and len(token.surface) > 1:
```

1文字の名詞（「人」「国」「日」など）は意味が広すぎて、キーワードとしての情報量が低いため除外しています。

## Tokenizerのキャッシュ

Janomeの `Tokenizer()` は内部辞書の読み込みに少し時間がかかるので、Streamlitのキャッシュで使い回しています。

```python
@st.cache_resource
def load_tokenizer():
    return Tokenizer()
```

## 参考リンク

- [Janome公式ドキュメント](https://mocobeta.github.io/janome/)
- [PyPI - janome](https://pypi.org/project/Janome/)
