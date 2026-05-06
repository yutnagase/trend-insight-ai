# ワードクラウド — wordcloudライブラリ

## これは何？

ワードクラウドは、テキスト中の単語を出現頻度に応じた大きさで配置する可視化手法です。よく出てくる単語ほど大きく表示されるので、「何が話題の中心か」がひと目でわかります。

本プロジェクトでは、メディア・BlueSky・はてブそれぞれの頻出語をワードクラウドにして並べることで、ソースごとの論点の違いを視覚的に比較しています。

## 基本的な使い方

```python
from wordcloud import WordCloud

# テキストからワードクラウドを生成
text = "Python データ分析 Python 機械学習 Python AI データ分析 統計"
wc = WordCloud(width=800, height=400, background_color="white")
wc.generate(text)

# 画像として保存
wc.to_file("wordcloud.png")
```

これだけで、「Python」が一番大きく、「データ分析」がその次に大きい画像が生成されます。

## 日本語で使うときの注意点

wordcloudライブラリは英語前提で作られているため、日本語を扱うには2つの対応が必要です。

### 1. 事前に単語分割する

英語はスペース区切りで単語を認識しますが、日本語にはスペースがありません。そのため、Janomeで形態素解析した結果を渡す必要があります。

本プロジェクトでは `generate_from_frequencies()` を使い、単語と出現回数の辞書を直接渡しています。

```python
# Janomeで名詞を抽出し、出現回数をカウント済みの状態
word_freq = [("経済", 15), ("政策", 12), ("成長", 8), ("規制", 6)]

wc = WordCloud(
    font_path="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    width=800,
    height=400,
    background_color="white",
)
wc.generate_from_frequencies(dict(word_freq))
```

### 2. 日本語フォントを指定する

`font_path` を指定しないと日本語が文字化け（豆腐□□□）になります。システムにインストールされている日本語フォントのパスを渡します。

```python
# Linux（Ubuntu/Debian系）
font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"

# macOS
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"

# Windows
font_path = "C:/Windows/Fonts/msgothic.ttc"
```

## 本プロジェクトでの実装

### ワードクラウド生成

```python
from wordcloud import WordCloud

def generate_wordcloud(word_freq):
    if not word_freq:
        return None
    wc = WordCloud(
        font_path="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    )
    wc.generate_from_frequencies(dict(word_freq))
    return wc
```

`colormap="viridis"` は配色の指定です。matplotlibのカラーマップ名を指定でき、見た目の印象を変えられます。

### Streamlitでの表示

生成したワードクラウドは画像としてStreamlit上に表示しています。

```python
wc = generate_wordcloud(word_freq)
if wc:
    st.image(wc.to_array(), use_container_width=True)
```

`to_array()` でNumPy配列に変換し、`st.image()` に渡しています。ファイルに保存せずにそのまま表示できるので手軽です。

### 画像の保存（履歴用）

過去の分析結果を再閲覧するために、ワードクラウド画像をPNGファイルとしても保存しています。

```python
def save_wordcloud_image(wc, timestamp, source):
    filepath = f"data/images/wordcloud_{source}_{timestamp}.png"
    wc.to_file(filepath)
    return filepath
```

## generate() と generate_from_frequencies() の違い

| メソッド | 入力 | 用途 |
|----------|------|------|
| `generate(text)` | スペース区切りのテキスト | 英語テキストをそのまま渡す場合 |
| `generate_from_frequencies(dict)` | `{"単語": 出現回数}` の辞書 | 事前に単語分割・集計済みの場合 |

日本語では形態素解析で単語分割してからカウントするので、`generate_from_frequencies()` の方が自然です。

## 参考リンク

- [wordcloud PyPI](https://pypi.org/project/wordcloud/)
- [wordcloud GitHub](https://github.com/amueller/word_cloud)
