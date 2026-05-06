# Streamlit — WebアプリのUI層

## Streamlitとは

Streamlitは、Pythonだけでブラウザ上に動くWebアプリを作れるフレームワークです。HTML/CSS/JavaScriptを一切書かずに、データの入力フォームやグラフ、テーブルなどを表示できます。

本プロジェクトでは、キーワード入力→分析実行→結果表示という一連の流れをStreamlitで構築しています。

## Streamlitの選定理由

- Pythonだけで完結する（フロントエンドが不要）
- データ分析系のUIに特化しており、グラフやメトリクス表示が簡単
- `streamlit run app.py` の1コマンドで起動できる手軽さ
- 状態管理やキャッシュの仕組みが組み込まれている

FlaskやDjangoだとHTMLテンプレートを書く必要がありますが、Streamlitなら全部Pythonで済む点もメリットとして考慮

## 基本的な使い方

```python
import streamlit as st

# タイトルを表示
st.title("📊 TrendInsight AI")

# テキスト入力
keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")

# ボタン
if st.button("分析開始"):
    st.write(f"「{keyword}」を分析します...")
```

これだけで、ブラウザ上にタイトル・入力欄・ボタンが表示される

## 本プロジェクトで使っている主な機能

### レイアウト（カラム分割）

感情スコアを横並びで表示するために `st.columns` を使っている

```python
col1, col2, col3 = st.columns(3)
col1.metric("ポジティブ", "35.0%")
col2.metric("中立", "50.0%")
col3.metric("ネガティブ", "15.0%")
```

### タブ切り替え

メディア・BlueSky・はてブの記事一覧をタブで切り替えて表示しています。

```python
tab_news, tab_bsky = st.tabs(["📰 メディア", "💬 BlueSky"])

with tab_news:
    st.write("ニュース記事の一覧")

with tab_bsky:
    st.write("BlueSky投稿の一覧")
```

### プログレスバー

感情分析の進捗を表示するのに使っています。

```python
progress = st.progress(0, text="分析中...")
for i, article in enumerate(articles):
    # 分析処理
    progress.progress((i + 1) / len(articles))
progress.empty()  # 完了後に消す
```

### キャッシュ（@st.cache_resource）

BERTモデルやLLMは読み込みに時間がかかるので、一度読み込んだら使い回しする

```python
@st.cache_resource
def load_sentiment_model():
    """アプリ起動中に1回だけ実行される"""
    return pipeline("sentiment-analysis", model="...")
```

`@st.cache_resource` を付けた関数は、Streamlitがページを再描画しても再実行されません。重いモデルの読み込みを毎回やらずに済むので、2回目以降の分析が高速になる

### サイドバー

設定項目（BlueSky認証情報、過去の履歴選択）はサイドバーに配置しています。

```python
with st.sidebar:
    st.header("⚙️ 設定")
    handle = st.text_input("ハンドル", placeholder="yourname.bsky.social")
```

## Streamlitの動作の仕組み

Streamlitには独特な動作モデルがあります。ボタンを押したり入力を変えたりするたびに、**app.pyが上から下まで全部再実行**されます。

これは最初は戸惑いますが、`@st.cache_resource` や `st.session_state` を使えば、必要な状態を保持しつつ画面を更新できます。

## 参考リンク

- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [APIリファレンス](https://docs.streamlit.io/library/api-reference)
