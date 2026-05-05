"""TrendInsight AI - キーワードに基づくニューストレンド感情分析アプリ."""

import json
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path

import feedparser
import streamlit as st
from asari.api import Sonar
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud

HISTORY_PATH = Path("data/analysis_history.json")
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
FETCH_COUNT = 30

sonar = Sonar()
tokenizer = Tokenizer()

# ストップワード（助詞・助動詞・不要語）
STOP_WORDS: set[str] = {
    "の", "に", "は", "が", "を", "で", "と", "も", "た", "だ", "する", "いる",
    "ある", "こと", "それ", "これ", "ない", "なる", "れる", "られる", "よう",
    "さん", "ため", "から", "まで", "など", "について", "として", "における",
    "Yahoo", "ニュース", "新聞", "速報", "記事", "配信", "発表",
}


def fetch_articles(keyword: str) -> list[dict[str, str]]:
    """GoogleニュースRSSからキーワード関連記事を最大30件取得する.

    Args:
        keyword: 検索キーワード.

    Returns:
        タイトルとURLを含む辞書のリスト.
    """
    encoded = urllib.parse.quote(keyword)
    # タイムスタンプで確実に最新を取得（キャッシュ回避）
    ts = datetime.now().timestamp()
    url = (
        f"https://news.google.com/rss/search?q={encoded}&hl=ja&gl=JP"
        f"&ceid=JP:ja&_t={ts}"
    )
    feed = feedparser.parse(url)
    return [
        {"title": entry.title, "url": entry.link}
        for entry in feed.entries[:FETCH_COUNT]
    ]


def analyze_sentiment(title: str) -> dict[str, float | str]:
    """記事タイトルの感情分析をasariで実施する.

    Args:
        title: 分析対象のテキスト.

    Returns:
        positive/negativeスコアとラベルを含む辞書.
    """
    result = sonar.ping(text=title)
    scores = {item["class_name"]: item["confidence"] for item in result["classes"]}
    return {
        "positive": scores.get("positive", 0.0),
        "negative": scores.get("negative", 0.0),
        "label": result["top_class"],
    }


def extract_keywords(
    titles: list[str], search_keyword: str
) -> list[tuple[str, int]]:
    """記事タイトル群から名詞を抽出し頻出順に返す.

    Args:
        titles: 記事タイトルのリスト.
        search_keyword: 除外する検索キーワード.

    Returns:
        (単語, 出現回数)のリスト（頻出順）.
    """
    stop = STOP_WORDS | {search_keyword}
    words: list[str] = []
    for title in titles:
        for token in tokenizer.tokenize(title):
            part = token.part_of_speech.split(",")[0]
            surface = token.surface
            if part == "名詞" and len(surface) > 1 and surface not in stop:
                words.append(surface)
    return Counter(words).most_common()


def generate_wordcloud(word_freq: list[tuple[str, int]]) -> WordCloud:
    """頻出単語からワードクラウドを生成する.

    Args:
        word_freq: (単語, 出現回数)のリスト.

    Returns:
        生成されたWordCloudオブジェクト.
    """
    freq_dict = dict(word_freq)
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    )
    wc.generate_from_frequencies(freq_dict)
    return wc


def save_history(keyword: str, results: list[dict]) -> None:
    """分析結果を履歴JSONに追記保存する.

    Args:
        keyword: 検索キーワード.
        results: 分析結果リスト.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    history.append({
        "timestamp": datetime.now().isoformat(),
        "keyword": keyword,
        "results": results,
    })
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """Streamlit UIのメインエントリポイント."""
    st.set_page_config(page_title="TrendInsight AI", page_icon="📊")
    st.title("📊 TrendInsight AI")
    st.caption("キーワードから世の中の空気感を読み取る")

    keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")

    if st.button("分析開始", disabled=not keyword):
        with st.spinner("記事を収集・分析中..."):
            articles = fetch_articles(keyword)
            if not articles:
                st.warning("記事が見つかりませんでした。")
                return

            results: list[dict] = []
            for article in articles:
                scores = analyze_sentiment(article["title"])
                results.append({**article, **scores})

            # 平均スコア算出
            avg_positive = sum(r["positive"] for r in results) / len(results)
            avg_negative = sum(r["negative"] for r in results) / len(results)

            # 世の中の空気感表示
            st.subheader("🌡️ 世の中の空気感")
            col1, col2 = st.columns(2)
            col1.metric("ポジティブ", f"{avg_positive:.1%}")
            col2.metric("ネガティブ", f"{avg_negative:.1%}")

            if avg_positive > avg_negative:
                st.success("全体的にポジティブな傾向です 😊")
            else:
                st.error("全体的にネガティブな傾向です 😟")

            # ワードクラウド表示
            titles = [r["title"] for r in results]
            word_freq = extract_keywords(titles, keyword)

            if word_freq:
                st.subheader("☁️ ワードクラウド")
                wc = generate_wordcloud(word_freq)
                st.image(wc.to_array(), use_container_width=True)

                # トップ5キーワード表示
                st.subheader("🔑 トレンド・キーワード TOP5")
                for i, (word, count) in enumerate(word_freq[:5], 1):
                    st.markdown(f"**{i}.** {word}（{count}回）")

            # 個別記事表示
            st.subheader("📰 記事一覧")
            for r in results:
                emoji = "🟢" if r["label"] == "positive" else "🔴"
                st.markdown(
                    f"{emoji} [{r['title']}]({r['url']}) "
                    f"(positive: {r['positive']:.1%})"
                )

            # 履歴保存
            save_history(keyword, results)
            st.info("💾 分析結果を履歴に保存しました。")


if __name__ == "__main__":
    main()
