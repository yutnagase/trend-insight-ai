"""TrendInsight AI - キーワードに基づくニューストレンド感情分析アプリ."""

import json
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path

import feedparser
import streamlit as st
import torch
from janome.tokenizer import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from wordcloud import WordCloud

HISTORY_PATH = Path("data/analysis_history.json")
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
FETCH_COUNT = 30
MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"

# ストップワード（助詞・助動詞・不要語）
STOP_WORDS: set[str] = {
    "の", "に", "は", "が", "を", "で", "と", "も", "た", "だ", "する", "いる",
    "ある", "こと", "それ", "これ", "ない", "なる", "れる", "られる", "よう",
    "さん", "ため", "から", "まで", "など", "について", "として", "における",
    "Yahoo", "ニュース", "新聞", "速報", "記事", "配信", "発表",
}

# ネガティブ補正用辞書（強いネガティブ語）
NEGATIVE_BOOST_WORDS: set[str] = {
    "戦争", "悲惨", "孤児", "死亡", "倒産", "殺害", "虐待", "災害", "被害",
    "犠牲", "破壊", "崩壊", "暴力", "貧困", "飢餓", "難民", "紛争", "侵攻",
    "爆撃", "テロ", "事故", "汚染", "感染", "死者", "遺体", "自殺", "破綻",
    "詐欺", "逮捕", "懲役", "不正", "隠蔽", "搾取", "差別", "迫害",
}
NEGATIVE_BOOST_WEIGHT: float = 0.3


@st.cache_resource
def load_sentiment_model():
    """感情分析モデルを起動時に一度だけロードする.

    Returns:
        transformers pipelineオブジェクト.
    """
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,  # CPU
        truncation=True,
        max_length=512,
    )


@st.cache_resource
def load_tokenizer() -> Tokenizer:
    """janomeトークナイザーを起動時に一度だけロードする.

    Returns:
        Tokenizerインスタンス.
    """
    return Tokenizer()


def fetch_articles(keyword: str) -> list[dict[str, str]]:
    """GoogleニュースRSSからキーワード関連記事を最大30件取得する.

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
        for entry in feed.entries[:FETCH_COUNT]
    ]


def analyze_sentiment(title: str) -> dict[str, float | str]:
    """記事タイトルの感情分析をBERTモデル+辞書補正で実施する.

    モデルの確信度が低い（中立的な文）場合は50:50として扱い、
    強ネガティブ語が含まれる場合のみ補正を適用する。

    Args:
        title: 分析対象のテキスト.

    Returns:
        positive/negativeスコアとラベルを含む辞書.
    """
    classifier = load_sentiment_model()
    result = classifier(title)[0]
    label = result["label"].upper()
    score = result["score"]

    # モデルが3クラス（POSITIVE/NEGATIVE/NEUTRAL）を出力
    if label == "POSITIVE":
        pos_score = score
        neg_score = 1.0 - score
    elif label == "NEGATIVE":
        neg_score = score
        pos_score = 1.0 - score
    else:  # NEUTRAL
        pos_score = 0.5
        neg_score = 0.5

    # ネガティブ辞書による補正（強ネガティブ語が存在する場合のみ）
    boost = sum(1 for w in NEGATIVE_BOOST_WORDS if w in title)
    if boost > 0:
        adjustment = min(boost * NEGATIVE_BOOST_WEIGHT, 0.5)
        neg_score = min(neg_score + adjustment, 1.0)
        pos_score = max(pos_score - adjustment, 0.0)

    # 最終ラベル判定（3段階）
    if abs(pos_score - neg_score) < 0.1:
        final_label = "neutral"
    elif pos_score > neg_score:
        final_label = "positive"
    else:
        final_label = "negative"
    return {"positive": pos_score, "negative": neg_score, "label": final_label}


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
    tok = load_tokenizer()
    stop = STOP_WORDS | {search_keyword}
    words: list[str] = []
    for title in titles:
        for token in tok.tokenize(title):
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
            progress = st.progress(0)
            for i, article in enumerate(articles):
                scores = analyze_sentiment(article["title"])
                results.append({**article, **scores})
                progress.progress((i + 1) / len(articles))
            progress.empty()

            # ラベル別集計
            total = len(results)
            pos_count = sum(1 for r in results if r["label"] == "positive")
            neg_count = sum(1 for r in results if r["label"] == "negative")
            neu_count = total - pos_count - neg_count
            pos_pct = pos_count / total
            neg_pct = neg_count / total
            neu_pct = neu_count / total

            # 世の中の空気感表示
            st.subheader("🌡️ 世の中の空気感")
            col1, col2, col3 = st.columns(3)
            col1.metric("ポジティブ", f"{pos_pct:.1%}")
            col2.metric("中立", f"{neu_pct:.1%}")
            col3.metric("ネガティブ", f"{neg_pct:.1%}")

            dominant = max(
                ("positive", pos_pct), ("neutral", neu_pct), ("negative", neg_pct),
                key=lambda x: x[1],
            )
            if dominant[0] == "positive":
                st.success("全体的にポジティブな傾向です 😊")
            elif dominant[0] == "negative":
                st.error("全体的にネガティブな傾向です 😟")
            else:
                st.info("全体的に中立的な傾向です 😐")

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
                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[
                    r["label"]
                ]
                st.markdown(
                    f"{emoji} [{r['title']}]({r['url']}) "
                    f"(positive: {r['positive']:.1%})"
                )

            # 履歴保存
            save_history(keyword, results)
            st.info("💾 分析結果を履歴に保存しました。")


if __name__ == "__main__":
    main()
