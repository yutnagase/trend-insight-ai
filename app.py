"""TrendInsight AI - キーワードに基づくニューストレンド感情分析アプリ."""

import json
import urllib.parse
from datetime import datetime
from pathlib import Path

import feedparser
import streamlit as st
from asari.api import Sonar

HISTORY_PATH = Path("data/analysis_history.json")
sonar = Sonar()


def fetch_articles(keyword: str) -> list[dict[str, str]]:
    """GoogleニュースRSSからキーワード関連記事を最大10件取得する.

    Args:
        keyword: 検索キーワード.

    Returns:
        タイトルとURLを含む辞書のリスト.
    """
    encoded = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(url)
    return [
        {"title": entry.title, "url": entry.link}
        for entry in feed.entries[:10]
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
