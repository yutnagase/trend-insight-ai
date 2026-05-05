"""TrendInsight AI - メディアとSNSの両論併記によるトレンド感情分析アプリ."""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from wordcloud import WordCloud

from src.analyzer import analyze_sentiment, compute_sentiment_stats, load_tokenizer
from src.collector import fetch_bluesky_posts, fetch_news_articles

load_dotenv()

HISTORY_PATH = Path("data/analysis_history.json")
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"

# ストップワード（助詞・助動詞・不要語）
STOP_WORDS: set[str] = {
    "の", "に", "は", "が", "を", "で", "と", "も", "た", "だ", "する", "いる",
    "ある", "こと", "それ", "これ", "ない", "なる", "れる", "られる", "よう",
    "さん", "ため", "から", "まで", "など", "について", "として", "における",
    "Yahoo", "ニュース", "新聞", "速報", "記事", "配信", "発表",
    "https", "http", "www", "com", "jp",
}


def extract_keywords(
    titles: list[str], search_keyword: str
) -> list[tuple[str, int]]:
    """テキスト群から名詞を抽出し頻出順に返す.

    Args:
        titles: テキストのリスト.
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


def generate_wordcloud(word_freq: list[tuple[str, int]]) -> WordCloud | None:
    """頻出単語からワードクラウドを生成する.

    Args:
        word_freq: (単語, 出現回数)のリスト.

    Returns:
        生成されたWordCloudオブジェクト、またはデータ不足時None.
    """
    if not word_freq:
        return None
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    )
    wc.generate_from_frequencies(dict(word_freq))
    return wc


def generate_insight(
    news_stats: dict[str, float], sns_stats: dict[str, float]
) -> str:
    """メディアとSNSの感情スコアを比較しインサイトを生成する.

    Args:
        news_stats: メディア側の感情比率.
        sns_stats: SNS側の感情比率.

    Returns:
        比較インサイトのテキスト.
    """
    def dominant(stats: dict[str, float]) -> str:
        return max(stats, key=stats.get)

    news_dom = dominant(news_stats)
    sns_dom = dominant(sns_stats)

    tone_map = {
        "positive": "ポジティブ（好意的）",
        "neutral": "中立的（事実報道中心）",
        "negative": "ネガティブ（批判的・懸念）",
    }

    if news_dom == sns_dom:
        return (
            f"📊 メディアもSNSも**{tone_map[news_dom]}**な論調が中心です。"
            f"世論とメディアの方向性が一致しています。"
        )

    return (
        f"📊 メディアでは**{tone_map[news_dom]}**な報道が中心ですが、"
        f"SNSでは**{tone_map[sns_dom]}**な反応が目立ちます。"
        f"メディアと世論の間にギャップがあります。"
    )


def save_history(keyword: str, news_results: list[dict], sns_results: list[dict]) -> None:
    """分析結果を履歴JSONに追記保存する.

    Args:
        keyword: 検索キーワード.
        news_results: メディア分析結果.
        sns_results: SNS分析結果.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    history.append({
        "timestamp": datetime.now().isoformat(),
        "keyword": keyword,
        "news_results": news_results,
        "sns_results": sns_results,
    })
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def render_sentiment_metrics(stats: dict[str, float]) -> None:
    """感情比率を3カラムで表示する.

    Args:
        stats: positive/neutral/negativeの比率.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("ポジティブ", f"{stats['positive']:.1%}")
    col2.metric("中立", f"{stats['neutral']:.1%}")
    col3.metric("ネガティブ", f"{stats['negative']:.1%}")


def render_article_list(results: list[dict], show_author: bool = False) -> None:
    """記事/投稿一覧を表示する.

    Args:
        results: 分析結果リスト.
        show_author: 著者を表示するか.
    """
    for r in results:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[r["label"]]
        author = f" ({r['author']})" if show_author and "author" in r else ""
        st.markdown(
            f"{emoji} [{r['title'][:80]}]({r['url']}){author} "
            f"(positive: {r['positive']:.1%})"
        )


def main() -> None:
    """Streamlit UIのメインエントリポイント."""
    st.set_page_config(page_title="TrendInsight AI", page_icon="📊", layout="wide")
    st.title("📊 TrendInsight AI")
    st.caption("メディアとSNSの両面から世の中の空気感を読み取る")

    # サイドバー: BlueSky認証設定
    with st.sidebar:
        st.header("⚙️ 設定")
        st.subheader("BlueSky認証")
        bsky_handle = st.text_input(
            "ハンドル",
            value=os.getenv("BLUESKY_HANDLE", ""),
            placeholder="yourname.bsky.social",
        )
        bsky_password = st.text_input(
            "アプリパスワード",
            value=os.getenv("BLUESKY_APP_PASSWORD", ""),
            type="password",
        )
        st.markdown("[アプリパスワード生成](https://bsky.app/settings/app-passwords)")

    # メイン: キーワード入力
    keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")

    if st.button("分析開始", disabled=not keyword):
        # --- データ収集 ---
        with st.spinner("📰 ニュース記事を収集中..."):
            news_articles = fetch_news_articles(keyword)

        sns_articles: list[dict[str, str]] = []
        if bsky_handle and bsky_password:
            with st.spinner("💬 BlueSky投稿を収集中..."):
                try:
                    sns_articles = fetch_bluesky_posts(
                        keyword, bsky_handle, bsky_password
                    )
                except Exception as e:
                    st.warning(f"BlueSky取得エラー: {e}")
        else:
            st.info("💡 サイドバーでBlueSky認証を設定すると、SNSの声も分析できます。")

        if not news_articles and not sns_articles:
            st.warning("記事・投稿が見つかりませんでした。")
            return

        # --- 感情分析 ---
        news_results: list[dict] = []
        sns_results: list[dict] = []

        if news_articles:
            progress = st.progress(0, text="メディア記事を分析中...")
            for i, article in enumerate(news_articles):
                scores = analyze_sentiment(article["title"])
                news_results.append({**article, **scores})
                progress.progress((i + 1) / len(news_articles))
            progress.empty()

        if sns_articles:
            progress = st.progress(0, text="SNS投稿を分析中...")
            for i, post in enumerate(sns_articles):
                scores = analyze_sentiment(post["title"])
                sns_results.append({**post, **scores})
                progress.progress((i + 1) / len(sns_articles))
            progress.empty()

        # --- 感情スコア表示 ---
        news_stats = compute_sentiment_stats(news_results)
        sns_stats = compute_sentiment_stats(sns_results)

        if sns_results:
            # 両論併記モード
            st.subheader("🌡️ 世の中の空気感")
            col_news, col_sns = st.columns(2)

            with col_news:
                st.markdown("#### 📰 メディア（Googleニュース）")
                render_sentiment_metrics(news_stats)

            with col_sns:
                st.markdown("#### 💬 SNS（BlueSky）")
                render_sentiment_metrics(sns_stats)

            # インサイト表示
            st.divider()
            st.markdown(generate_insight(news_stats, sns_stats))
        else:
            # メディアのみモード
            st.subheader("🌡️ 世の中の空気感（メディア）")
            render_sentiment_metrics(news_stats)

        # --- ワードクラウド ---
        news_titles = [r["title"] for r in news_results]
        sns_titles = [r["title"] for r in sns_results]

        if sns_results:
            st.subheader("☁️ ワードクラウド")
            wc_col1, wc_col2 = st.columns(2)

            with wc_col1:
                st.markdown("**📰 メディア**")
                wc = generate_wordcloud(extract_keywords(news_titles, keyword))
                if wc:
                    st.image(wc.to_array(), use_container_width=True)

            with wc_col2:
                st.markdown("**💬 SNS**")
                wc = generate_wordcloud(extract_keywords(sns_titles, keyword))
                if wc:
                    st.image(wc.to_array(), use_container_width=True)
        else:
            word_freq = extract_keywords(news_titles, keyword)
            if word_freq:
                st.subheader("☁️ ワードクラウド")
                wc = generate_wordcloud(word_freq)
                if wc:
                    st.image(wc.to_array(), use_container_width=True)

        # --- トレンドキーワード TOP5 ---
        st.subheader("🔑 トレンド・キーワード TOP5")
        if sns_results:
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown("**📰 メディア**")
                for i, (word, count) in enumerate(
                    extract_keywords(news_titles, keyword)[:5], 1
                ):
                    st.markdown(f"**{i}.** {word}（{count}回）")
            with kw_col2:
                st.markdown("**💬 SNS**")
                for i, (word, count) in enumerate(
                    extract_keywords(sns_titles, keyword)[:5], 1
                ):
                    st.markdown(f"**{i}.** {word}（{count}回）")
        else:
            for i, (word, count) in enumerate(
                extract_keywords(news_titles, keyword)[:5], 1
            ):
                st.markdown(f"**{i}.** {word}（{count}回）")

        # --- 記事一覧 ---
        st.subheader("📰 記事一覧")
        if sns_results:
            tab_news, tab_sns = st.tabs(["📰 メディア", "💬 SNS"])
            with tab_news:
                render_article_list(news_results)
            with tab_sns:
                render_article_list(sns_results, show_author=True)
        else:
            render_article_list(news_results)

        # --- 履歴保存 ---
        save_history(keyword, news_results, sns_results)
        st.info("💾 分析結果を履歴に保存しました。")


if __name__ == "__main__":
    main()
