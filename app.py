"""TrendInsight AI - メディア・SNS・掲示板の3ソース比較によるトレンド感情分析."""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from wordcloud import WordCloud

from src.analyzer import analyze_sentiment, compute_sentiment_stats, load_tokenizer
from src.collector import fetch_bluesky_posts, fetch_news_articles, fetch_reddit_posts

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
    news_stats: dict[str, float],
    bsky_stats: dict[str, float] | None = None,
    reddit_stats: dict[str, float] | None = None,
) -> str:
    """各ソースの感情スコアを比較しインサイトを生成する.

    Args:
        news_stats: メディア側の感情比率.
        bsky_stats: BlueSky側の感情比率.
        reddit_stats: Reddit側の感情比率.

    Returns:
        比較インサイトのテキスト.
    """
    def dominant(stats: dict[str, float]) -> str:
        return max(stats, key=stats.get)

    tone_map = {
        "positive": "ポジティブ",
        "neutral": "中立的",
        "negative": "ネガティブ",
    }

    parts: list[str] = [f"メディアは**{tone_map[dominant(news_stats)]}**"]
    if bsky_stats:
        parts.append(f"BlueSkyは**{tone_map[dominant(bsky_stats)]}**")
    if reddit_stats:
        parts.append(f"Redditは**{tone_map[dominant(reddit_stats)]}**")

    summary = "、".join(parts) + "な論調です。"

    # ギャップ検出
    all_stats = [("メディア", news_stats)]
    if bsky_stats:
        all_stats.append(("BlueSky", bsky_stats))
    if reddit_stats:
        all_stats.append(("Reddit", reddit_stats))

    dominants = {name: dominant(s) for name, s in all_stats}
    unique_tones = set(dominants.values())

    if len(unique_tones) == 1:
        return f"📊 {summary} 各ソース間で論調が一致しています。"
    return f"📊 {summary} ソース間で温度差があります。"


def save_history(
    keyword: str,
    news_results: list[dict],
    bsky_results: list[dict],
    reddit_results: list[dict],
) -> None:
    """分析結果を履歴JSONに追記保存する.

    Args:
        keyword: 検索キーワード.
        news_results: メディア分析結果.
        bsky_results: BlueSky分析結果.
        reddit_results: Reddit分析結果.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    history.append({
        "timestamp": datetime.now().isoformat(),
        "keyword": keyword,
        "news_results": news_results,
        "bsky_results": bsky_results,
        "reddit_results": reddit_results,
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
        author = ""
        if show_author:
            if "author" in r:
                author = f" ({r['author']})"
            if "subreddit" in r:
                author = f" ({r['subreddit']} / {r.get('author', '')})"
        st.markdown(
            f"{emoji} [{r['title'][:80]}]({r['url']}){author} "
            f"(positive: {r['positive']:.1%})"
        )


def analyze_batch(
    items: list[dict[str, str]], label: str
) -> list[dict]:
    """記事/投稿リストをバッチで感情分析する.

    Args:
        items: タイトルを含む辞書のリスト.
        label: プログレスバーに表示するラベル.

    Returns:
        感情スコアを付与した辞書のリスト.
    """
    results: list[dict] = []
    if not items:
        return results
    progress = st.progress(0, text=f"{label}を分析中...")
    for i, item in enumerate(items):
        scores = analyze_sentiment(item["title"])
        results.append({**item, **scores})
        progress.progress((i + 1) / len(items))
    progress.empty()
    return results


def main() -> None:
    """Streamlit UIのメインエントリポイント."""
    st.set_page_config(page_title="TrendInsight AI", page_icon="📊", layout="wide")
    st.title("📊 TrendInsight AI")
    st.caption("メディア・SNS・掲示板の3面から世の中の空気感を読み取る")

    # サイドバー: 認証設定
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

        st.divider()
        st.subheader("Reddit API認証")
        reddit_client_id = st.text_input(
            "Client ID",
            value=os.getenv("REDDIT_CLIENT_ID", ""),
            type="password",
        )
        reddit_client_secret = st.text_input(
            "Client Secret",
            value=os.getenv("REDDIT_CLIENT_SECRET", ""),
            type="password",
        )
        st.markdown("[Reddit APIキー取得](https://www.reddit.com/prefs/apps)")

    # メイン: キーワード入力
    keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")

    if st.button("分析開始", disabled=not keyword):
        # --- データ収集 ---
        with st.spinner("📰 ニュース記事を収集中..."):
            news_articles = fetch_news_articles(keyword)

        bsky_articles: list[dict[str, str]] = []
        if bsky_handle and bsky_password:
            with st.spinner("💬 BlueSky投稿を収集中..."):
                try:
                    bsky_articles = fetch_bluesky_posts(
                        keyword, bsky_handle, bsky_password
                    )
                except Exception as e:
                    st.warning(f"BlueSky取得エラー: {e}")

        reddit_articles: list[dict[str, str]] = []
        if reddit_client_id and reddit_client_secret:
            with st.spinner("🔥 Reddit投稿を収集中..."):
                try:
                    reddit_articles = fetch_reddit_posts(
                        keyword, reddit_client_id, reddit_client_secret
                    )
                except Exception as e:
                    st.warning(f"Reddit取得エラー: {e}")

        if not news_articles and not bsky_articles and not reddit_articles:
            st.warning("記事・投稿が見つかりませんでした。")
            return

        # 未設定ソースの案内
        missing: list[str] = []
        if not bsky_handle or not bsky_password:
            missing.append("BlueSky")
        if not reddit_client_id or not reddit_client_secret:
            missing.append("Reddit")
        if missing:
            st.info(f"💡 サイドバーで{' / '.join(missing)}の認証を設定すると、より多角的な分析が可能です。")

        # --- 感情分析 ---
        news_results = analyze_batch(news_articles, "📰 メディア記事")
        bsky_results = analyze_batch(bsky_articles, "💬 BlueSky投稿")
        reddit_results = analyze_batch(reddit_articles, "🔥 Reddit投稿")

        # --- 感情スコア表示 ---
        news_stats = compute_sentiment_stats(news_results)
        bsky_stats = compute_sentiment_stats(bsky_results) if bsky_results else None
        reddit_stats = compute_sentiment_stats(reddit_results) if reddit_results else None

        st.subheader("🌡️ 世の中の空気感")

        # アクティブなソース数に応じてカラム構成を変更
        active_sources: list[tuple[str, dict[str, float]]] = [
            ("📰 メディア", news_stats)
        ]
        if bsky_stats:
            active_sources.append(("💬 BlueSky", bsky_stats))
        if reddit_stats:
            active_sources.append(("🔥 Reddit", reddit_stats))

        cols = st.columns(len(active_sources))
        for col, (source_name, stats) in zip(cols, active_sources):
            with col:
                st.markdown(f"#### {source_name}")
                render_sentiment_metrics(stats)

        # インサイト表示
        st.divider()
        st.markdown(generate_insight(news_stats, bsky_stats, reddit_stats))

        # --- ワードクラウド ---
        st.subheader("☁️ ワードクラウド")
        news_titles = [r["title"] for r in news_results]
        bsky_titles = [r["title"] for r in bsky_results]
        reddit_titles = [r["title"] for r in reddit_results]

        wc_cols = st.columns(len(active_sources))
        all_titles = [news_titles, bsky_titles, reddit_titles]
        for i, (col, (source_name, _)) in enumerate(zip(wc_cols, active_sources)):
            with col:
                st.markdown(f"**{source_name}**")
                titles = all_titles[i] if i < len(all_titles) else []
                wc = generate_wordcloud(extract_keywords(titles, keyword))
                if wc:
                    st.image(wc.to_array(), use_container_width=True)

        # --- トレンドキーワード TOP5 ---
        st.subheader("🔑 トレンド・キーワード TOP5")
        kw_cols = st.columns(len(active_sources))
        for i, (col, (source_name, _)) in enumerate(zip(kw_cols, active_sources)):
            with col:
                st.markdown(f"**{source_name}**")
                titles = all_titles[i] if i < len(all_titles) else []
                for j, (word, count) in enumerate(
                    extract_keywords(titles, keyword)[:5], 1
                ):
                    st.markdown(f"**{j}.** {word}（{count}回）")

        # --- 記事一覧 ---
        st.subheader("📰 記事・投稿一覧")
        tab_names: list[str] = ["📰 メディア"]
        tab_data: list[tuple[list[dict], bool]] = [(news_results, False)]
        if bsky_results:
            tab_names.append("💬 BlueSky")
            tab_data.append((bsky_results, True))
        if reddit_results:
            tab_names.append("🔥 Reddit")
            tab_data.append((reddit_results, True))

        tabs = st.tabs(tab_names)
        for tab, (results, show_author) in zip(tabs, tab_data):
            with tab:
                render_article_list(results, show_author=show_author)

        # --- 履歴保存 ---
        save_history(keyword, news_results, bsky_results, reddit_results)
        st.info("💾 分析結果を履歴に保存しました。")


if __name__ == "__main__":
    main()
