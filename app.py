"""TrendInsight AI - メディア・SNS・はてブの多角的トレンド感情分析アプリ."""

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from wordcloud import WordCloud

from src.analyzer import analyze_sentiment, compute_sentiment_stats, load_tokenizer
from src.collector import HatenaCollector, fetch_bluesky_posts, fetch_news_articles

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
    hatena_stats: dict[str, float] | None = None,
) -> str:
    """各ソースの感情スコアを比較しインサイトを生成する.

    Args:
        news_stats: メディア側の感情比率.
        bsky_stats: BlueSky側の感情比率.
        hatena_stats: はてなブックマーク側の感情比率.

    Returns:
        比較インサイトのテキスト.
    """
    def dominant(stats: dict[str, float]) -> str:
        return max(stats, key=stats.get)

    tone_map = {
        "positive": "ポジティブ（好意的）",
        "neutral": "中立的",
        "negative": "ネガティブ（批判的・懸念）",
    }

    parts: list[str] = [f"メディアは **{tone_map[dominant(news_stats)]}**"]
    if bsky_stats:
        parts.append(f"BlueSkyは **{tone_map[dominant(bsky_stats)]}**")
    if hatena_stats:
        parts.append(f"はてなブックマークは **{tone_map[dominant(hatena_stats)]}**")

    summary = "、".join(parts) + " な論調です。"

    # ギャップ検出
    all_dominants = [dominant(news_stats)]
    if bsky_stats:
        all_dominants.append(dominant(bsky_stats))
    if hatena_stats:
        all_dominants.append(dominant(hatena_stats))

    if len(set(all_dominants)) == 1:
        return f"📊 {summary} 各ソース間で論調が一致しています。"
    return f"📊 {summary} ソース間で温度差があります。"


def save_history(
    keyword: str,
    news_results: list[dict],
    sns_results: list[dict],
    hatena_results: list[dict],
) -> None:
    """分析結果を履歴JSONに追記保存する.

    Args:
        keyword: 検索キーワード.
        news_results: メディア分析結果.
        sns_results: SNS分析結果.
        hatena_results: はてブ分析結果.
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
        "hatena_results": hatena_results,
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


def render_hatena_section(
    hatena_data: list[dict], hatena_results: list[dict], keyword: str
) -> None:
    """はてなブックマークセクションを表示する.

    Args:
        hatena_data: 記事ごとのコメントデータ.
        hatena_results: 感情分析済みコメントリスト.
        keyword: 検索キーワード.
    """
    st.subheader("💬 はてなブックマーク（第三者のツッコミ）")
    st.caption("出典: はてなブックマーク (https://b.hatena.ne.jp)")

    # 感情スコア
    hatena_stats = compute_sentiment_stats(hatena_results)
    render_sentiment_metrics(hatena_stats)

    # ワードクラウド
    hatena_texts = [r["title"] for r in hatena_results]
    wc = generate_wordcloud(extract_keywords(hatena_texts, keyword))
    if wc:
        st.image(wc.to_array(), use_container_width=True)

    # 記事ごとの象徴的コメント表示
    st.markdown("---")
    for article_data in hatena_data:
        with st.expander(
            f"📰 {article_data['title'][:60]}... "
            f"（{article_data['bookmark_count']}ブックマーク）"
        ):
            # 上位3件のコメントをピックアップ
            for comment in article_data["comments"][:3]:
                st.markdown(
                    f"> {comment['comment']}\n>\n"
                    f"> — *{comment['user']}*"
                )

    return hatena_stats


def main() -> None:
    """Streamlit UIのメインエントリポイント."""
    st.set_page_config(page_title="TrendInsight AI", page_icon="📊", layout="wide")
    st.title("📊 TrendInsight AI")
    st.caption("メディア・SNS・はてブの多角的視点から世の中の空気感を読み取る")

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

        # はてなブックマークコメント取得（キーワードで直接検索）
        hatena_data: list[dict] = []
        with st.spinner("📝 はてなブックマークのコメントを収集中..."):
            try:
                collector = HatenaCollector()
                hatena_data = collector.fetch_comments_by_keyword(keyword)
            except Exception as e:
                st.warning(f"はてなブックマーク取得エラー: {e}")

        if not news_articles and not sns_articles:
            st.warning("記事・投稿が見つかりませんでした。")
            return

        # --- 感情分析 ---
        news_results: list[dict] = []
        sns_results: list[dict] = []
        hatena_results: list[dict] = []

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

        # はてブコメントの感情分析
        all_comments: list[dict[str, str]] = []
        for article_data in hatena_data:
            for comment in article_data["comments"]:
                all_comments.append({
                    "title": comment["comment"],
                    "url": article_data["url"],
                    "author": comment["user"],
                })

        if all_comments:
            progress = st.progress(0, text="はてブコメントを分析中...")
            for i, comment in enumerate(all_comments):
                scores = analyze_sentiment(comment["title"])
                hatena_results.append({**comment, **scores})
                progress.progress((i + 1) / len(all_comments))
            progress.empty()

        # --- 感情スコア表示 ---
        news_stats = compute_sentiment_stats(news_results)
        bsky_stats = compute_sentiment_stats(sns_results) if sns_results else None
        hatena_stats = compute_sentiment_stats(hatena_results) if hatena_results else None

        st.subheader("🌡️ 世の中の空気感")

        # アクティブソースに応じてカラム構成
        active_sources: list[tuple[str, dict[str, float]]] = [
            ("📰 メディア", news_stats)
        ]
        if bsky_stats:
            active_sources.append(("💬 BlueSky", bsky_stats))
        if hatena_stats:
            active_sources.append(("📝 はてブ", hatena_stats))

        cols = st.columns(len(active_sources))
        for col, (source_name, stats) in zip(cols, active_sources):
            with col:
                st.markdown(f"#### {source_name}")
                render_sentiment_metrics(stats)

        # インサイト表示
        st.divider()
        st.markdown(generate_insight(news_stats, bsky_stats, hatena_stats))

        # --- ワードクラウド ---
        st.subheader("☁️ ワードクラウド")
        news_titles = [r["title"] for r in news_results]
        bsky_titles = [r["title"] for r in sns_results]
        hatena_texts = [r["title"] for r in hatena_results]

        source_titles = [("📰 メディア", news_titles)]
        if sns_results:
            source_titles.append(("💬 BlueSky", bsky_titles))
        if hatena_results:
            source_titles.append(("📝 はてブ", hatena_texts))

        wc_cols = st.columns(len(source_titles))
        for col, (source_name, titles) in zip(wc_cols, source_titles):
            with col:
                st.markdown(f"**{source_name}**")
                wc = generate_wordcloud(extract_keywords(titles, keyword))
                if wc:
                    st.image(wc.to_array(), use_container_width=True)

        # --- トレンドキーワード TOP5 ---
        st.subheader("🔑 トレンド・キーワード TOP5")
        kw_cols = st.columns(len(source_titles))
        for col, (source_name, titles) in zip(kw_cols, source_titles):
            with col:
                st.markdown(f"**{source_name}**")
                for i, (word, count) in enumerate(
                    extract_keywords(titles, keyword)[:5], 1
                ):
                    st.markdown(f"**{i}.** {word}（{count}回）")

        # --- はてブ詳細セクション ---
        if hatena_data:
            st.divider()
            st.subheader("📝 はてなブックマーク（第三者のツッコミ）")
            st.caption("出典: はてなブックマーク (https://b.hatena.ne.jp)")
            for article_data in hatena_data:
                with st.expander(
                    f"📰 {article_data['title'][:60]} "
                    f"（{article_data['bookmark_count']}ブックマーク）"
                ):
                    for comment in article_data["comments"][:3]:
                        st.markdown(
                            f"> {comment['comment']}\n>\n"
                            f"> — *{comment['user']}*"
                        )

        # --- 記事一覧 ---
        st.subheader("📰 記事・投稿一覧")
        tab_names: list[str] = ["📰 メディア"]
        tab_data: list[tuple[list[dict], bool]] = [(news_results, False)]
        if sns_results:
            tab_names.append("💬 BlueSky")
            tab_data.append((sns_results, True))
        if hatena_results:
            tab_names.append("📝 はてブコメント")
            tab_data.append((hatena_results, True))

        tabs = st.tabs(tab_names)
        for tab, (results, show_author) in zip(tabs, tab_data):
            with tab:
                render_article_list(results, show_author=show_author)

        # --- 履歴保存 ---
        save_history(keyword, news_results, sns_results, hatena_results)
        st.info("💾 分析結果を履歴に保存しました。")


if __name__ == "__main__":
    main()
