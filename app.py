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
from src.clients import BlueskyClient, GoogleNewsClient, HatenaClient
from src.models.article import Article
from src.reporter import generate_report

load_dotenv()

HISTORY_PATH = Path("data/analysis_history.json")
IMAGES_DIR = Path("data/images")
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


def save_wordcloud_image(wc: WordCloud, timestamp: str, source: str) -> str:
    """ワードクラウド画像をタイムスタンプ付きで保存する.

    Args:
        wc: WordCloudオブジェクト.
        timestamp: ISO形式のタイムスタンプ.
        source: ソース名（news, bsky, hatena）.

    Returns:
        保存先のファイルパス.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp.replace(":", "").replace("-", "").replace("T", "_")[:15]
    filename = f"wordcloud_{source}_{ts}.png"
    filepath = IMAGES_DIR / filename
    wc.to_file(str(filepath))
    return str(filepath)


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
    wordcloud_images: dict[str, str],
    ai_report: str,
) -> None:
    """分析結果を履歴JSONに追記保存する.

    Args:
        keyword: 検索キーワード.
        news_results: メディア分析結果.
        sns_results: SNS分析結果.
        hatena_results: はてブ分析結果.
        wordcloud_images: ソース名→画像パスのマッピング.
        ai_report: AI総評レポートテキスト.
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
        "wordcloud_images": wordcloud_images,
        "ai_report": ai_report,
    })
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_history() -> list[dict]:
    """履歴JSONを読み込む.

    Returns:
        履歴データのリスト（新しい順）.
    """
    if not HISTORY_PATH.exists():
        return []
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return list(reversed(history))


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


def render_hatena_tab(hatena_results: list[dict]) -> None:
    """はてブタブ内にコメントを表示する（過去データ用・expander無し）.

    Args:
        hatena_results: 感情分析済みコメントリスト.
    """
    st.caption("出典: はてなブックマーク (https://b.hatena.ne.jp)")
    for r in hatena_results:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[r["label"]]
        st.markdown(
            f"{emoji} {r['title'][:100]} "
            f"({r.get('author', '')}) (positive: {r['positive']:.1%})"
        )


def render_hatena_tab_live(hatena_data: list[dict], hatena_results: list[dict]) -> None:
    """はてブタブ内に記事ごとのexpander形式でコメントを表示する（ライブ分析用）.

    Args:
        hatena_data: 記事ごとのコメントデータ.
        hatena_results: 感情分析済みコメントリスト.
    """
    st.caption("出典: はてなブックマーク (https://b.hatena.ne.jp)")

    comments_by_url: dict[str, list[dict]] = {}
    for r in hatena_results:
        comments_by_url.setdefault(r["url"], []).append(r)

    for article_data in hatena_data:
        entry_url = article_data["url"].replace("https://", "").replace("http://", "")
        hatena_entry_url = f"https://b.hatena.ne.jp/entry/s/{entry_url}"

        with st.expander(
            f"📰 {article_data['title'][:60]} "
            f"（{article_data['bookmark_count']}ブックマーク）"
        ):
            st.markdown(f"▶ [はてなブックマークで見る]({hatena_entry_url})")
            st.markdown("---")
            article_comments = comments_by_url.get(article_data["url"], [])
            for c in article_comments:
                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[c["label"]]
                st.markdown(
                    f"{emoji} {c['title'][:100]} "
                    f"({c['author']}) (positive: {c['positive']:.1%})"
                )


def render_archived_analysis(entry: dict) -> None:
    """過去の分析結果を表示する.

    Args:
        entry: 履歴エントリ.
    """
    keyword = entry["keyword"]
    ts = datetime.fromisoformat(entry["timestamp"])
    st.warning(f"📂 過去の分析結果を表示中：{ts.strftime('%Y/%m/%d %H:%M')} [{keyword}]")

    news_results = entry.get("news_results", [])
    sns_results = entry.get("sns_results", [])
    hatena_results = entry.get("hatena_results", [])
    wordcloud_images = entry.get("wordcloud_images", {})
    ai_report = entry.get("ai_report", "")

    # 感情スコア表示
    news_stats = compute_sentiment_stats(news_results)
    bsky_stats = compute_sentiment_stats(sns_results) if sns_results else None
    hatena_stats = compute_sentiment_stats(hatena_results) if hatena_results else None

    st.subheader("🌡️ 世の中の空気感")

    active_sources: list[tuple[str, dict[str, float]]] = [("📰 メディア", news_stats)]
    if bsky_stats:
        active_sources.append(("💬 BlueSky", bsky_stats))
    if hatena_stats:
        active_sources.append(("📝 はてブ", hatena_stats))

    cols = st.columns(len(active_sources))
    for col, (source_name, stats) in zip(cols, active_sources):
        with col:
            st.markdown(f"#### {source_name}")
            render_sentiment_metrics(stats)

    st.divider()
    st.markdown(generate_insight(news_stats, bsky_stats, hatena_stats))

    # ワードクラウド画像表示
    st.subheader("☁️ ワードクラウド")
    if wordcloud_images:
        source_labels = {"news": "📰 メディア", "bsky": "💬 BlueSky", "hatena": "📝 はてブ"}
        available = [(k, v) for k, v in wordcloud_images.items() if Path(v).exists()]
        if available:
            wc_cols = st.columns(len(available))
            for col, (source, path) in zip(wc_cols, available):
                with col:
                    st.markdown(f"**{source_labels.get(source, source)}**")
                    st.image(path, use_container_width=True)
        else:
            st.info("ワードクラウド画像が見つかりません（古い履歴データの可能性があります）。")
    else:
        st.info("この分析にはワードクラウド画像が保存されていません。")

    # AI総評レポート
    if ai_report:
        st.divider()
        st.subheader("🤖 AIによる総合マーケット・インサイト")
        st.markdown(ai_report)

    # 記事一覧
    st.subheader("📰 記事・投稿一覧")
    tab_names: list[str] = ["📰 メディア"]
    tab_data_list: list[str] = ["news"]
    if sns_results:
        tab_names.append("💬 BlueSky")
        tab_data_list.append("bsky")
    if hatena_results:
        tab_names.append("📝 はてブ（第三者のコメント）")
        tab_data_list.append("hatena")

    tabs = st.tabs(tab_names)
    for tab, data_type in zip(tabs, tab_data_list):
        with tab:
            if data_type == "news":
                render_article_list(news_results)
            elif data_type == "bsky":
                render_article_list(sns_results, show_author=True)
            elif data_type == "hatena":
                render_hatena_tab(hatena_results)


def main() -> None:
    """Streamlit UIのメインエントリポイント."""
    st.set_page_config(page_title="TrendInsight AI", page_icon="📊", layout="wide")
    st.title("📊 TrendInsight AI")
    st.caption("メディア・SNS・はてブの多角的視点から世の中の空気感を読み取る")

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")

        # BlueSky認証
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

        # 過去の分析参照
        st.divider()
        st.subheader("📂 過去の分析を参照")
        history = load_history()
        history_options = ["（最新の分析）"] + [
            f"{datetime.fromisoformat(h['timestamp']).strftime('%Y-%m-%d %H:%M')} [{h['keyword']}]"
            for h in history
        ]
        selected_history = st.selectbox(
            "履歴を選択", history_options, index=0
        )

    # メイン: キーワード入力（常に表示）
    keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")
    run_clicked = st.button("分析開始", disabled=not keyword)

    # 過去データ表示モード（分析開始が押されていない場合のみ）
    if selected_history != "（最新の分析）" and not run_clicked:
        selected_idx = history_options.index(selected_history) - 1
        render_archived_analysis(history[selected_idx])
        st.stop()

    if run_clicked:
        # --- データ収集 ---
        news_client = GoogleNewsClient()
        bsky_client = BlueskyClient(handle=bsky_handle, app_password=bsky_password)
        hatena_client = HatenaClient()

        with st.spinner("📰 ニュース記事を収集中..."):
            news_articles = news_client.safe_fetch(keyword)
            if not news_articles:
                st.warning("ニュース記事の取得に失敗しました。")

        sns_articles: list[Article] = []
        if bsky_client.is_configured:
            with st.spinner("💬 BlueSky投稿を収集中..."):
                sns_articles = bsky_client.safe_fetch(keyword)
                if not sns_articles:
                    st.warning("BlueSky投稿の取得に失敗しました。")
        else:
            st.info("💡 サイドバーでBlueSky認証を設定すると、SNSの声も分析できます。")

        hatena_articles: list[Article] = []
        hatena_entry_data: list[dict] = []
        with st.spinner("📝 はてなブックマークのコメントを収集中..."):
            hatena_articles, hatena_entry_data = hatena_client.fetch_with_entries(keyword)

        if not news_articles and not sns_articles:
            st.warning("記事・投稿が見つかりませんでした。")
            return

        # --- 感情分析 ---
        def analyze_articles(articles: list[Article], label: str) -> list[dict]:
            """Articleリストに感情分析を適用し、dict化して返す."""
            results: list[dict] = []
            if not articles:
                return results
            progress = st.progress(0, text=f"{label}を分析中...")
            for i, article in enumerate(articles):
                scores = analyze_sentiment(article.title)
                results.append({**article.model_dump(), **scores})
                progress.progress((i + 1) / len(articles))
            progress.empty()
            return results

        news_results = analyze_articles(news_articles, "メディア記事")
        sns_results = analyze_articles(sns_articles, "SNS投稿")
        hatena_results = analyze_articles(hatena_articles, "はてブコメント")

        # --- 感情スコア表示 ---
        news_stats = compute_sentiment_stats(news_results)
        bsky_stats = compute_sentiment_stats(sns_results) if sns_results else None
        hatena_stats = compute_sentiment_stats(hatena_results) if hatena_results else None

        st.subheader("🌡️ 世の中の空気感")

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

        st.divider()
        st.markdown(generate_insight(news_stats, bsky_stats, hatena_stats))

        # --- ワードクラウド ---
        st.subheader("☁️ ワードクラウド")
        news_titles = [r["title"] for r in news_results]
        bsky_titles = [r["title"] for r in sns_results]
        hatena_texts = [r["title"] for r in hatena_results]

        source_titles = [("news", "📰 メディア", news_titles)]
        if sns_results:
            source_titles.append(("bsky", "💬 BlueSky", bsky_titles))
        if hatena_results:
            source_titles.append(("hatena", "📝 はてブ", hatena_texts))

        timestamp = datetime.now().isoformat()
        wordcloud_images: dict[str, str] = {}

        wc_cols = st.columns(len(source_titles))
        for col, (source_key, source_name, titles) in zip(wc_cols, source_titles):
            with col:
                st.markdown(f"**{source_name}**")
                wc = generate_wordcloud(extract_keywords(titles, keyword))
                if wc:
                    st.image(wc.to_array(), use_container_width=True)
                    path = save_wordcloud_image(wc, timestamp, source_key)
                    wordcloud_images[source_key] = path

        # --- トレンドキーワード TOP5 ---
        st.subheader("🔑 トレンド・キーワード TOP5")
        kw_cols = st.columns(len(source_titles))
        for col, (source_key, source_name, titles) in zip(kw_cols, source_titles):
            with col:
                st.markdown(f"**{source_name}**")
                for i, (word, count) in enumerate(
                    extract_keywords(titles, keyword)[:5], 1
                ):
                    st.markdown(f"**{i}.** {word}（{count}回）")

        # --- 記事・投稿一覧 ---
        st.subheader("📰 記事・投稿一覧")
        tab_names: list[str] = ["📰 メディア"]
        tab_data_list: list[str] = ["news"]
        if sns_results:
            tab_names.append("💬 BlueSky")
            tab_data_list.append("bsky")
        if hatena_entry_data:
            tab_names.append("📝 はてブ（第三者のコメント）")
            tab_data_list.append("hatena")

        tabs = st.tabs(tab_names)
        for tab, data_type in zip(tabs, tab_data_list):
            with tab:
                if data_type == "news":
                    render_article_list(news_results)
                elif data_type == "bsky":
                    render_article_list(sns_results, show_author=True)
                elif data_type == "hatena":
                    render_hatena_tab_live(hatena_entry_data, hatena_results)

        # --- AI総評レポート ---
        st.divider()
        st.subheader("🤖 AIによる総合マーケット・インサイト")
        ai_report = ""
        with st.spinner("🧠 AIが総評レポートを生成中（初回はモデルダウンロードのため数分かかります）..."):
            try:
                news_kw = extract_keywords(news_titles, keyword)
                bsky_kw = extract_keywords(bsky_titles, keyword) if sns_results else None
                hatena_kw = extract_keywords(hatena_texts, keyword) if hatena_results else None

                ai_report = generate_report(
                    keyword=keyword,
                    news_stats=news_stats,
                    news_keywords=news_kw,
                    bsky_stats=bsky_stats,
                    bsky_keywords=bsky_kw,
                    hatena_stats=hatena_stats,
                    hatena_keywords=hatena_kw,
                )
                st.markdown(ai_report)
            except Exception as e:
                st.warning(f"AI総評レポートの生成に失敗しました: {e}")

        # --- 履歴保存 ---
        save_history(
            keyword, news_results, sns_results, hatena_results,
            wordcloud_images, ai_report,
        )
        st.info("💾 分析結果を履歴に保存しました。")


if __name__ == "__main__":
    main()
