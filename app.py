"""TrendInsight AI - メディア・SNS・はてブの多角的トレンド感情分析アプリ."""

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.analyzer import (
    SentimentAnalyzer,
    compute_net_score,
    compute_sentiment_stats,
    select_representative,
)
from src.clients import BlueskyClient, GoogleNewsClient, HatenaClient
from src.models.article import Article
from src.reporter import generate_report
from src.services.history import load_history, save_history
from src.services.insight import generate_insight
from src.services.text_processor import create_tokenizer, extract_keywords
from src.services.analysis_type import detect_analysis_types
from src.services.topic_sentiment import compute_topic_sentiments
from src.services.wordcloud_generator import generate_wordcloud, save_wordcloud_image

load_dotenv()


@st.cache_resource
def _get_analyzer() -> SentimentAnalyzer:
    """SentimentAnalyzerをStreamlitキャッシュで保持する."""
    return SentimentAnalyzer()


@st.cache_resource
def _get_tokenizer():
    """Janomeトークナイザーをキャッシュで保持する."""
    return create_tokenizer()


# --- UI描画ヘルパー ---


def _score_label(score: float) -> str:
    """スコアから直感的なラベルを返す."""
    if score >= 0.3:
        return "ポジティブ優勢"
    elif score >= 0.1:
        return "ややポジ寄り"
    elif score > -0.1:
        return "中立的"
    elif score > -0.3:
        return "やや懸念寄り"
    return "ネガティブ優勢"


def render_sentiment_metrics(stats: dict[str, float]) -> None:
    """感情比率と温度計バーを表示する."""
    score = compute_net_score(stats)
    col1, col2, col3 = st.columns(3)
    col1.metric("ポジティブ", f"{stats['positive']:.1%}")
    col2.metric("中立", f"{stats['neutral']:.1%}")
    col3.metric("ネガティブ", f"{stats['negative']:.1%}")

    # 温度計バー（テキスト描画）
    label = _score_label(score)
    bar_len = 20
    pos = int(round((score + 1) / 2 * bar_len))
    pos = max(0, min(bar_len, pos))
    bar = "━" * pos + "●" + "━" * (bar_len - pos)
    st.text(f"ネガ ◀{bar}▶ ポジ")
    st.caption(f"スコア: {score:+.2f}（{label}）")


def render_article_list(results: list[dict], show_author: bool = False) -> None:
    """記事/投稿一覧を表示する."""
    for r in results:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[r["label"]]
        author = f" ({r['author']})" if show_author and "author" in r else ""
        st.markdown(
            f"{emoji} [{r['title'][:80]}]({r['url']}){author} "
            f"(positive: {r['positive']:.1%})"
        )


def render_hatena_tab(hatena_results: list[dict]) -> None:
    """はてブタブ内にコメントを表示する（過去データ用・expander無し）."""
    st.caption("出典: はてなブックマーク (https://b.hatena.ne.jp)")
    for r in hatena_results:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[r["label"]]
        st.markdown(
            f"{emoji} {r['title'][:100]} "
            f"({r.get('author', '')}) (positive: {r['positive']:.1%})"
        )


def render_hatena_tab_live(hatena_data: list[dict], hatena_results: list[dict]) -> None:
    """はてブタブ内に記事ごとのexpander形式でコメントを表示する（ライブ分析用）."""
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
    """過去の分析結果を表示する."""
    keyword = entry["keyword"]
    ts = datetime.fromisoformat(entry["timestamp"])
    st.warning(f"📂 過去の分析結果を表示中：{ts.strftime('%Y/%m/%d %H:%M')} [{keyword}]")

    news_results = entry.get("news_results", [])
    sns_results = entry.get("sns_results", [])
    hatena_results = entry.get("hatena_results", [])
    wordcloud_images = entry.get("wordcloud_images", {})
    ai_report = entry.get("ai_report", "")

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


# --- メインフロー ---


def main() -> None:
    """Streamlit UIのメインエントリポイント."""
    st.set_page_config(page_title="TrendInsight AI", page_icon="📊", layout="wide")
    st.title("📊 TrendInsight AI")
    st.caption("メディア・SNS・はてブの多角的視点から世の中の空気感を読み取る")

    # サイドバー
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
        st.subheader("📂 過去の分析を参照")
        history = load_history()
        history_options = ["（最新の分析）"] + [
            f"{datetime.fromisoformat(h['timestamp']).strftime('%Y-%m-%d %H:%M')} [{h['keyword']}]"
            for h in history
        ]
        selected_history = st.selectbox("履歴を選択", history_options, index=0)

    # メイン: キーワード入力
    keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")
    run_clicked = st.button("分析開始", disabled=not keyword)

    # 過去データ表示モード
    if selected_history != "（最新の分析）" and not run_clicked:
        selected_idx = history_options.index(selected_history) - 1
        render_archived_analysis(history[selected_idx])
        st.stop()

    if run_clicked:
        _run_analysis(keyword, bsky_handle, bsky_password)


def _run_analysis(keyword: str, bsky_handle: str, bsky_password: str) -> None:
    """分析メインフローを実行する."""
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
    analyzer = _get_analyzer()
    tokenizer = _get_tokenizer()

    def analyze_articles(articles: list[Article], label: str) -> list[dict]:
        """Articleリストに感情分析を適用し、dict化して返す."""
        results: list[dict] = []
        if not articles:
            return results
        progress = st.progress(0, text=f"{label}を分析中...")
        for i, article in enumerate(articles):
            scores = analyzer.analyze(article.title)
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

    # --- 分析タイプ判定用の事前計算 ---
    net_scores: dict[str, float] = {"news": compute_net_score(news_stats)}
    neutral_ratios: dict[str, float] = {"news": news_stats["neutral"]}
    if bsky_stats:
        net_scores["bsky"] = compute_net_score(bsky_stats)
        neutral_ratios["bsky"] = bsky_stats["neutral"]
    if hatena_stats:
        net_scores["hatena"] = compute_net_score(hatena_stats)
        neutral_ratios["hatena"] = hatena_stats["neutral"]

    # --- 代表コメント ---
    st.subheader("💬 代表的な意見")
    news_samples = select_representative(news_results)
    bsky_samples = select_representative(sns_results) if sns_results else None
    hatena_samples = select_representative(hatena_results) if hatena_results else None

    sample_sources = [("📰 メディア", news_samples)]
    if bsky_samples:
        sample_sources.append(("💬 BlueSky", bsky_samples))
    if hatena_samples:
        sample_sources.append(("📝 はてブ", hatena_samples))

    sample_cols = st.columns(len(sample_sources))
    for col, (src_name, samples) in zip(sample_cols, sample_sources):
        with col:
            st.markdown(f"**{src_name}**")
            for s in samples.get("positive", []):
                st.markdown(f"🟢 {s}")
            for s in samples.get("negative", []):
                st.markdown(f"🔴 {s}")

    # --- テキストリスト（トピック分析・ワードクラウド共通） ---
    news_titles = [r["title"] for r in news_results]
    bsky_titles = [r["title"] for r in sns_results]
    hatena_texts = [r["title"] for r in hatena_results]

    # メディア名を動的ストップワードとして抽出（GoogleニュースRSSの「タイトル - メディア名」形式から）
    # メディア名全体 + スペース分割した各トークンも追加（Janomeが個別トークンに分割するため）
    news_media_names: set[str] = set()
    for title in news_titles:
        if " - " in title:
            media = title.rsplit(" - ", 1)[-1].strip()
            if media:
                news_media_names.add(media)
                for token in media.split():
                    if len(token) > 1:
                        news_media_names.add(token)

    # --- トピック別感情分析 ---
    st.subheader("🎯 トピック別感情分析")
    st.caption("各話題がポジティブ／ネガティブどちらに寄与しているかを表示")

    news_kw_for_topic = extract_keywords(
        news_titles, keyword, tokenizer=tokenizer, extra_stop_words=news_media_names
    )
    bsky_kw_for_topic = (
        extract_keywords(bsky_titles, keyword, tokenizer=tokenizer)
        if sns_results else None
    )
    hatena_kw_for_topic = (
        extract_keywords(hatena_texts, keyword, tokenizer=tokenizer)
        if hatena_results else None
    )

    topic_sources = [("📰 メディア", news_results, news_kw_for_topic)]
    if sns_results and bsky_kw_for_topic:
        topic_sources.append(("💬 BlueSky", sns_results, bsky_kw_for_topic))
    if hatena_results and hatena_kw_for_topic:
        topic_sources.append(("📝 はてブ", hatena_results, hatena_kw_for_topic))

    topic_cols = st.columns(len(topic_sources))
    all_topic_sentiments: dict[str, list[dict]] = {}
    for col, (src_name, src_results, src_kw) in zip(topic_cols, topic_sources):
        with col:
            st.markdown(f"**{src_name}**")
            topics = compute_topic_sentiments(src_results, src_kw)
            # ソースキー保存（LLMプロンプト用）
            key = src_name.split(" ")[1] if " " in src_name else src_name
            all_topic_sentiments[key] = topics
            if topics:
                for t in topics[:8]:
                    score = t["net_score"]
                    emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "⚪"
                    bar_width = int(abs(score) * 5)
                    bar = "█" * max(bar_width, 1)
                    st.markdown(
                        f"{emoji} **{t['topic']}** {score:+.2f} {bar}  "
                        f"(pos:{t['pos']} neg:{t['neg']} 計:{t['count']})"
                    )
            else:
                st.caption("十分なデータがありません")

    # --- 分析タイプ判定 ---
    combined_topics = [t for topics in all_topic_sentiments.values() for t in topics]

    from src.services.insight import compute_divergences
    divergences = compute_divergences(net_scores) if len(net_scores) >= 2 else []
    max_div = divergences[0][2] if divergences else 0.0

    analysis_types = detect_analysis_types(
        net_scores=net_scores,
        max_divergence=max_div,
        topic_sentiments=combined_topics,
        neutral_ratios=neutral_ratios,
        sample_counts={
            "news": len(news_results),
            "bsky": len(sns_results),
            "hatena": len(hatena_results),
        },
    )

    st.subheader("📋 分析タイプ")
    for at in analysis_types:
        st.markdown(f"{at['emoji']} **{at['label']}** — {at['reason']}")

    # --- ワードクラウド ---
    st.subheader("☁️ ワードクラウド")

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
            wc = generate_wordcloud(
                extract_keywords(
                    titles, keyword, tokenizer=tokenizer,
                    extra_stop_words=news_media_names if source_key == "news" else None,
                )
            )
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
                extract_keywords(
                    titles, keyword, tokenizer=tokenizer,
                    extra_stop_words=news_media_names if source_key == "news" else None,
                )[:5], 1
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
            news_kw = extract_keywords(
                news_titles, keyword, tokenizer=tokenizer,
                extra_stop_words=news_media_names,
            )
            bsky_kw = (
                extract_keywords(bsky_titles, keyword, tokenizer=tokenizer)
                if sns_results
                else None
            )
            hatena_kw = (
                extract_keywords(hatena_texts, keyword, tokenizer=tokenizer)
                if hatena_results
                else None
            )

            ai_report = generate_report(
                keyword=keyword,
                news_stats=news_stats,
                news_keywords=news_kw,
                news_count=len(news_results),
                news_samples=news_samples,
                bsky_stats=bsky_stats,
                bsky_keywords=bsky_kw,
                bsky_count=len(sns_results),
                bsky_samples=bsky_samples,
                hatena_stats=hatena_stats,
                hatena_keywords=hatena_kw,
                hatena_count=len(hatena_results),
                hatena_samples=hatena_samples,
                topic_sentiments=all_topic_sentiments,
                analysis_types=analysis_types,
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
