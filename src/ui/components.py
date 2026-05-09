"""再利用可能なStreamlit UIコンポーネント."""

from pathlib import Path

import streamlit as st

from src.analyzer import compute_net_score
from src.models.analysis_result import AnalysisResult


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


def render_sentiment_metrics(stats_dict: dict[str, float]) -> None:
    """感情比率と温度計バーを表示する."""
    score = compute_net_score(stats_dict)
    col1, col2, col3 = st.columns(3)
    col1.metric("ポジティブ", f"{stats_dict['positive']:.1%}")
    col2.metric("中立", f"{stats_dict['neutral']:.1%}")
    col3.metric("ネガティブ", f"{stats_dict['negative']:.1%}")

    label = _score_label(score)
    bar_len = 20
    pos = round((score + 1) / 2 * bar_len)
    pos = max(0, min(bar_len, pos))
    bar = "━" * pos + "●" + "━" * (bar_len - pos)
    st.text(f"ネガ ◀{bar}▶ ポジ")
    st.caption(f"スコア: {score:+.2f}（{label}）")


def render_source_metrics(sources: list[tuple[str, dict[str, float]]]) -> None:
    """複数ソースの感情メトリクスを横並びで表示する."""
    cols = st.columns(len(sources))
    for col, (source_name, stats) in zip(cols, sources, strict=False):
        with col:
            st.markdown(f"#### {source_name}")
            render_sentiment_metrics(stats)


def render_article_list(results: list[dict], show_author: bool = False) -> None:
    """記事/投稿一覧を表示する."""
    for r in results:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[r["label"]]
        author = f" ({r['author']})" if show_author and "author" in r else ""
        st.markdown(
            f"{emoji} [{r['title'][:80]}]({r['url']}){author} (positive: {r['positive']:.1%})"
        )


def render_representative_comments(
    sources: list[tuple[str, dict[str, list[str]]]],
) -> None:
    """代表コメントを横並びで表示する."""
    cols = st.columns(len(sources))
    for col, (src_name, samples) in zip(cols, sources, strict=False):
        with col:
            st.markdown(f"**{src_name}**")
            for s in samples.get("positive", []):
                st.markdown(f"🟢 {s}")
            for s in samples.get("negative", []):
                st.markdown(f"🔴 {s}")


def render_topic_sentiments(topic_sentiments: dict[str, list[dict]]) -> None:
    """トピック別感情分析を表示する."""
    source_labels = {"メディア": "📰 メディア", "BlueSky": "💬 BlueSky", "はてブ": "📝 はてブ"}
    active = [(k, v) for k, v in topic_sentiments.items() if v]
    if not active:
        st.caption("十分なデータがありません")
        return

    cols = st.columns(len(active))
    for col, (src_key, topics) in zip(cols, active, strict=False):
        with col:
            st.markdown(f"**{source_labels.get(src_key, src_key)}**")
            for t in topics[:8]:
                score = t["net_score"]
                emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "⚪"
                bar_width = int(abs(score) * 5)
                bar = "█" * max(bar_width, 1)
                st.markdown(
                    f"{emoji} **{t['topic']}** {score:+.2f} {bar}  "
                    f"(pos:{t['pos']} neg:{t['neg']} 計:{t['count']})"
                )


def render_analysis_types(analysis_types: list[dict]) -> None:
    """分析タイプを表示する."""
    for at in analysis_types:
        st.markdown(f"{at['emoji']} **{at['label']}** — {at['reason']}")


def render_wordclouds(wordcloud_images: dict[str, str]) -> None:
    """ワードクラウド画像を表示する."""
    source_labels = {"news": "📰 メディア", "bsky": "💬 BlueSky", "hatena": "📝 はてブ"}
    available = [(k, v) for k, v in wordcloud_images.items() if Path(v).exists()]
    if not available:
        st.info("ワードクラウド画像が見つかりません。")
        return
    cols = st.columns(len(available))
    for col, (source, path) in zip(cols, available, strict=False):
        with col:
            st.markdown(f"**{source_labels.get(source, source)}**")
            st.image(path, use_container_width=True)


def render_wordclouds_live(
    result: AnalysisResult,
) -> None:
    """ライブ分析時のワードクラウドを表示する（画像生成済み前提）."""
    from src.services.wordcloud_generator import generate_wordcloud

    source_labels = {"news": "📰 メディア", "bsky": "💬 BlueSky", "hatena": "📝 はてブ"}
    sources = [
        ("news", result.news),
        ("bsky", result.bsky),
        ("hatena", result.hatena),
    ]
    active = [(k, s) for k, s in sources if s.keywords]
    if not active:
        return

    cols = st.columns(len(active))
    for col, (source_key, src_analysis) in zip(cols, active, strict=False):
        with col:
            st.markdown(f"**{source_labels[source_key]}**")
            wc = generate_wordcloud(src_analysis.keywords)
            if wc:
                st.image(wc.to_array(), use_container_width=True)


def render_trend_keywords(result: AnalysisResult) -> None:
    """トレンドキーワードTOP5を表示する."""
    source_labels = {"news": "📰 メディア", "bsky": "💬 BlueSky", "hatena": "📝 はてブ"}
    sources = [
        ("news", result.news),
        ("bsky", result.bsky),
        ("hatena", result.hatena),
    ]
    active = [(k, s) for k, s in sources if s.keywords]
    if not active:
        return

    cols = st.columns(len(active))
    for col, (source_key, src_analysis) in zip(cols, active, strict=False):
        with col:
            st.markdown(f"**{source_labels[source_key]}**")
            for i, (word, count) in enumerate(src_analysis.keywords[:5], 1):
                st.markdown(f"**{i}.** {word}（{count}回）")


def render_hatena_tab_live(hatena_data: list[dict], hatena_results: list[dict]) -> None:
    """はてブタブ内に記事ごとのexpander形式でコメントを表示する."""
    st.caption("出典: はてなブックマーク (https://b.hatena.ne.jp)")

    comments_by_url: dict[str, list[dict]] = {}
    for r in hatena_results:
        comments_by_url.setdefault(r["url"], []).append(r)

    for article_data in hatena_data:
        entry_url = article_data["url"].replace("https://", "").replace("http://", "")
        hatena_entry_url = f"https://b.hatena.ne.jp/entry/s/{entry_url}"

        with st.expander(
            f"📰 {article_data['title'][:60]} （{article_data['bookmark_count']}ブックマーク）"
        ):
            st.markdown(f"▶ [はてなブックマークで見る]({hatena_entry_url})")
            st.markdown("---")
            article_comments = comments_by_url.get(article_data["url"], [])
            for c in article_comments:
                emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[c["label"]]
                st.markdown(
                    f"{emoji} {c['title'][:100]} ({c['author']}) (positive: {c['positive']:.1%})"
                )


def render_article_tabs(
    result: AnalysisResult,
    live: bool = True,
) -> None:
    """記事・投稿一覧をタブ形式で表示する."""
    news_dicts = [r.model_dump() for r in result.news.results]
    sns_dicts = [r.model_dump() for r in result.bsky.results]
    hatena_dicts = [r.model_dump() for r in result.hatena.results]

    tab_names: list[str] = ["📰 メディア"]
    tab_data_list: list[str] = ["news"]
    if sns_dicts:
        tab_names.append("💬 BlueSky")
        tab_data_list.append("bsky")
    if hatena_dicts:
        tab_names.append("📝 はてブ（第三者のコメント）")
        tab_data_list.append("hatena")

    tabs = st.tabs(tab_names)
    for tab, data_type in zip(tabs, tab_data_list, strict=False):
        with tab:
            if data_type == "news":
                render_article_list(news_dicts)
            elif data_type == "bsky":
                render_article_list(sns_dicts, show_author=True)
            elif data_type == "hatena":
                if live and result.hatena_entry_data:
                    render_hatena_tab_live(result.hatena_entry_data, hatena_dicts)
                else:
                    _render_hatena_tab_simple(hatena_dicts)


def _render_hatena_tab_simple(hatena_results: list[dict]) -> None:
    """はてブタブ（過去データ用・expander無し）."""
    st.caption("出典: はてなブックマーク (https://b.hatena.ne.jp)")
    for r in hatena_results:
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}[r["label"]]
        st.markdown(
            f"{emoji} {r['title'][:100]} ({r.get('author', '')}) (positive: {r['positive']:.1%})"
        )
