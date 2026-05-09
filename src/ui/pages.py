"""ページ描画ロジック - ライブ分析・過去分析の表示."""

from datetime import datetime
from typing import Any

import streamlit as st

from src.analyzer import compute_sentiment_stats
from src.models.analysis_result import AnalysisResult
from src.services.insight import generate_insight
from src.ui.components import (
    render_analysis_types,
    render_article_tabs,
    render_representative_comments,
    render_source_metrics,
    render_topic_sentiments,
    render_trend_keywords,
    render_wordclouds,
    render_wordclouds_live,
)


def _build_active_sources(result: AnalysisResult) -> list[tuple[str, dict[str, float]]]:
    """表示用のアクティブソースリストを構築する."""
    sources: list[tuple[str, dict[str, float]]] = []
    if result.news.stats:
        sources.append(("📰 メディア", result.news.stats.model_dump()))
    if result.bsky.stats:
        sources.append(("💬 BlueSky", result.bsky.stats.model_dump()))
    if result.hatena.stats:
        sources.append(("📝 はてブ", result.hatena.stats.model_dump()))
    return sources


def render_live_analysis(result: AnalysisResult) -> None:
    """ライブ分析結果を表示する."""
    active_sources = _build_active_sources(result)

    # 感情メトリクス
    st.subheader("🌡️ 世の中の空気感")
    render_source_metrics(active_sources)

    st.divider()
    news_stats = result.news.stats.model_dump() if result.news.stats else None
    bsky_stats = result.bsky.stats.model_dump() if result.bsky.stats else None
    hatena_stats = result.hatena.stats.model_dump() if result.hatena.stats else None
    if news_stats:
        st.markdown(generate_insight(news_stats, bsky_stats, hatena_stats))

    # 代表コメント
    st.subheader("💬 代表的な意見")
    sample_sources = []
    if result.news.samples:
        sample_sources.append(("📰 メディア", result.news.samples))
    if result.bsky.samples:
        sample_sources.append(("💬 BlueSky", result.bsky.samples))
    if result.hatena.samples:
        sample_sources.append(("📝 はてブ", result.hatena.samples))
    if sample_sources:
        render_representative_comments(sample_sources)

    # トピック別感情分析
    st.subheader("🎯 トピック別感情分析")
    st.caption("各話題がポジティブ／ネガティブどちらに寄与しているかを表示")
    render_topic_sentiments(result.topic_sentiments)

    # 分析タイプ
    st.subheader("📋 分析タイプ")
    render_analysis_types(result.analysis_types)

    # ワードクラウド
    st.subheader("☁️ ワードクラウド")
    render_wordclouds_live(result)

    # トレンドキーワード
    st.subheader("🔑 トレンド・キーワード TOP5")
    render_trend_keywords(result)

    # 記事一覧
    st.subheader("📰 記事・投稿一覧")
    render_article_tabs(result, live=True)


def render_archived_analysis(entry: dict[str, Any]) -> None:
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

    # 感情メトリクス
    st.subheader("🌡️ 世の中の空気感")
    active_sources: list[tuple[str, dict[str, float]]] = [("📰 メディア", news_stats)]
    if bsky_stats:
        active_sources.append(("💬 BlueSky", bsky_stats))
    if hatena_stats:
        active_sources.append(("📝 はてブ", hatena_stats))
    render_source_metrics(active_sources)

    st.divider()
    st.markdown(generate_insight(news_stats, bsky_stats, hatena_stats))

    # ワードクラウド
    st.subheader("☁️ ワードクラウド")
    if wordcloud_images:
        render_wordclouds(wordcloud_images)
    else:
        st.info("この分析にはワードクラウド画像が保存されていません。")

    # AI総評
    if ai_report:
        st.divider()
        st.subheader("🤖 AIによる総合マーケット・インサイト")
        st.markdown(ai_report)

    # 記事一覧（過去データ用の簡易表示）
    st.subheader("📰 記事・投稿一覧")
    from src.ui.components import _render_hatena_tab_simple, render_article_list

    tab_names: list[str] = ["📰 メディア"]
    tab_data_list: list[str] = ["news"]
    if sns_results:
        tab_names.append("💬 BlueSky")
        tab_data_list.append("bsky")
    if hatena_results:
        tab_names.append("📝 はてブ（第三者のコメント）")
        tab_data_list.append("hatena")

    tabs = st.tabs(tab_names)
    for tab, data_type in zip(tabs, tab_data_list, strict=False):
        with tab:
            if data_type == "news":
                render_article_list(news_results)
            elif data_type == "bsky":
                render_article_list(sns_results, show_author=True)
            elif data_type == "hatena":
                _render_hatena_tab_simple(hatena_results)
