"""分析オーケストレーター - 実行制御とUI通知の仲介."""

import logging
from typing import Protocol

import streamlit as st

from src.analyzer import SentimentAnalyzer
from src.models.analysis_result import AnalysisResult
from src.reporter import generate_report
from src.services.analysis_pipeline import collect_data, run_analysis
from src.services.history import save_history
from src.ui.pages import render_live_analysis

logger = logging.getLogger(__name__)


class ProgressNotifier(Protocol):
    """進捗通知のインターフェース."""

    def on_progress(self, label: str, current: int, total: int) -> None: ...


class AnalysisOrchestrator:
    """分析フロー全体の実行制御を担うクラス.

    責務: データ収集 → 感情分析 → AI総評 → 表示 → 履歴保存
    """

    def __init__(self, analyzer: SentimentAnalyzer) -> None:
        self._analyzer = analyzer

    def execute(self, keyword: str, bsky_handle: str, bsky_password: str) -> None:
        """分析フローを実行し、結果を表示・保存する."""
        articles = self._collect(keyword, bsky_handle, bsky_password)
        if articles is None:
            return

        news_articles, sns_articles, hatena_articles, hatena_entry_data = articles

        result = self._analyze(
            keyword, news_articles, sns_articles, hatena_articles, hatena_entry_data
        )
        self._generate_ai_report(result)
        render_live_analysis(result)
        self._save(result)

    def _collect(
        self, keyword: str, bsky_handle: str, bsky_password: str
    ) -> tuple | None:
        """データ収集フェーズ."""
        with st.spinner("📰 データを収集中..."):
            news, sns, hatena, hatena_entries = collect_data(
                keyword, bsky_handle, bsky_password
            )

        if not news and not sns:
            st.warning("記事・投稿が見つかりませんでした。")
            return None

        if not news:
            st.warning("ニュース記事の取得に失敗しました。")
        if not sns and bsky_handle:
            st.warning("BlueSky投稿の取得に失敗しました。")
        elif not bsky_handle:
            st.info("💡 サイドバーでBlueSky認証を設定すると、SNSの声も分析できます。")

        return news, sns, hatena, hatena_entries

    def _analyze(self, keyword, news, sns, hatena, hatena_entries) -> AnalysisResult:
        """感情分析パイプライン実行フェーズ."""
        progress_bar = st.progress(0, text="分析中...")

        def on_progress(label: str, current: int, total: int) -> None:
            progress_bar.progress(current / total, text=f"{label}を分析中...")

        result = run_analysis(
            keyword=keyword,
            news_articles=news,
            sns_articles=sns,
            hatena_articles=hatena,
            hatena_entry_data=hatena_entries,
            analyzer=self._analyzer,
            progress_callback=on_progress,
        )
        progress_bar.empty()
        return result

    def _generate_ai_report(self, result: AnalysisResult) -> None:
        """AI総評レポート生成フェーズ."""
        with st.spinner("🧠 AIが総評レポートを生成中（初回はモデルダウンロードのため数分かかります）..."):
            try:
                result.ai_report = generate_report(
                    keyword=result.keyword,
                    news_stats=result.news.stats.model_dump() if result.news.stats else {},
                    news_keywords=result.news.keywords,
                    news_count=len(result.news.results),
                    news_samples=result.news.samples or None,
                    bsky_stats=result.bsky.stats.model_dump() if result.bsky.stats else None,
                    bsky_keywords=result.bsky.keywords or None,
                    bsky_count=len(result.bsky.results),
                    bsky_samples=result.bsky.samples or None,
                    hatena_stats=result.hatena.stats.model_dump() if result.hatena.stats else None,
                    hatena_keywords=result.hatena.keywords or None,
                    hatena_count=len(result.hatena.results),
                    hatena_samples=result.hatena.samples or None,
                    topic_sentiments=result.topic_sentiments,
                    analysis_types=result.analysis_types,
                )
            except Exception as e:
                logger.error("AI総評生成失敗: %s", e, exc_info=True)
                st.warning(f"AI総評レポートの生成に失敗しました: {e}")

    def _save(self, result: AnalysisResult) -> None:
        """履歴保存フェーズ."""
        save_history(
            result.keyword,
            [r.model_dump() for r in result.news.results],
            [r.model_dump() for r in result.bsky.results],
            [r.model_dump() for r in result.hatena.results],
            result.wordcloud_images,
            result.ai_report,
        )
        st.info("💾 分析結果を履歴に保存しました。")
