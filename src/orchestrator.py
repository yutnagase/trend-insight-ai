"""分析オーケストレーター - 実行制御とUI通知の仲介."""

import logging

import streamlit as st

from src.models.analysis_result import AnalysisResult
from src.protocols import (
    DataCollectorProtocol,
    HistoryRepositoryProtocol,
    ReportGeneratorProtocol,
    SentimentAnalyzerProtocol,
)
from src.services.analysis_pipeline import run_analysis
from src.ui.pages import render_live_analysis

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """分析フロー全体の実行制御を担うクラス.

    全依存をコンストラクタで受け取り、具象実装への直接依存を持たない。
    テスト時はProtocol準拠のモックを注入するだけで、
    Streamlit/LLM/ファイルI/O無しにロジックを検証できる。
    """

    def __init__(
        self,
        analyzer: SentimentAnalyzerProtocol,
        collector: DataCollectorProtocol,
        report_generator: ReportGeneratorProtocol,
        history_repository: HistoryRepositoryProtocol,
    ) -> None:
        self._analyzer = analyzer
        self._collector = collector
        self._report_generator = report_generator
        self._history = history_repository

    def execute(self, keyword: str) -> None:
        """分析フローを実行し、結果を表示・保存する."""
        articles = self._collect(keyword)
        if articles is None:
            return

        news_articles, sns_articles, hatena_articles, hatena_entry_data = articles

        result = self._analyze(
            keyword, news_articles, sns_articles, hatena_articles, hatena_entry_data
        )

        # AI総評以外を先に描画し、ユーザーを待たせない
        render_live_analysis(result)

        # AI総評は最後に生成・表示
        self._generate_ai_report(result)
        self._render_ai_report(result)
        self._save(result)

    def _collect(self, keyword: str) -> tuple | None:
        """データ収集フェーズ."""
        with st.spinner("📰 データを収集中..."):
            news, sns, hatena, hatena_entries = self._collector.collect(keyword)

        if not news and not sns:
            st.warning("記事・投稿が見つかりませんでした。")
            return None

        if not news:
            st.warning("ニュース記事の取得に失敗しました。")
        if not sns and hasattr(self._collector, "bsky_configured"):
            if self._collector.bsky_configured:
                st.warning("BlueSky投稿の取得に失敗しました。")
            else:
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
                result.ai_report = self._report_generator.generate(result)
            except Exception as e:
                logger.error("AI総評生成失敗: %s", e, exc_info=True)
                st.warning(f"AI総評レポートの生成に失敗しました: {e}")

    def _render_ai_report(self, result: AnalysisResult) -> None:
        """AI総評レポート表示フェーズ."""
        st.divider()
        st.subheader("🤖 AIによる総合マーケット・インサイト")
        if result.ai_report:
            st.markdown(result.ai_report)

    def _save(self, result: AnalysisResult) -> None:
        """履歴保存フェーズ."""
        self._history.save(result)
        st.info("💾 分析結果を履歴に保存しました。")
