"""分析オーケストレーター - 実行制御とUI通知の仲介."""

import logging

import streamlit as st

from src.exceptions import (
    AnalysisPipelineError,
    BlueskyAuthError,
    DataCollectionError,
    HistorySaveError,
    ReportGenerationError,
    TrendInsightError,
)
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


def _show_error(error: TrendInsightError) -> None:
    """ユーザー向けエラーメッセージを表示する."""
    st.error(f"⚠️ {error.user_message}")
    st.info(f"💡 {error.user_hint}")


class AnalysisOrchestrator:
    """分析フロー全体の実行制御を担うクラス.

    全依存をコンストラクタで受け取り、具象実装への直接依存を持たない。
    各フェーズでTrendInsightError系例外をcatchし、
    ユーザーフレンドリーなメッセージを表示する。
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
        # データ収集
        articles = self._collect(keyword)
        if articles is None:
            return

        news_articles, sns_articles, hatena_articles, hatena_entry_data = articles

        # 感情分析
        result = self._analyze(
            keyword, news_articles, sns_articles, hatena_articles, hatena_entry_data
        )
        if result is None:
            return

        # AI総評以外を先に描画し、ユーザーを待たせない
        render_live_analysis(result)

        # AI総評は最後に生成・表示
        self._generate_ai_report(result)
        self._render_ai_report(result)
        self._save(result)

    def _collect(self, keyword: str) -> tuple | None:
        """データ収集フェーズ."""
        with st.spinner("📰 データを収集中..."):
            try:
                news, sns, hatena, hatena_entries = self._collector.collect(keyword)
            except BlueskyAuthError as e:
                _show_error(e)
                # BlueSky認証失敗でも他ソースで続行を試みる
                try:
                    news, sns, hatena, hatena_entries = self._collector.collect(keyword)
                except TrendInsightError:
                    return None
                except Exception as exc:
                    logger.error("データ収集失敗: %s", exc, exc_info=True)
                    _show_error(DataCollectionError(str(exc)))
                    return None
            except DataCollectionError as e:
                _show_error(e)
                return None
            except Exception as e:
                logger.error("データ収集で予期しないエラー: %s", e, exc_info=True)
                _show_error(DataCollectionError(str(e)))
                return None

        if not news and not sns:
            st.warning(
                "📭 記事・投稿が見つかりませんでした。\n\n"
                "💡 キーワードを変えるか、より一般的な表現でお試しください。"
            )
            return None

        if not news:
            st.warning("ニュース記事が見つかりませんでした。はてブ・SNSのデータで分析を続行します。")
        if not sns and hasattr(self._collector, "bsky_configured"):
            if self._collector.bsky_configured:
                st.warning(
                    "BlueSky投稿が見つかりませんでした。メディア・はてブのデータで分析を続行します。"
                )
            else:
                st.info("💡 サイドバーでBlueSky認証を設定すると、SNSの声も分析できます。")

        return news, sns, hatena, hatena_entries

    def _analyze(
        self, keyword, news, sns, hatena, hatena_entries
    ) -> AnalysisResult | None:
        """感情分析パイプライン実行フェーズ."""
        progress_bar = st.progress(0, text="分析中...")

        def on_progress(label: str, current: int, total: int) -> None:
            progress_bar.progress(current / total, text=f"{label}を分析中...")

        try:
            result = run_analysis(
                keyword=keyword,
                news_articles=news,
                sns_articles=sns,
                hatena_articles=hatena,
                hatena_entry_data=hatena_entries,
                analyzer=self._analyzer,
                progress_callback=on_progress,
            )
        except Exception as e:
            progress_bar.empty()
            logger.error("分析パイプライン失敗: %s", e, exc_info=True)
            _show_error(AnalysisPipelineError(str(e)))
            return None

        progress_bar.empty()
        return result

    def _generate_ai_report(self, result: AnalysisResult) -> None:
        """AI総評レポート生成フェーズ."""
        with st.spinner("🧠 AIが総評レポートを生成中（初回はモデルダウンロードのため数分かかります）..."):
            try:
                result.ai_report = self._report_generator.generate(result)
            except ReportGenerationError as e:
                _show_error(e)
            except Exception as e:
                logger.error("AI総評生成で予期しないエラー: %s", e, exc_info=True)
                _show_error(ReportGenerationError(str(e)))

    def _render_ai_report(self, result: AnalysisResult) -> None:
        """AI総評レポート表示フェーズ."""
        st.divider()
        st.subheader("🤖 AIによる総合マーケット・インサイト")
        if result.ai_report:
            st.markdown(result.ai_report)
        else:
            st.caption("AI総評は生成されませんでした。上記の分析結果をご参照ください。")

    def _save(self, result: AnalysisResult) -> None:
        """履歴保存フェーズ."""
        try:
            self._history.save(result)
            st.info("💾 分析結果を履歴に保存しました。")
        except HistorySaveError as e:
            _show_error(e)
        except Exception as e:
            logger.error("履歴保存で予期しないエラー: %s", e, exc_info=True)
            _show_error(HistorySaveError(str(e)))
