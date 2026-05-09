"""分析オーケストレーター - 実行制御とUI通知の仲介."""

import time
from concurrent.futures import Future, ThreadPoolExecutor

import streamlit as st
import structlog

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

logger = structlog.get_logger(__name__)


def _show_error(error: TrendInsightError) -> None:
    """ユーザー向けエラーメッセージを表示する."""
    st.error(f"⚠️ {error.user_message}")
    st.info(f"💡 {error.user_hint}")


class AnalysisOrchestrator:
    """分析フロー全体の実行制御を担うクラス.

    全依存をコンストラクタで受け取り、具象実装への直接依存を持たない。
    各フェーズでTrendInsightError系例外をcatchし、
    ユーザーフレンドリーなメッセージを表示する。

    パフォーマンス最適化:
    - データ収集: 3ソースを並列フェッチ（adapters層で実装）
    - AI総評生成: UI描画と並行してバックグラウンド実行
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
        log = logger.bind(keyword=keyword)
        log.info("分析フロー開始")
        start = time.perf_counter()

        # データ収集（3ソース並列）
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

        # AI総評をバックグラウンドで開始し、UI描画と並行させる
        report_future = self._start_ai_report_async(result)

        # UI描画（AI総評以外）— LLM推論と並行して実行される
        render_live_analysis(result)

        # AI総評の結果を取得・表示
        self._await_ai_report(result, report_future)
        self._render_ai_report(result)
        self._save(result)

        elapsed = time.perf_counter() - start
        log.info("分析フロー完了", elapsed_sec=round(elapsed, 2))

    def _collect(self, keyword: str) -> tuple | None:
        """データ収集フェーズ."""
        log = logger.bind(phase="collect", keyword=keyword)
        start = time.perf_counter()

        with st.spinner("📰 データを収集中..."):
            try:
                news, sns, hatena, hatena_entries = self._collector.collect(keyword)
            except BlueskyAuthError as e:
                _show_error(e)
                try:
                    news, sns, hatena, hatena_entries = self._collector.collect(keyword)
                except TrendInsightError:
                    return None
                except Exception as exc:
                    log.error("データ収集失敗", error=str(exc), exc_info=True)
                    _show_error(DataCollectionError(str(exc)))
                    return None
            except DataCollectionError as e:
                _show_error(e)
                return None
            except Exception as e:
                log.error("予期しないエラー", error=str(e), exc_info=True)
                _show_error(DataCollectionError(str(e)))
                return None

        elapsed = time.perf_counter() - start
        log.info(
            "データ収集完了",
            news_count=len(news),
            bsky_count=len(sns),
            hatena_count=len(hatena),
            elapsed_sec=round(elapsed, 2),
        )

        if not news and not sns:
            st.warning(
                "📭 記事・投稿が見つかりませんでした。\n\n"
                "💡 キーワードを変えるか、より一般的な表現でお試しください。"
            )
            return None

        if not news:
            st.warning(
                "ニュース記事が見つかりませんでした。はてブ・SNSのデータで分析を続行します。"
            )
        if not sns and hasattr(self._collector, "bsky_configured"):
            if self._collector.bsky_configured:
                st.warning(
                    "BlueSky投稿が見つかりませんでした。メディア・はてブのデータで分析を続行します。"
                )
            else:
                st.info("💡 サイドバーでBlueSky認証を設定すると、SNSの声も分析できます。")

        return news, sns, hatena, hatena_entries

    def _analyze(
        self,
        keyword: str,
        news: list,
        sns: list,
        hatena: list,
        hatena_entries: list,
    ) -> AnalysisResult | None:
        """感情分析パイプライン実行フェーズ."""
        log = logger.bind(phase="analyze", keyword=keyword)
        start = time.perf_counter()
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
            log.error("分析パイプライン失敗", error=str(e), exc_info=True)
            _show_error(AnalysisPipelineError(str(e)))
            return None

        progress_bar.empty()
        elapsed = time.perf_counter() - start
        log.info(
            "感情分析完了",
            total_articles=len(result.news.results)
            + len(result.bsky.results)
            + len(result.hatena.results),
            analysis_types=[at["label"] for at in result.analysis_types],
            elapsed_sec=round(elapsed, 2),
        )
        return result

    def _start_ai_report_async(self, result: AnalysisResult) -> Future:
        """AI総評レポート生成をバックグラウンドスレッドで開始する."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._report_generator.generate, result)
        executor.shutdown(wait=False)
        return future

    def _await_ai_report(self, result: AnalysisResult, future: Future) -> None:
        """バックグラウンドのAI総評生成結果を取得する."""
        log = logger.bind(phase="report", keyword=result.keyword)
        start = time.perf_counter()

        with st.spinner("🧠 AIが総評レポートを生成中..."):
            try:
                result.ai_report = future.result()
            except ReportGenerationError as e:
                _show_error(e)
            except Exception as e:
                log.error("AI総評生成失敗", error=str(e), exc_info=True)
                _show_error(ReportGenerationError(str(e)))

        elapsed = time.perf_counter() - start
        log.info(
            "AI総評生成完了",
            report_length=len(result.ai_report) if result.ai_report else 0,
            elapsed_sec=round(elapsed, 2),
        )

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
        log = logger.bind(phase="save", keyword=result.keyword)
        try:
            self._history.save(result)
            st.info("💾 分析結果を履歴に保存しました。")
            log.info("履歴保存完了")
        except HistorySaveError as e:
            _show_error(e)
        except Exception as e:
            log.error("履歴保存失敗", error=str(e), exc_info=True)
            _show_error(HistorySaveError(str(e)))
