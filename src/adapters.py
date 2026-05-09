"""Protocol具象実装 - 既存モジュールをDIインターフェースに適合させるアダプター."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import structlog

from src.clients import BlueskyClient, GoogleNewsClient, HatenaClient
from src.exceptions import (
    BlueskyAuthError,
    DataCollectionError,
    HistoryLoadError,
    HistorySaveError,
    ReportGenerationError,
)
from src.models.analysis_result import AnalysisResult
from src.models.article import Article
from src.reporter import generate_report
from src.services.history import load_history, save_history

logger = structlog.get_logger(__name__)


class MultiSourceCollector:
    """複数ソースからデータを並列収集する具象実装."""

    def __init__(self, bsky_handle: str = "", bsky_password: str = "") -> None:
        self._bsky_handle = bsky_handle
        self._bsky_password = bsky_password

    @property
    def bsky_configured(self) -> bool:
        return bool(self._bsky_handle and self._bsky_password)

    def collect(
        self, keyword: str
    ) -> tuple[list[Article], list[Article], list[Article], list[dict[str, Any]]]:
        """3ソースを並列にフェッチし、全完了後に結果を返す."""
        news_client = GoogleNewsClient()
        bsky_client = BlueskyClient(handle=self._bsky_handle, app_password=self._bsky_password)
        hatena_client = HatenaClient()

        news_articles: list[Article] = []
        sns_articles: list[Article] = []
        hatena_articles: list[Article] = []
        hatena_entry_data: list[dict[str, Any]] = []

        def fetch_news() -> list[Article]:
            return news_client.safe_fetch(keyword)

        def fetch_bsky() -> list[Article]:
            if bsky_client.is_configured:
                return bsky_client.fetch(keyword)
            return []

        def fetch_hatena() -> tuple[list[Article], list[dict[str, Any]]]:
            return hatena_client.fetch_with_entries(keyword)

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_news = executor.submit(fetch_news)
            future_bsky = executor.submit(fetch_bsky)
            future_hatena = executor.submit(fetch_hatena)

            # ニュース
            try:
                news_articles = future_news.result()
            except Exception as e:
                logger.error(
                    "ニュース取得で予期しないエラー",
                    phase="collect",
                    source="news",
                    error=str(e),
                    exc_info=True,
                )
                raise DataCollectionError(str(e)) from e

            # BlueSky
            try:
                sns_articles = future_bsky.result()
            except Exception as e:
                if "auth" in str(e).lower() or "login" in str(e).lower():
                    logger.warning(
                        "BlueSky認証失敗", phase="collect", source="bluesky", error=str(e)
                    )
                    raise BlueskyAuthError(str(e)) from e
                logger.warning("BlueSky取得失敗", phase="collect", source="bluesky", error=str(e))
                sns_articles = []

            # はてブ
            try:
                hatena_articles, hatena_entry_data = future_hatena.result()
            except Exception as e:
                logger.error(
                    "はてブ取得で予期しないエラー",
                    phase="collect",
                    source="hatena",
                    error=str(e),
                    exc_info=True,
                )
                raise DataCollectionError(str(e)) from e

        logger.info(
            "データ収集完了",
            phase="collect",
            news_count=len(news_articles),
            bsky_count=len(sns_articles),
            hatena_count=len(hatena_articles),
        )
        return news_articles, sns_articles, hatena_articles, hatena_entry_data


class LLMReportGenerator:
    """LLMベースのレポート生成具象実装."""

    def generate(self, result: AnalysisResult) -> str:
        try:
            return generate_report(
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
            logger.error("レポート生成失敗", phase="report", error=str(e), exc_info=True)
            raise ReportGenerationError(str(e)) from e


class JsonHistoryRepository:
    """JSON履歴ファイルによる永続化具象実装."""

    def save(self, result: AnalysisResult) -> None:
        try:
            save_history(
                result.keyword,
                [r.model_dump() for r in result.news.results],
                [r.model_dump() for r in result.bsky.results],
                [r.model_dump() for r in result.hatena.results],
                result.wordcloud_images,
                result.ai_report,
            )
        except Exception as e:
            logger.error("履歴保存失敗", phase="save", error=str(e), exc_info=True)
            raise HistorySaveError(str(e)) from e

    def load(self) -> list[dict[str, Any]]:
        try:
            return load_history()
        except Exception as e:
            logger.error("履歴読み込み失敗", phase="save", error=str(e), exc_info=True)
            raise HistoryLoadError(str(e)) from e
