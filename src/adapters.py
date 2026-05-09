"""Protocol具象実装 - 既存モジュールをDIインターフェースに適合させるアダプター."""

import logging

from src.clients import BlueskyClient, GoogleNewsClient, HatenaClient
from src.models.analysis_result import AnalysisResult
from src.models.article import Article
from src.reporter import generate_report
from src.services.history import load_history, save_history

logger = logging.getLogger(__name__)


class MultiSourceCollector:
    """複数ソースからデータを収集する具象実装."""

    def __init__(self, bsky_handle: str = "", bsky_password: str = "") -> None:
        self._bsky_handle = bsky_handle
        self._bsky_password = bsky_password

    @property
    def bsky_configured(self) -> bool:
        return bool(self._bsky_handle and self._bsky_password)

    def collect(
        self, keyword: str
    ) -> tuple[list[Article], list[Article], list[Article], list[dict]]:
        news_client = GoogleNewsClient()
        bsky_client = BlueskyClient(
            handle=self._bsky_handle, app_password=self._bsky_password
        )
        hatena_client = HatenaClient()

        news_articles = news_client.safe_fetch(keyword)
        logger.info("ニュース記事: %d件取得", len(news_articles))

        sns_articles: list[Article] = []
        if bsky_client.is_configured:
            sns_articles = bsky_client.safe_fetch(keyword)
            logger.info("BlueSky投稿: %d件取得", len(sns_articles))

        hatena_articles, hatena_entry_data = hatena_client.fetch_with_entries(keyword)
        logger.info("はてブコメント: %d件取得", len(hatena_articles))

        return news_articles, sns_articles, hatena_articles, hatena_entry_data


class LLMReportGenerator:
    """LLMベースのレポート生成具象実装."""

    def generate(self, result: AnalysisResult) -> str:
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


class JsonHistoryRepository:
    """JSON履歴ファイルによる永続化具象実装."""

    def save(self, result: AnalysisResult) -> None:
        save_history(
            result.keyword,
            [r.model_dump() for r in result.news.results],
            [r.model_dump() for r in result.bsky.results],
            [r.model_dump() for r in result.hatena.results],
            result.wordcloud_images,
            result.ai_report,
        )

    def load(self) -> list[dict]:
        return load_history()
