"""依存性注入用のProtocolインターフェース定義."""

from typing import Protocol

from src.models.analysis_result import AnalysisResult
from src.models.article import Article


class SentimentAnalyzerProtocol(Protocol):
    """感情分析器のインターフェース."""

    def analyze(self, text: str) -> dict[str, float | str]: ...

    def analyze_batch(self, texts: list[str]) -> list[dict[str, float | str]]: ...


class DataCollectorProtocol(Protocol):
    """データ収集のインターフェース."""

    def collect(
        self, keyword: str
    ) -> tuple[list[Article], list[Article], list[Article], list[dict]]:
        """全ソースからデータを収集する.

        Returns:
            (news_articles, sns_articles, hatena_articles, hatena_entry_data)
        """
        ...


class ReportGeneratorProtocol(Protocol):
    """AI総評レポート生成のインターフェース."""

    def generate(self, result: AnalysisResult) -> str: ...


class HistoryRepositoryProtocol(Protocol):
    """履歴永続化のインターフェース."""

    def save(self, result: AnalysisResult) -> None: ...

    def load(self) -> list[dict]: ...
