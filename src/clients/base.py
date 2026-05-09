"""クライアント共通インターフェース."""

import structlog
from abc import ABC, abstractmethod

from src.models.article import Article

logger = structlog.get_logger(__name__)


class ClientError(Exception):
    """クライアント処理中のエラー."""


class BaseClient(ABC):
    """全データソースクライアントの基底クラス.

    各クライアントは fetch(keyword) -> list[Article] を実装する。
    例外処理・リトライの共通ロジックをここに集約する。
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """データソース識別子を返す."""

    @abstractmethod
    def fetch(self, keyword: str) -> list[Article]:
        """キーワードに関連する記事/投稿を取得する.

        Args:
            keyword: 検索キーワード.

        Returns:
            正規化された Article のリスト.

        Raises:
            ClientError: 取得処理に失敗した場合.
        """

    def safe_fetch(self, keyword: str) -> list[Article]:
        """例外を握りつぶしてログに記録し、空リストを返す.

        UI層で個別にtry/exceptを書かなくて済むようにする。

        Args:
            keyword: 検索キーワード.

        Returns:
            取得結果。失敗時は空リスト.
        """
        try:
            return self.fetch(keyword)
        except Exception as e:
            logger.warning(
                "fetch失敗",
                phase="collect",
                source=self.source_name,
                keyword=keyword,
                error=str(e),
            )
            return []
