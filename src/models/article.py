"""記事・投稿の統一データモデル."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Article(BaseModel):
    """全ソース共通の記事/投稿データモデル.

    Googleニュース・BlueSky・はてなブックマークのいずれから取得したデータも
    このモデルに正規化して扱う。
    """

    title: str
    url: str
    source: str = Field(description="データソース識別子 (news, bluesky, hatena)")
    author: str | None = None
    published_at: datetime | None = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
