"""分析結果の型付きデータモデル."""


from typing import Any
from pydantic import BaseModel, Field


class AnalyzedArticle(BaseModel):
    """感情分析済みの記事/投稿."""

    title: str
    url: str
    source: str
    author: str | None = None
    positive: float
    negative: float
    label: str  # "positive" | "neutral" | "negative"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceStats(BaseModel):
    """ソース別の感情統計."""

    positive: float
    neutral: float
    negative: float


class SourceAnalysis(BaseModel):
    """1ソースの分析結果一式."""

    results: list[AnalyzedArticle] = Field(default_factory=list)
    stats: SourceStats | None = None
    keywords: list[tuple[str, int]] = Field(default_factory=list)
    samples: dict[str, list[str]] = Field(default_factory=dict)
    net_score: float = 0.0


class AnalysisResult(BaseModel):
    """分析パイプライン全体の出力."""

    keyword: str
    news: SourceAnalysis
    bsky: SourceAnalysis
    hatena: SourceAnalysis
    hatena_entry_data: list[dict[str, Any]] = Field(default_factory=list)
    topic_sentiments: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    analysis_types: list[dict[str, str]] = Field(default_factory=list)
    divergences: list[tuple[str, str, float, str]] = Field(default_factory=list)
    wordcloud_images: dict[str, str] = Field(default_factory=dict)
    ai_report: str = ""
