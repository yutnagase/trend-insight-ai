"""履歴管理サービス - 分析結果の永続化."""

import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path("data/analysis_history.json")


def save_history(
    keyword: str,
    news_results: list[dict],
    sns_results: list[dict],
    hatena_results: list[dict],
    wordcloud_images: dict[str, str],
    ai_report: str,
) -> None:
    """分析結果を履歴JSONに追記保存する.

    Args:
        keyword: 検索キーワード.
        news_results: メディア分析結果.
        sns_results: SNS分析結果.
        hatena_results: はてブ分析結果.
        wordcloud_images: ソース名→画像パスのマッピング.
        ai_report: AI総評レポートテキスト.
    """
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    history.append({
        "timestamp": datetime.now().isoformat(),
        "keyword": keyword,
        "news_results": news_results,
        "sns_results": sns_results,
        "hatena_results": hatena_results,
        "wordcloud_images": wordcloud_images,
        "ai_report": ai_report,
    })
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_history() -> list[dict]:
    """履歴JSONを読み込む.

    Returns:
        履歴データのリスト（新しい順）.
    """
    if not HISTORY_PATH.exists():
        return []
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return list(reversed(history))
