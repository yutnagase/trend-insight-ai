"""インサイト生成サービス - ルールベースのギャップ検出."""


def generate_insight(
    news_stats: dict[str, float],
    bsky_stats: dict[str, float] | None = None,
    hatena_stats: dict[str, float] | None = None,
) -> str:
    """各ソースの感情スコアを比較しインサイトを生成する.

    Args:
        news_stats: メディア側の感情比率.
        bsky_stats: BlueSky側の感情比率.
        hatena_stats: はてなブックマーク側の感情比率.

    Returns:
        比較インサイトのテキスト.
    """
    def dominant(stats: dict[str, float]) -> str:
        return max(stats, key=stats.get)

    tone_map = {
        "positive": "ポジティブ（好意的）",
        "neutral": "中立的",
        "negative": "ネガティブ（批判的・懸念）",
    }

    parts: list[str] = [f"メディアは **{tone_map[dominant(news_stats)]}**"]
    if bsky_stats:
        parts.append(f"BlueSkyは **{tone_map[dominant(bsky_stats)]}**")
    if hatena_stats:
        parts.append(f"はてなブックマークは **{tone_map[dominant(hatena_stats)]}**")

    summary = "、".join(parts) + " な論調です。"

    all_dominants = [dominant(news_stats)]
    if bsky_stats:
        all_dominants.append(dominant(bsky_stats))
    if hatena_stats:
        all_dominants.append(dominant(hatena_stats))

    if len(set(all_dominants)) == 1:
        return f"📊 {summary} 各ソース間で論調が一致しています。"
    return f"📊 {summary} ソース間で温度差があります。"
