"""インサイト生成サービス - ルールベースのギャップ検出."""

from src.analyzer import compute_net_score


def _divergence_label(gap: float) -> str:
    """乖離幅から段階ラベルを返す."""
    if gap > 0.5:
        return "構造的乖離"
    elif gap > 0.3:
        return "明確な意見差"
    elif gap > 0.2:
        return "やや差あり"
    return "大差なし"


def compute_divergences(
    scores: dict[str, float],
) -> list[tuple[str, str, float, str]]:
    """全ソースペアの乖離を計算し、乖離幅降順で返す.

    Args:
        scores: ソース名→ネットスコアの辞書.

    Returns:
        (ソースA, ソースB, 乖離幅, ラベル) のリスト（降順）.
    """
    keys = list(scores.keys())
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            gap = abs(scores[keys[i]] - scores[keys[j]])
            pairs.append((keys[i], keys[j], round(gap, 2), _divergence_label(gap)))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


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
    source_names = {"news": "メディア", "bsky": "BlueSky", "hatena": "はてブ"}

    # ネットスコア算出
    net_scores: dict[str, float] = {"news": compute_net_score(news_stats)}
    all_stats = {"news": news_stats}
    if bsky_stats:
        net_scores["bsky"] = compute_net_score(bsky_stats)
        all_stats["bsky"] = bsky_stats
    if hatena_stats:
        net_scores["hatena"] = compute_net_score(hatena_stats)
        all_stats["hatena"] = hatena_stats

    # スコアサマリー
    score_parts = []
    for key, score in net_scores.items():
        neutral_pct = all_stats[key]["neutral"]
        score_parts.append(f"{source_names[key]}: スコア {score:+.2f}（中立 {neutral_pct:.0%}）")
    summary = " / ".join(score_parts)

    # 乖離分析
    if len(net_scores) < 2:
        return f"📊 {summary}"

    divergences = compute_divergences(net_scores)
    max_pair = divergences[0]
    _, _, _, _ = max_pair

    div_lines = []
    for src_a_i, src_b_i, gap_i, label_i in divergences:
        marker = "🔥" if gap_i > 0.5 else "⚡" if gap_i > 0.3 else "•"
        div_lines.append(
            f"  {marker} {source_names[src_a_i]} vs {source_names[src_b_i]}: "
            f"{gap_i:.2f}（{label_i}）"
        )

    result = f"📊 {summary}\n\n**乖離分析:**\n" + "\n".join(div_lines)
    return result
