"""分析タイプ判定サービス - ルールベースのパターン分類."""


def detect_analysis_types(
    net_scores: dict[str, float],
    max_divergence: float,
    topic_sentiments: list[dict],
    neutral_ratios: dict[str, float],
) -> list[dict]:
    """分析結果から該当する分析タイプを判定する.

    Args:
        net_scores: ソース名→ネットスコア（pos - neg）の辞書.
        max_divergence: 全ソースペアの最大乖離値.
        topic_sentiments: compute_topic_sentimentsの出力（全ソース統合）.
        neutral_ratios: ソース名→neutral比率の辞書.

    Returns:
        該当する分析タイプのリスト（優先度順）.
        各要素は {"type": str, "emoji": str, "label": str, "reason": str}.
    """
    types: list[dict] = []

    # 1. 構造的乖離: ソース間で評価方向が逆
    if max_divergence > 0.5:
        scores = list(net_scores.values())
        has_pos = any(s > 0.1 for s in scores)
        has_neg = any(s < -0.1 for s in scores)
        if has_pos and has_neg:
            types.append({
                "type": "structural_divergence",
                "emoji": "🔥",
                "label": "構造的乖離",
                "reason": f"最大乖離 {max_divergence:.2f}、ソース間で評価方向が逆転",
            })

    # 2. トピック集中型（相対評価ベース）
    if len(topic_sentiments) >= 3:
        scores_list = [t["net_score"] for t in topic_sentiments]
        mean = sum(scores_list) / len(scores_list)
        # 標準偏差の簡易計算
        variance = sum((s - mean) ** 2 for s in scores_list) / len(scores_list)
        std = variance ** 0.5

        if std > 0.1:  # 分散がある場合のみ判定
            # 平均から1σ以上離れたトピックを「集中」とみなす
            neg_outliers = [
                t for t in topic_sentiments
                if t["net_score"] < mean - std and t["count"] >= 3
            ]
            pos_outliers = [
                t for t in topic_sentiments
                if t["net_score"] > mean + std and t["count"] >= 3
            ]

            if neg_outliers:
                top_neg = neg_outliers[0]["topic"]
                types.append({
                    "type": "topic_concentrated_neg",
                    "emoji": "⚡",
                    "label": "トピック集中型ネガティブ",
                    "reason": f"「{top_neg}」等に批判が集中（平均{mean:+.2f}から乖離）",
                })
            if pos_outliers:
                top_pos = pos_outliers[-1]["topic"]
                types.append({
                    "type": "topic_concentrated_pos",
                    "emoji": "🌟",
                    "label": "トピック集中型ポジティブ",
                    "reason": f"「{top_pos}」等に高評価が集中（平均{mean:+.2f}から乖離）",
                })

    # 3. 中立支配: 全ソースでneutral比率が高い
    if neutral_ratios and all(r > 0.6 for r in neutral_ratios.values()):
        avg_neutral = sum(neutral_ratios.values()) / len(neutral_ratios)
        types.append({
            "type": "neutral_dominant",
            "emoji": "⚪",
            "label": "中立支配",
            "reason": f"全ソースの中立率が60%超（平均{avg_neutral:.0%}）",
        })

    # 4. 感情一致: 全ソースで方向が同じ＆乖離小
    if len(net_scores) >= 2 and max_divergence < 0.2:
        scores = list(net_scores.values())
        all_pos = all(s > 0.05 for s in scores)
        all_neg = all(s < -0.05 for s in scores)
        if all_pos or all_neg:
            direction = "ポジティブ" if all_pos else "ネガティブ"
            types.append({
                "type": "consensus",
                "emoji": "🤝",
                "label": "感情一致",
                "reason": f"全ソースが{direction}方向で一致（乖離{max_divergence:.2f}）",
            })

    # デフォルト: どれにも該当しない
    if not types:
        types.append({
            "type": "mixed",
            "emoji": "🔀",
            "label": "混在型",
            "reason": "明確な単一パターンに分類されない複合的な状態",
        })

    return types
