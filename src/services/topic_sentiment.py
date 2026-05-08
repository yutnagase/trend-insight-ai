"""トピック別感情分析サービス - キーワード単位での感情集約."""

from collections import defaultdict


def compute_topic_sentiments(
    results: list[dict], keywords: list[tuple[str, int]], min_count: int = 2
) -> list[dict]:
    """キーワードごとに感情スコアを集約し、トピック×感情マップを返す.

    Args:
        results: 感情分析済みの記事/投稿dictリスト（title, label を含む）.
        keywords: extract_keywordsの出力（単語, 出現回数）のリスト.
        min_count: 最低出現数（これ未満のトピックは除外）.

    Returns:
        トピック別スコアのリスト（ネットスコア昇順）.
    """
    topic_scores = defaultdict(lambda: {"pos": 0, "neg": 0, "count": 0})

    # 上位キーワードのみ対象（最大20語）
    target_words = [w for w, _ in keywords[:20]]

    for r in results:
        text = r.get("title", "") + " " + (r.get("content") or "")
        label = r.get("label", "neutral")

        for word in target_words:
            if word in text:
                topic_scores[word]["count"] += 1
                if label == "positive":
                    topic_scores[word]["pos"] += 1
                elif label == "negative":
                    topic_scores[word]["neg"] += 1

    output = []
    for word, scores in topic_scores.items():
        if scores["count"] < min_count:
            continue
        total = scores["pos"] + scores["neg"]
        net = (scores["pos"] - scores["neg"]) / total if total > 0 else 0.0
        output.append({"topic": word, "net_score": net, **scores})

    return sorted(output, key=lambda x: x["net_score"])
