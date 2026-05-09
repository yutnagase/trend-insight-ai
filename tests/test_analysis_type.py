"""分析タイプ判定のテスト."""

from src.services.analysis_type import detect_analysis_types


class TestStructuralDivergence:
    """構造的乖離の判定テスト."""

    def test_opposite_directions_with_large_gap(self):
        """ソース間で評価方向が逆かつ乖離>0.5で構造的乖離."""
        result = detect_analysis_types(
            net_scores={"news": 0.4, "bsky": -0.3},
            max_divergence=0.7,
            topic_sentiments=[],
            neutral_ratios={"news": 0.3, "bsky": 0.3},
            sample_counts={"news": 10, "bsky": 10},
        )
        assert any(t["type"] == "structural_divergence" for t in result)

    def test_same_direction_no_structural_divergence(self):
        """同方向なら乖離が大きくても構造的乖離にならない."""
        result = detect_analysis_types(
            net_scores={"news": 0.6, "bsky": 0.05},
            max_divergence=0.55,
            topic_sentiments=[],
            neutral_ratios={"news": 0.2, "bsky": 0.4},
            sample_counts={"news": 10, "bsky": 10},
        )
        assert not any(t["type"] == "structural_divergence" for t in result)

    def test_small_sample_excluded(self):
        """サンプル数不足のソースは乖離判定から除外."""
        result = detect_analysis_types(
            net_scores={"news": 0.5, "bsky": -0.5},
            max_divergence=1.0,
            topic_sentiments=[],
            neutral_ratios={"news": 0.3, "bsky": 0.3},
            sample_counts={"news": 10, "bsky": 3},  # bskyが閾値未満
        )
        assert not any(t["type"] == "structural_divergence" for t in result)


class TestTopicConcentrated:
    """トピック集中型の判定テスト."""

    def _make_topics(self, scores: list[float]) -> list[dict]:
        return [{"topic": f"word{i}", "net_score": s, "count": 10} for i, s in enumerate(scores)]

    def test_negative_outlier_detected(self):
        """平均-1σ未満のトピックがあればネガティブ集中型."""
        # mean=0.0, std≈0.45 → -0.8は平均-1σ未満
        topics = self._make_topics([0.4, 0.4, 0.0, -0.8])
        result = detect_analysis_types(
            net_scores={"news": 0.0},
            max_divergence=0.0,
            topic_sentiments=topics,
            neutral_ratios={"news": 0.5},
        )
        assert any(t["type"] == "topic_concentrated_neg" for t in result)

    def test_positive_outlier_detected(self):
        """平均+1σ超のトピックがあればポジティブ集中型."""
        topics = self._make_topics([-0.4, -0.4, 0.0, 0.8])
        result = detect_analysis_types(
            net_scores={"news": 0.0},
            max_divergence=0.0,
            topic_sentiments=topics,
            neutral_ratios={"news": 0.5},
        )
        assert any(t["type"] == "topic_concentrated_pos" for t in result)

    def test_no_outlier_when_uniform(self):
        """スコアが均一ならトピック集中型にならない."""
        topics = self._make_topics([0.1, 0.1, 0.1, 0.1])
        result = detect_analysis_types(
            net_scores={"news": 0.0},
            max_divergence=0.0,
            topic_sentiments=topics,
            neutral_ratios={"news": 0.5},
        )
        assert not any(t["type"].startswith("topic_concentrated") for t in result)

    def test_low_count_topic_excluded(self):
        """出現数が少ないトピックは外れ値判定から除外."""
        topics = [
            {"topic": "w0", "net_score": 0.1, "count": 10},
            {"topic": "w1", "net_score": 0.1, "count": 10},
            {"topic": "w2", "net_score": 0.1, "count": 10},
            {"topic": "w3", "net_score": -0.9, "count": 3},  # count < 5
        ]
        result = detect_analysis_types(
            net_scores={"news": 0.0},
            max_divergence=0.0,
            topic_sentiments=topics,
            neutral_ratios={"news": 0.5},
        )
        assert not any(t["type"] == "topic_concentrated_neg" for t in result)


class TestNeutralDominant:
    """中立支配の判定テスト."""

    def test_all_sources_high_neutral(self):
        """全ソースのneutral>60%で中立支配."""
        result = detect_analysis_types(
            net_scores={"news": 0.0, "bsky": 0.05},
            max_divergence=0.05,
            topic_sentiments=[],
            neutral_ratios={"news": 0.7, "bsky": 0.65},
            sample_counts={"news": 10, "bsky": 10},
        )
        assert any(t["type"] == "neutral_dominant" for t in result)

    def test_one_source_low_neutral(self):
        """1ソースでもneutral<=60%なら中立支配にならない."""
        result = detect_analysis_types(
            net_scores={"news": 0.0, "bsky": 0.0},
            max_divergence=0.0,
            topic_sentiments=[],
            neutral_ratios={"news": 0.7, "bsky": 0.5},
            sample_counts={"news": 10, "bsky": 10},
        )
        assert not any(t["type"] == "neutral_dominant" for t in result)


class TestConsensus:
    """感情一致の判定テスト."""

    def test_all_positive_low_divergence(self):
        """全ソースがポジティブ方向で乖離<0.2なら感情一致."""
        result = detect_analysis_types(
            net_scores={"news": 0.3, "bsky": 0.2, "hatena": 0.25},
            max_divergence=0.1,
            topic_sentiments=[],
            neutral_ratios={"news": 0.3, "bsky": 0.3, "hatena": 0.3},
            sample_counts={"news": 10, "bsky": 10, "hatena": 10},
        )
        assert any(t["type"] == "consensus" for t in result)

    def test_all_negative_consensus(self):
        """全ソースがネガティブ方向でも感情一致."""
        result = detect_analysis_types(
            net_scores={"news": -0.3, "bsky": -0.2},
            max_divergence=0.1,
            topic_sentiments=[],
            neutral_ratios={"news": 0.3, "bsky": 0.3},
            sample_counts={"news": 10, "bsky": 10},
        )
        assert any(t["type"] == "consensus" for t in result)

    def test_mixed_directions_no_consensus(self):
        """方向が混在していれば感情一致にならない."""
        result = detect_analysis_types(
            net_scores={"news": 0.1, "bsky": -0.06},
            max_divergence=0.16,
            topic_sentiments=[],
            neutral_ratios={"news": 0.3, "bsky": 0.3},
            sample_counts={"news": 10, "bsky": 10},
        )
        assert not any(t["type"] == "consensus" for t in result)


class TestMixed:
    """混在型（デフォルト）の判定テスト."""

    def test_no_pattern_matches(self):
        """どのパターンにも該当しなければ混在型."""
        result = detect_analysis_types(
            net_scores={"news": 0.1, "bsky": -0.05},
            max_divergence=0.15,
            topic_sentiments=[],
            neutral_ratios={"news": 0.4, "bsky": 0.5},
            sample_counts={"news": 10, "bsky": 10},
        )
        assert any(t["type"] == "mixed" for t in result)

    def test_returns_at_least_one_type(self):
        """常に1つ以上のタイプが返る."""
        result = detect_analysis_types(
            net_scores={},
            max_divergence=0.0,
            topic_sentiments=[],
            neutral_ratios={},
        )
        assert len(result) >= 1
