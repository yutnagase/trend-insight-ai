"""トピック別感情分析のテスト."""

from src.services.topic_sentiment import compute_topic_sentiments


class TestComputeTopicSentiments:
    """compute_topic_sentimentsのテスト."""

    def _make_results(self, items: list[tuple[str, str]]) -> list[dict]:
        """(title, label)のリストからresultsを生成."""
        return [{"title": title, "label": label} for title, label in items]

    def test_basic_aggregation(self):
        """キーワードごとにpositive/negativeが集約される."""
        results = self._make_results(
            [
                ("AI技術の進化", "positive"),
                ("AI規制の問題", "negative"),
                ("AI活用事例", "positive"),
            ]
        )
        keywords = [("AI", 3)]
        output = compute_topic_sentiments(results, keywords)

        assert len(output) == 1
        assert output[0]["topic"] == "AI"
        assert output[0]["pos"] == 2
        assert output[0]["neg"] == 1

    def test_net_score_calculation(self):
        """ネットスコアが (pos - neg) / (pos + neg) で計算される."""
        results = self._make_results(
            [
                ("技術革新", "positive"),
                ("技術問題", "negative"),
                ("技術課題", "negative"),
                ("技術発展", "negative"),
            ]
        )
        keywords = [("技術", 4)]
        output = compute_topic_sentiments(results, keywords)

        # (1 - 3) / (1 + 3) = -0.5
        assert abs(output[0]["net_score"] - (-0.5)) < 0.01

    def test_min_count_filter(self):
        """出現数がmin_count未満のトピックは除外."""
        results = self._make_results(
            [
                ("AI技術", "positive"),
            ]
        )
        keywords = [("AI", 1)]
        output = compute_topic_sentiments(results, keywords, min_count=3)

        assert output == []

    def test_neutral_only_excluded(self):
        """pos+neg < 2のトピックは除外（中立のみ）."""
        results = self._make_results(
            [
                ("AI技術の動向", "neutral"),
                ("AI関連ニュース", "neutral"),
                ("AIの現状", "positive"),
            ]
        )
        keywords = [("AI", 3)]
        output = compute_topic_sentiments(results, keywords)

        # pos=1, neg=0 → total=1 < 2 → 除外
        assert output == []

    def test_multiple_keywords(self):
        """複数キーワードが独立して集約される."""
        results = self._make_results(
            [
                ("AI技術の進化", "positive"),
                ("AI規制の問題", "negative"),
                ("規制緩和の動き", "positive"),
                ("規制強化の議論", "negative"),
            ]
        )
        keywords = [("AI", 2), ("規制", 3)]
        output = compute_topic_sentiments(results, keywords)

        topics = {t["topic"] for t in output}
        assert "AI" in topics
        assert "規制" in topics

    def test_sorted_by_net_score_ascending(self):
        """出力はネットスコア昇順でソートされる."""
        results = self._make_results(
            [
                ("良いAI", "positive"),
                ("良いAI", "positive"),
                ("悪い規制", "negative"),
                ("悪い規制", "negative"),
                ("良いAI", "positive"),
                ("悪い規制", "negative"),
            ]
        )
        keywords = [("AI", 3), ("規制", 3)]
        output = compute_topic_sentiments(results, keywords)

        assert len(output) >= 2
        scores = [t["net_score"] for t in output]
        assert scores == sorted(scores)

    def test_empty_results(self):
        """空入力で空リストを返す."""
        assert compute_topic_sentiments([], [("AI", 5)]) == []

    def test_empty_keywords(self):
        """キーワードが空なら空リストを返す."""
        results = self._make_results([("テスト", "positive")])
        assert compute_topic_sentiments(results, []) == []

    def test_case_insensitive_matching(self):
        """マッチングは大文字小文字を区別しない."""
        results = self._make_results(
            [
                ("Python活用", "positive"),
                ("PYTHON入門", "positive"),
                ("python問題", "negative"),
            ]
        )
        keywords = [("Python", 3)]
        output = compute_topic_sentiments(results, keywords)

        assert len(output) == 1
        assert output[0]["count"] == 3

    def test_keyword_limit_20(self):
        """上位20キーワードのみ処理される."""
        keywords = [(f"word{i}", 100 - i) for i in range(25)]
        results = self._make_results([(f"word{i}の話題", "positive") for i in range(25)] * 2)
        output = compute_topic_sentiments(results, keywords)

        # word20〜word24は処理されない
        topics = {t["topic"] for t in output}
        assert "word0" in topics
        assert "word24" not in topics
