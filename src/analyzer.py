"""感情分析モジュール - BERTモデル+辞書補正による感情スコアリング."""

from janome.tokenizer import Tokenizer
from transformers import pipeline

MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"

# ネガティブ補正用辞書（強いネガティブ語）
NEGATIVE_BOOST_WORDS: set[str] = {
    "戦争", "悲惨", "孤児", "死亡", "倒産", "殺害", "虐待", "災害", "被害",
    "犠牲", "破壊", "崩壊", "暴力", "貧困", "飢餓", "難民", "紛争", "侵攻",
    "爆撃", "テロ", "事故", "汚染", "感染", "死者", "遺体", "自殺", "破綻",
    "詐欺", "逮捕", "懲役", "不正", "隠蔽", "搾取", "差別", "迫害",
}
NEGATIVE_BOOST_WEIGHT: float = 0.3


class SentimentAnalyzer:
    """BERT感情分析 + 辞書補正を行うクラス.

    インスタンス生成時にモデルをロードし、以降は analyze() で推論する。
    """

    def __init__(self) -> None:
        self._classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1,
            truncation=True,
            max_length=512,
        )

    def analyze(self, text: str) -> dict[str, float | str]:
        """テキストの感情分析をBERTモデル+辞書補正で実施する.

        Args:
            text: 分析対象のテキスト.

        Returns:
            positive/negativeスコアとラベルを含む辞書.
        """
        result = self._classifier(text)[0]
        label = result["label"].upper()
        score = result["score"]

        if label == "POSITIVE":
            pos_score = score
            neg_score = 1.0 - score
        elif label == "NEGATIVE":
            neg_score = score
            pos_score = 1.0 - score
        else:  # NEUTRAL
            pos_score = 0.5
            neg_score = 0.5

        # ネガティブ辞書による補正
        boost = sum(1 for w in NEGATIVE_BOOST_WORDS if w in text)
        if boost > 0:
            adjustment = min(boost * NEGATIVE_BOOST_WEIGHT, 0.5)
            neg_score = min(neg_score + adjustment, 1.0)
            pos_score = max(pos_score - adjustment, 0.0)

        # 最終ラベル判定（3段階）
        if abs(pos_score - neg_score) < 0.1:
            final_label = "neutral"
        elif pos_score > neg_score:
            final_label = "positive"
        else:
            final_label = "negative"

        return {"positive": pos_score, "negative": neg_score, "label": final_label}


def create_tokenizer() -> Tokenizer:
    """Janomeトークナイザーを生成する.

    Returns:
        Tokenizerインスタンス.
    """
    return Tokenizer()


def compute_sentiment_stats(results: list[dict]) -> dict[str, float]:
    """分析結果リストからラベル別の比率を算出する.

    Args:
        results: analyze()の結果を含む辞書のリスト.

    Returns:
        positive/neutral/negativeの比率を含む辞書.
    """
    total = len(results)
    if total == 0:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    pos_count = sum(1 for r in results if r["label"] == "positive")
    neg_count = sum(1 for r in results if r["label"] == "negative")
    neu_count = total - pos_count - neg_count
    return {
        "positive": pos_count / total,
        "neutral": neu_count / total,
        "negative": neg_count / total,
    }
