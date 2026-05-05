"""感情分析モジュール - BERTモデル+辞書補正による感情スコアリング."""

import streamlit as st
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


@st.cache_resource
def load_sentiment_model():
    """感情分析モデルを起動時に一度だけロードする.

    Returns:
        transformers pipelineオブジェクト.
    """
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,
        truncation=True,
        max_length=512,
    )


@st.cache_resource
def load_tokenizer() -> Tokenizer:
    """janomeトークナイザーを起動時に一度だけロードする.

    Returns:
        Tokenizerインスタンス.
    """
    return Tokenizer()


def analyze_sentiment(title: str) -> dict[str, float | str]:
    """テキストの感情分析をBERTモデル+辞書補正で実施する.

    Args:
        title: 分析対象のテキスト.

    Returns:
        positive/negativeスコアとラベルを含む辞書.
    """
    classifier = load_sentiment_model()
    result = classifier(title)[0]
    label = result["label"].upper()
    score = result["score"]

    # モデルが3クラス（POSITIVE/NEGATIVE/NEUTRAL）を出力
    if label == "POSITIVE":
        pos_score = score
        neg_score = 1.0 - score
    elif label == "NEGATIVE":
        neg_score = score
        pos_score = 1.0 - score
    else:  # NEUTRAL
        pos_score = 0.5
        neg_score = 0.5

    # ネガティブ辞書による補正（強ネガティブ語が存在する場合のみ）
    boost = sum(1 for w in NEGATIVE_BOOST_WORDS if w in title)
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


def compute_sentiment_stats(results: list[dict]) -> dict[str, float]:
    """分析結果リストからラベル別の比率を算出する.

    Args:
        results: analyze_sentimentの結果を含む辞書のリスト.

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
