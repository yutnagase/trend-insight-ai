"""感情分析モジュール - 複数BERTモデルのアンサンブルによる感情スコアリング."""

import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = structlog.get_logger(__name__)

# アンサンブル対象モデル定義
ENSEMBLE_MODELS: list[dict[str, str | float]] = [
    {
        "name": "koheiduck/bert-japanese-finetuned-sentiment",
        "weight": 0.334,
    },
    {
        "name": "christian-phu/bert-finetuned-japanese-sentiment",
        "weight": 0.333,
    },
    {
        "name": "llm-book/bert-base-japanese-v3-marc-ja",
        "weight": 0.333,
    },
]

BATCH_SIZE = 16


class _ModelUnit:
    """単一モデルのロード・推論を担当する内部クラス."""

    def __init__(self, model_name: str, weight: float) -> None:
        self.model_name = model_name
        self.weight = weight
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # ラベルマッピングを取得
        self.id2label: dict[int, str] = self.model.config.id2label
        self._label_map = self._build_label_map()

    def _build_label_map(self) -> dict[str, int]:
        """モデルのラベル体系を pos/neu/neg のインデックスに変換するマップを構築."""
        label_map: dict[str, int] = {}
        for idx, label in self.id2label.items():
            normalized = label.upper().replace("LABEL_", "")
            if normalized in ("POSITIVE", "2", "POS"):
                label_map["positive"] = int(idx)
            elif normalized in ("NEGATIVE", "0", "NEG"):
                label_map["negative"] = int(idx)
            elif normalized in ("NEUTRAL", "1", "NEU"):
                label_map["neutral"] = int(idx)
        # 2クラスモデルの場合はneutralなし
        if "neutral" not in label_map:
            label_map["neutral"] = -1
        return label_map

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """バッチ推論で各テキストの正規化スコアを返す."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)

        results = []
        for prob in probs:
            pos_idx = self._label_map.get("positive")
            neg_idx = self._label_map.get("negative")
            neu_idx = self._label_map.get("neutral")

            pos_score = prob[pos_idx].item() if pos_idx is not None and pos_idx >= 0 else 0.0
            neg_score = prob[neg_idx].item() if neg_idx is not None and neg_idx >= 0 else 0.0
            neu_score = prob[neu_idx].item() if neu_idx is not None and neu_idx >= 0 else 0.0

            # 2クラスモデルの場合: neutral = 1 - pos - neg（残余）
            if self._label_map.get("neutral", -1) == -1:
                neu_score = max(1.0 - pos_score - neg_score, 0.0)

            results.append(
                {
                    "positive": pos_score,
                    "negative": neg_score,
                    "neutral": neu_score,
                }
            )
        return results


class SentimentAnalyzer:
    """複数BERTモデルのアンサンブルによる感情分析クラス.

    各モデルの全クラス確率を加重平均（ソフト投票）し、最終ラベルを判定する。
    """

    def __init__(self, models_config: list[dict] | None = None) -> None:
        config = models_config or ENSEMBLE_MODELS
        logger.info("アンサンブルモデルをロード中", model_count=len(config))
        self._units: list[_ModelUnit] = []
        for cfg in config:
            try:
                unit = _ModelUnit(str(cfg["name"]), float(cfg["weight"]))
                self._units.append(unit)
                logger.info("モデルロード成功", model=cfg["name"], weight=cfg["weight"])
            except Exception as e:
                logger.warning("モデルロード失敗", model=cfg["name"], error=str(e))
        if not self._units:
            raise RuntimeError("有効なモデルが1つもロードできませんでした")
        # ウェイトを正規化
        total_weight = sum(u.weight for u in self._units)
        for u in self._units:
            u.weight = u.weight / total_weight

    def analyze(self, text: str) -> dict[str, float | str]:
        """単一テキストの感情分析（既存インターフェース互換）."""
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str]) -> list[dict[str, float | str]]:
        """バッチでテキストの感情分析を実施する.

        Args:
            texts: 分析対象テキストのリスト.

        Returns:
            各テキストのpositive/negative/labelを含む辞書のリスト.
        """
        if not texts:
            return []

        # 各モデルでバッチ推論
        all_model_results: list[list[dict[str, float]]] = []
        for unit in self._units:
            model_results = []
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                model_results.extend(unit.predict_batch(batch))
            all_model_results.append(model_results)

        # 加重平均でアンサンブル
        final_results = []
        for idx in range(len(texts)):
            pos = sum(
                all_model_results[m][idx]["positive"] * self._units[m].weight
                for m in range(len(self._units))
            )
            neg = sum(
                all_model_results[m][idx]["negative"] * self._units[m].weight
                for m in range(len(self._units))
            )
            neu = sum(
                all_model_results[m][idx]["neutral"] * self._units[m].weight
                for m in range(len(self._units))
            )

            # 最終ラベル判定
            if abs(pos - neg) < 0.1 or neu > max(pos, neg):
                final_label = "neutral"
            elif pos > neg:
                final_label = "positive"
            else:
                final_label = "negative"

            final_results.append(
                {
                    "positive": pos,
                    "negative": neg,
                    "label": final_label,
                }
            )

        return final_results


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


def compute_net_score(stats: dict[str, float]) -> float:
    """感情比率からネットスコア（positive - negative）を算出する.

    Args:
        stats: compute_sentiment_statsの出力.

    Returns:
        -1.0〜1.0のネットスコア.
    """
    return stats["positive"] - stats["negative"]


def select_representative(
    results: list[dict],
    top_n: int = 1,
) -> dict[str, list[str]]:
    """分析結果からpositive/negativeの代表的な意見を抽出する.

    ラベルが該当する記事のみから抽出し、同一記事が両方に出ることを防ぐ。

    Args:
        results: analyze()の結果を含む辞書のリスト.
        top_n: 各極性から抽出する件数.

    Returns:
        positive/negativeそれぞれの代表テキストリスト.
    """
    if not results:
        return {"positive": [], "negative": []}
    pos_items = [r for r in results if r["label"] == "positive"]
    neg_items = [r for r in results if r["label"] == "negative"]
    pos_sorted = sorted(pos_items, key=lambda r: r["positive"], reverse=True)
    neg_sorted = sorted(neg_items, key=lambda r: r["negative"], reverse=True)
    return {
        "positive": [r["title"][:80] for r in pos_sorted[:top_n]],
        "negative": [r["title"][:80] for r in neg_sorted[:top_n]],
    }
