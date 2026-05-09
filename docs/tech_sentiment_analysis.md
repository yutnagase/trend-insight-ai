# 感情分析（Sentiment Analysis） — 複数BERTモデルのアンサンブル

## 感情分析とは

感情分析は、テキストが「ポジティブ」「ネガティブ」「中立」のどれに近いかを判定する自然言語処理の技術です。

本プロジェクトでは、Googleニュースの見出し・BlueSkyの投稿・はてブのコメントに対して感情分析を行い、各ソースの「空気感」を数値化しています。

## 使っている技術

- **transformers** — Hugging Face社が開発した自然言語処理ライブラリ。学習済みモデルを簡単に呼び出せる
- **PyTorch** — モデル推論基盤。バッチ処理・softmax確率計算に使用

## アンサンブル構成

単一モデルでは中立寄りに保守的な出力をしやすい傾向があるため、異なるデータで学習された複数モデルの加重平均（ソフト投票）で判定しています。

| モデル | 特性 | ウェイト |
|--------|------|----------|
| `koheiduck/bert-japanese-finetuned-sentiment` | 汎用・ニュース寄り。3クラス分類 | 0.334 |
| `christian-phu/bert-finetuned-japanese-sentiment` | レビュー特化。明確なポジ/ネガ検出力が高い | 0.333 |
| `llm-book/bert-base-japanese-v3-marc-ja` | MARC-jaデータセット。2クラスで「はっきり判定する」役割 | 0.333 |

### なぜアンサンブルか

- 単一モデル依存を避け、特定モデルの中立バイアスを相互補完
- 恣意性の排除（「人間がキーワードを決めた」ではなく「複数の学習済みモデルが合意した」）
- 各モデルが異なるデータで学習されているため、トレンド変化に強い

## BERTとは

BERT（Bidirectional Encoder Representations from Transformers）は、Googleが2018年に発表した言語モデルです。大量のテキストで事前学習されており、文脈を理解した上でテキストの意味を捉えられます。

「事前学習済みモデル」を使うので、自分で大量のデータを用意して学習させる必要がありません。Hugging Faceに公開されているモデルをダウンロードするだけで使えます。

## 本プロジェクトでの実装

### アーキテクチャ概要

```
テキスト入力
    ↓
┌─────────────────────────────────────────┐
│  Model 1 (koheiduck)    → [pos, neu, neg] × weight  │
│  Model 2 (christian-phu) → [pos, neu, neg] × weight  │
│  Model 3 (llm-book)     → [pos, neu, neg] × weight  │
└─────────────────────────────────────────┘
    ↓ 加重平均
最終スコア [pos, neu, neg]
    ↓
ラベル判定 (positive / neutral / negative)
```

### モデルのロードと推論

`transformers.pipeline` ではなく `AutoModelForSequenceClassification` + `AutoTokenizer` を直接使い、バッチ推論に対応しています。

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# バッチ推論
encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)  # 全クラスの確率を取得
```

`top_k=None` 相当の処理を自前で行い、全クラスの確率分布を直接取得しています。

### ラベル体系の正規化

モデルごとに出力ラベルが異なります（`POSITIVE` / `LABEL_2` / `POS` など）。`model.config.id2label` を参照して自動的に positive / neutral / negative のインデックスにマッピングしています。

```python
id2label = model.config.id2label
# 例: {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
# 例: {0: "negative", 1: "positive"}（2クラスモデル）
```

2クラスモデル（neutral出力なし）の場合は、`neutral = 1.0 - pos - neg` の残余として扱います。

### アンサンブル（加重平均）

各モデルの正規化済みスコアを加重平均します。

```python
pos = sum(model_results[m]["positive"] * weights[m] for m in range(num_models))
neg = sum(model_results[m]["negative"] * weights[m] for m in range(num_models))
neu = sum(model_results[m]["neutral"] * weights[m] for m in range(num_models))
```

### 最終ラベルの決定

```python
if abs(pos - neg) < 0.1 or neu > max(pos, neg):
    final_label = "neutral"
elif pos > neg:
    final_label = "positive"
else:
    final_label = "negative"
```

- ポジティブとネガティブの差が0.1未満 → 中立
- 中立確率が最大 → 中立
- それ以外は大きい方のラベルを採用

### バッチ推論

1件ずつの推論ではなく、`BATCH_SIZE = 16` でまとめて処理します。これにより、3モデル×多数記事の推論でもパディング・並列計算の恩恵を受けられます。

```python
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    results.extend(model.predict_batch(batch))
```

### フォールバック設計

モデルのロードに失敗した場合（ネットワーク障害、モデル削除など）、そのモデルをスキップしてウェイトを再正規化します。最低1モデルが動作すれば分析は継続されます。

```python
# ウェイト再正規化
total_weight = sum(u.weight for u in self._units)
for u in self._units:
    u.weight = u.weight / total_weight
```

## 旧実装（辞書補正方式）からの移行理由

以前は単一BERTモデル + ハードコードされたネガティブ辞書（34語）でスコアを補正していました。

**旧方式の問題点**:
- 辞書のメンテナンスコスト（トレンドに追従できない）
- 恣意性の批判リスク（「人間が選んだキーワードで結果を操作している」と見なされうる）
- `1.0 - score` という不正確な変換（3クラスの確率分布を無視）

**アンサンブル方式の利点**:
- 辞書メンテナンス不要
- 「複数モデルの合意」という客観的な判定基準
- 全クラス確率を直接使用し、情報の損失なし
- モデル追加・差し替えが設定変更のみで可能

## なぜローカル実行か

OpenAIのAPIなど外部サービスを使えばもっと高精度な感情分析もできますが、本プロジェクトでは以下の理由でローカル実行を選んでいます。

- 分析対象のテキストを外部に送信しない（プライバシー保護）
- API利用料がかからない
- ネットワーク接続がなくても動作する

BERT-base × 3モデルで約1.2〜2.0GBのメモリを使用します。

## 参考リンク

- [Hugging Face transformers](https://huggingface.co/docs/transformers/)
- [koheiduck/bert-japanese-finetuned-sentiment](https://huggingface.co/koheiduck/bert-japanese-finetuned-sentiment)
- [christian-phu/bert-finetuned-japanese-sentiment](https://huggingface.co/christian-phu/bert-finetuned-japanese-sentiment)
- [llm-book/bert-base-japanese-v3-marc-ja](https://huggingface.co/llm-book/bert-base-japanese-v3-marc-ja)
- [BERTの解説（原論文）](https://arxiv.org/abs/1810.04805)
