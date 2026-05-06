# 感情分析（Sentiment Analysis） — transformers + BERT日本語モデル

## 感情分析とは

感情分析は、テキストが「ポジティブ」「ネガティブ」「中立」のどれに近いかを判定する自然言語処理の技術です。

本プロジェクトでは、Googleニュースの見出し・BlueSkyの投稿・はてブのコメントに対して感情分析を行い、各ソースの「空気感」を数値化しています。

## 使っている技術

- **transformers**
  - Hugging Face社が開発した自然言語処理ライブラリ。学習済みモデルを簡単に呼び出せる
- **koheiduck/bert-japanese-finetuned-sentiment**
  - 日本語テキストの感情を3クラス（POSITIVE / NEUTRAL / NEGATIVE）に分類するBERTモデル

## BERTとは

BERT（Bidirectional Encoder Representations from Transformers）は、Googleが2018年に発表した言語モデルです。大量のテキストで事前学習されており、文脈を理解した上でテキストの意味を捉えられます。

「事前学習済みモデル」を使うので、自分で大量のデータを用意して学習させる必要がありません。Hugging Faceに公開されているモデルをダウンロードするだけで使えます。

## 基本的な使い方

```python
from transformers import pipeline

# モデルを読み込む（初回はダウンロードが走る）
classifier = pipeline(
    "sentiment-analysis",
    model="koheiduck/bert-japanese-finetuned-sentiment",
    tokenizer="koheiduck/bert-japanese-finetuned-sentiment",
)

# テキストを判定
result = classifier("今日はとても良い天気ですね")
print(result)
# [{'label': 'POSITIVE', 'score': 0.92}]

result = classifier("大規模な災害が発生しました")
print(result)
# [{'label': 'NEGATIVE', 'score': 0.87}]
```

`pipeline` に `"sentiment-analysis"` とモデル名を渡すだけで、感情分析器が手に入ります。あとはテキストを渡せばラベルとスコアが返ってきます。

## 本プロジェクトでの実装

### 3クラス分類の処理

モデルの出力は `POSITIVE` / `NEUTRAL` / `NEGATIVE` のいずれかです。これをポジティブ度・ネガティブ度の数値に変換しています。

```python
result = classifier(title)[0]
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
```

NEUTRALの場合は「どちらでもない」ので、ポジティブ・ネガティブ両方を0.5にしています。ニュースの事実報道はほとんどがここに落ちます。

### 辞書補正（ネガティブブースト）

BERTモデルだけでは、「戦争」「死亡」のような明らかにネガティブな語を含む文が中立判定されることがあります。そこで、あらかじめ用意したネガティブ語辞書でスコアを補正しています。

```python
NEGATIVE_BOOST_WORDS = {"戦争", "悲惨", "孤児", "死亡", "倒産", ...}

boost = sum(1 for w in NEGATIVE_BOOST_WORDS if w in title)
if boost > 0:
    adjustment = min(boost * 0.3, 0.5)
    neg_score = min(neg_score + adjustment, 1.0)
    pos_score = max(pos_score - adjustment, 0.0)
```

ネガティブ語が1つ見つかるごとに0.3ポイント補正し、最大0.5まで。これにより「悲惨な経験をした二人のきょうだい」のような文が誤ってポジティブ判定されるのを防いでいます。

### 最終ラベルの決定

```python
if abs(pos_score - neg_score) < 0.1:
    final_label = "neutral"
elif pos_score > neg_score:
    final_label = "positive"
else:
    final_label = "negative"
```

ポジティブとネガティブの差が0.1未満なら「中立」とみなします。微妙な差で無理にどちらかに振り分けないための閾値です。

## なぜローカル実行か

OpenAIのAPIなど外部サービスを使えばもっと高精度な感情分析もできますが、本プロジェクトでは以下の理由でローカル実行を選んでいます。

- 分析対象のテキストを外部に送信しない（プライバシー保護）
- API利用料がかからない
- ネットワーク接続がなくても動作する

モデルサイズは約500MBで、一般的なPCのメモリに収まります。

## 参考リンク

- [Hugging Face transformers](https://huggingface.co/docs/transformers/)
- [koheiduck/bert-japanese-finetuned-sentiment](https://huggingface.co/koheiduck/bert-japanese-finetuned-sentiment)
- [BERTの解説（原論文）](https://arxiv.org/abs/1810.04805)
