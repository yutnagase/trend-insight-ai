# LLM推論 — llama-cpp-python + ELYZA-JP-8B

## これは何？

LLM（Large Language Model、大規模言語モデル）は、ChatGPTのように自然な文章を生成できるAIモデルです。本プロジェクトでは、感情分析の数値データをもとに「世の中の空気感」を言葉でまとめる総評レポートの生成に使っています。

ポイントは、OpenAIなどの外部APIを使わず、自分のPC上でLLMを動かしていることです。

## 使っている技術

- **llama-cpp-python** — C++で書かれたLLM推論エンジン「llama.cpp」のPythonバインディング。GPUがなくてもCPUだけでLLMを動かせる
- **ELYZA-JP-8B** — ELYZA社が公開した日本語特化の8Bパラメータモデル。日本語の生成品質が高い
- **GGUF形式** — モデルを量子化（圧縮）して保存するファイル形式。メモリ消費を抑えつつ実用的な品質を維持できる

## なぜローカルLLMか

| 方式 | メリット | デメリット |
|------|----------|------------|
| 外部API（OpenAI等） | 高品質、セットアップ不要 | 有料、データを外部送信する |
| ローカルLLM | 無料、データが外に出ない | メモリを食う、品質はやや劣る |

本プロジェクトでは「分析対象のデータを外部に送らない」「ランニングコストゼロ」を優先してローカルLLMを採用しています。

## 量子化とは

元のLLMモデルは数十GBありますが、量子化によって精度をわずかに犠牲にしつつサイズを大幅に削減できます。

| 量子化レベル | サイズ目安（8Bモデル） | 品質 |
|-------------|----------------------|------|
| FP16（無圧縮） | ~16GB | 最高 |
| Q8 | ~8GB | ほぼ劣化なし |
| Q4_K_M | ~4.5GB | 実用的（本プロジェクトで採用） |
| Q2 | ~3GB | 日本語が崩れやすい |

Q4_K_Mは「4bit量子化、Mixed precision」の略で、重要な層は高精度を保ちつつ全体を圧縮する方式です。12GBのRAMでBERTモデルと共存できるギリギリのラインです。

## 基本的な使い方

```python
from llama_cpp import Llama

# モデルを読み込む
llm = Llama(
    model_path="models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    n_ctx=2048,      # コンテキスト長（入力+出力の最大トークン数）
    n_threads=4,     # CPUスレッド数
    n_gpu_layers=0,  # GPU不使用（CPU only）
)

# テキスト生成
output = llm(
    "日本の経済について3行で説明してください。\n回答:",
    max_tokens=256,
    temperature=0.7,
)

print(output["choices"][0]["text"])
```

## 本プロジェクトでの実装

### モデルの自動ダウンロード

初回起動時にHugging Faceからモデルファイル（約4.5GB）を自動ダウンロードします。

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="elyza/Llama-3-ELYZA-JP-8B-GGUF",
    filename="Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
    local_dir="models",
)
```

2回目以降はローカルに保存されたファイルを使うので、ダウンロードは発生しません。

### プロンプトの設計

LLMに「何をどう書いてほしいか」を伝えるのがプロンプトです。本プロジェクトでは、感情分析の結果を構造化して渡し、分析の観点を明示しています。

```python
prompt = f"""以下は「{keyword}」に関する複数ソースの感情分析データです。

■ メディア（Googleニュース30件）
  感情: ポジティブ20% / 中立70% / ネガティブ10%
  頻出語: 首相, 訪問, 経済, 連携, 会談
■ はてなブックマーク（第三者コメント）
  感情: ポジティブ10% / 中立30% / ネガティブ60%
  頻出語: 批判, 疑問, 対応, 問題, 指摘

上記データに基づき、以下の観点で3〜5行の総評レポートを日本語で作成してください。
- メディア報道と世論の間に温度差やギャップがあるか
- 各ソースで注目されているポイントの違い
- このトピックに対する世の中の空気感の総合的な読み解き

総評:"""
```

「要約して」のような曖昧な指示ではなく、具体的な観点を列挙することで、出力の方向性を安定させています。

### 生成パラメータ

```python
output = llm(
    prompt,
    max_tokens=512,      # 最大出力トークン数
    temperature=0.7,     # ランダム性（0に近いほど決定的）
    top_p=0.9,           # 上位90%の確率の語から選ぶ
    stop=["\n\n\n", "---", "以上"],  # ここで生成を止める
)
```

- `temperature=0.7` — 適度にバリエーションを持たせつつ、破綻しない程度の値
- `stop` — レポートが終わったら余計な文を生成しないように停止条件を設定

### メモリ管理

```python
@st.cache_resource
def load_llm():
    return Llama(model_path=..., n_ctx=2048, n_threads=4)
```

4.5GBのモデルを毎回読み込むと数十秒かかるので、`@st.cache_resource` でアプリ起動中は1回だけ読み込むようにしています。

## 動作に必要なスペック

- RAM: 12GB以上（BERT 500MB + ELYZA 4.5GB + OS・その他）
- ディスク: 6GB以上（モデルファイル保存用）
- CPU: 4コア以上推奨（推論速度に影響）
- GPU: 不要（あれば `n_gpu_layers` を設定して高速化可能）

推論には1回あたり30秒〜2分程度かかります（CPUのみの場合）。

## 参考リンク

- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [ELYZA-JP-8B（Hugging Face）](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF)
- [GGUF形式について](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
