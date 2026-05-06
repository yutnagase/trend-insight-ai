# TrendInsight AI

**多角的な視点から「世の中の空気感」を読み解くAIインサイト抽出ツール**

Googleニュース、BlueSky、はてなブックマークを横断し、メディアの報道トーンと世論の乖離をローカルLLMで分析するStreamlitアプリケーション。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

TrendInsight AIは、任意のキーワードに対して以下の3つの情報源から「空気感」を抽出し、その温度差を可視化します。

| ソース             | 性質                     | 取得方法    |
| ------------------ | ------------------------ | ----------- |
| Googleニュース     | メディア報道（公式見解） | RSS         |
| BlueSky            | SNSの生の声（感情）      | AT Protocol |
| はてなブックマーク | ナレッジ層の批判的視点   | 公開API     |

すべての処理はローカルで完結し、外部APIへのデータ送信は行いません。

## 画面イメージ

### Topからワードクラウドまで

![screen1](docs/images/screen1.png)

### トレンド・キーワード TOP5 から記事・投稿一覧まで

![screen2](docs/images/screen2.png)

### AIによる総合マーケット・インサイト

![screen3](docs/images/screen3.png)

## Key Features

- **多方面データを使用した感情分析**
  - BERT日本語モデル（positive/neutral/negativeの3クラス分類）で、Googleニュース、BlueSky、はてなブックマークの論調を数値化
- **LLMによる総評出力**
  - ELYZA-8B Q4_K_M量子化モデルで総評レポートを生成。外部APIへの送信なし、費用ゼロ
- **ワードクラウド**
  - メディア・SNS・はてブそれぞれの頻出語をワードクラウドとして並べて表示し、論点の違いが一目でわかる
- **メディアと世論のギャップ検出**
  - ルールベース＋LLMで「報道と実感のズレ」を自動的に言語化
- **過去の分析結果を再閲覧**
  - ワードクラウド画像・AI総評付きで、いつでも過去分析結果を参照できる

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)            │
├─────────────────────────────────────────────────┤
│  src/clients/            │  src/analyzer.py      │
│  - GoogleNewsClient      │  - SentimentAnalyzer  │
│  - BlueskyClient         │  - BERT 3-class       │
│  - HatenaClient          │  - Dictionary Boost   │
├──────────────────────────┼───────────────────────┤
│  src/services/           │  src/reporter.py      │
│  - text_processor        │  - ELYZA-8B GGUF     │
│  - wordcloud_generator   │  - Structured Prompt  │
│  - insight / history     │                       │
├─────────────────────────────────────────────────┤
│  src/models/article.py (Pydantic)                │
└─────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.12+
- RAM 12GB以上（LLM推論で使用）
- ディスク空き容量 6GB以上（モデルファイルの保存先として必要）

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/trend-insight-ai.git
cd trend-insight-ai

# uv (推奨)
uv sync

# pip の場合
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# pipからuvに切り替える場合
rm -rf .venv
uv sync
```

### Configuration

BlueSky連携を有効にする場合、`.env`ファイルを作成します。

```bash
cp .env.example .env
```

```env
BLUESKY_HANDLE=yourname.bsky.social
BLUESKY_APP_PASSWORD=your-app-password
```

**BlueSky App Passwordの取得方法**

1. https://bsky.app にログイン
2. 設定 → アプリパスワード → 「アプリパスワードを追加」
3. 生成されたパスワードを`.env`に記入

> BlueSky未設定でもメディア＋はてなブックマークの2ソース分析は動作します。

### Run

```bash
# uv
uv run streamlit run app.py

# pip
streamlit run app.py
```

初回起動時にELYZA-8Bモデル（約4.5GB）が自動ダウンロードされます。

## Tech Stack

| Layer                  | Technology                                                   |
| ---------------------- | ------------------------------------------------------------ |
| UI                     | Streamlit                                                    |
| Sentiment Analysis     | transformers + `koheiduck/bert-japanese-finetuned-sentiment` |
| Morphological Analysis | Janome                                                       |
| Word Cloud             | wordcloud                                                    |
| LLM Inference          | llama-cpp-python + ELYZA-JP-8B (Q4_K_M GGUF)                 |
| Data Collection        | feedparser / atproto / requests                              |

各技術の詳細は [docs/](docs/) を参照してください。

## Project Structure

```
trend_insight_ai/
├── app.py                 # Streamlit UI + main flow
├── src/
│   ├── analyzer.py        # SentimentAnalyzer (BERT + dictionary boost)
│   ├── reporter.py        # LLM-based insight generation
│   ├── clients/           # Data collection clients
│   │   ├── base.py        # BaseClient abstract class
│   │   ├── google_news.py
│   │   ├── bluesky.py
│   │   └── hatena.py
│   ├── models/
│   │   └── article.py     # Pydantic Article model
│   └── services/
│       ├── history.py     # Analysis history persistence
│       ├── insight.py     # Rule-based gap detection
│       ├── text_processor.py  # Keyword extraction
│       └── wordcloud_generator.py
├── tests/                 # pytest test suite
├── data/                  # Auto-generated (gitignored)
├── models/                # Auto-downloaded GGUF (gitignored)
├── pyproject.toml         # Dependency management (uv)
├── uv.lock                # Reproducible lock file
└── .env.example
```

## License

[MIT](LICENSE)
