# Development Insight — 技術設計の背景と意思決定

TrendInsight AIの実装で行った技術的判断と、その根拠についてまとめます。

各セクションは「課題→検討→判断→実装」の構造で記述しています。

## 1. 感情分析の進化 — 単一モデル+辞書補正からアンサンブルへ

### 課題1: 2値→3値

初期実装で採用した2値分類（positive/negative）では、ニュース見出しの大半が「ネガティブ寄り」に判定されてしまう問題がありました。

```
高市首相が帰国の途 - nippon.com → positive: 7.9%（実際は中立）
オーストラリア訪問（２） - 首相官邸 → positive: 10.4%（実際は中立）
```

原因は単純で、感情分析モデルが「感情を含まない事実報道」をうまく扱えず、確信度の低いnegative判定を返していたためです。

`koheiduck/bert-japanese-finetuned-sentiment` は実際には3クラス（POSITIVE / NEUTRAL / NEGATIVE）を出力できるモデルでした。NEUTRAL出力を正しく拾うようにしたことで、事実報道は中立として扱われるようになりました。

### 課題2: 単一モデルの中立バイアス

3クラス対応後も、ニュース記事やSNS投稿のような実データではneutral寄りに保守的に出力しやすい傾向が残りました。「死亡」「災害」「戦争」などの強いネガティブ事象を記述していても、確率が中立寄りになるケースがありました。

当初はネガティブ辞書補正（34語のハードコード）で対処していましたが、以下の問題がありました。

- **メンテナンスコスト** — トレンドの変化に追従できない
- **恣意性の批判リスク** — 「人間が選んだキーワードで結果を操作している」と見なされうる
- **情報の損失** — `1.0 - score` という変換で3クラスの確率分布を無視していた

### 解決: 複数BERTモデルのアンサンブル

辞書補正を完全に廃止し、異なるデータで学習された複数モデルの加重平均（ソフト投票）に移行しました。

| モデル | 特性 | ウェイト |
|--------|------|----------|
| `koheiduck/bert-japanese-finetuned-sentiment` | 汎用・ニュース寄り。3クラス | 0.334 |
| `christian-phu/bert-finetuned-japanese-sentiment` | レビュー特化。明確なポジ/ネガ検出力が高い | 0.333 |
| `llm-book/bert-base-japanese-v3-marc-ja` | MARC-jaデータセット。2クラスで「はっきり判定する」役割 | 0.333 |

**設計判断のポイント**:
- 全クラス確率（softmax）を直接取得し、情報の損失なし
- 「複数の学習済みモデルが合意した」という客観的な判定基準
- モデル追加・差し替えが定数変更のみで可能
- バッチ推論対応で推論速度も改善

```python
# 各モデルの全クラス確率を加重平均
for unit in self._units:
    probs = torch.softmax(model(**encoded).logits, dim=-1)
final_pos = sum(results[m]["positive"] * weights[m] for m in range(num_models))
final_neg = sum(results[m]["negative"] * weights[m] for m in range(num_models))
```

## 2. 多角的データ — 3媒体使用の意図

### 情報源ごとの論調の違い

同じトピックでも、情報源によって論調は大きく異なります。

| ソース(媒体)       | 特性                   | 例（「生成AI規制」）          |
| ------------------ | ---------------------- | ----------------------------- |
| Googleニュース     | 事実報道中心、中立寄り | 「政府がAI規制法案を提出」    |
| BlueSky            | 個人の感情・期待・不安 | 「規制されたら仕事なくなる…」 |
| はてなブックマーク | 批判的分析、論点の指摘 | 「この法案の問題点は○○」      |

この3層構造によって、「メディアは肯定的に報じているが、世論は懸念している」といった温度差を定量的に捉えられます。

### はてなブックマークを選んだ理由

当初はReddit APIを検討しましたが、2023年以降の料金改定で個人開発には厳しくなりました。はてなブックマークには以下の利点があります。

- **認証不要** — APIキー不要で即座に利用可能
- **日本語ネイティブ** — 英語圏バイアスがない
- **批判的視点が集まりやすい** — ニュースの裏側を指摘するコメント文化

「第三者のコメント」としての役割に最適でした。

### GoogleニュースURL問題の解決

GoogleニュースRSSが返すURLは `https://news.google.com/rss/articles/CBMi...` というラッパーURLで、JavaScriptリダイレクトを使っているため、`requests`では実際の記事URLに到達できません。

はてなブックマークは実際の記事URLで登録されるので、直接マッチングは不可能でした。そこで、はてなブックマーク検索RSS（`/search/text?q=keyword&mode=rss`）でキーワード直接検索する方式にしています。

## 3. ローカルLLMの選定と推論設計

### モデル選定

| 候補                     | サイズ     | 日本語品質     | 判定          |
| ------------------------ | ---------- | -------------- | ------------- |
| gemma-2b-it (Q4)         | ~1.5GB     | △ 読めるが浅い | 不採用        |
| **ELYZA-JP-8B (Q4_K_M)** | **~4.5GB** | **◎ 実用的**   | **採用**      |
| Llama-3-70B              | ~40GB      | ◎              | RAM不足で不可 |

12GB RAM + 4GB swap環境で、BERTアンサンブル（3モデル、約1.2～2.0GB）と共存できる最大サイズとしてELYZA 8B Q4_K_Mを選びました。

### 量子化の選択

Q4_K_M（4bit量子化、Mixed precision）は品質と速度のバランスが一番良い方式と今回は判断しました。Q2やQ3だと日本語の流暢さが目に見えて落ち、Q5以上だとRAM消費が許容範囲を超えます。

### プロンプト設計 — 「LLMに分析させない」原則

本プロジェクトの最も重要な設計判断は、**LLMの役割を「説明」に限定する**ことです。

#### LLM丸投げの何が問題か

一般的なLLMアプリケーションでは「このデータを分析して」とLLMに丸投げしがちですが、実際に試すと以下の問題が顕在化します。

- **再現性がない** — 同じ入力でも実行ごとに異なる結論を出す。temperature=0でも量子化モデルでは完全な再現性は保証されない
- **検証できない** — 「ネガティブが多い」とLLMが言っても、それが正しいか確認する手段がない。数値根拠がなければ反論も修正もできない
- **ハルシネーション** — 存在しないデータや傾向を「もっともらしく」語る。特に少数データでは顕著
- **デバッグ不能** — 出力がおかしいとき、プロンプトの問題かモデルの問題か切り分けられない

実際に初期プロトタイプで「以下の記事タイトルの感情傾向を分析してください」とLLMに投げたところ、同じ20件の記事に対して「全体的にポジティブ」「やや懸念が見られる」「中立的」と3回の実行で異なる結論が返ってきました。これではツールとしての信頼性が成立しません。

#### 分離の設計

| 処理 | 担当 | 理由 |
|------|------|------|
| 感情分析 | BERTアンサンブル（決定的） | 同じ入力に対して常に同じ出力 |
| 乖離計算 | コード（決定的） | 数値演算なので再現性100% |
| トピック集約 | コード（決定的） | ルールベースで検証可能 |
| 分析タイプ判定 | コード（決定的） | 条件分岐で説明可能 |
| 総評テキスト生成 | LLM（確率的） | 上記データを「引用して説明する」のみ |

LLMに渡すプロンプトには、事前計算済みのスコア・乖離値・トピック感情・分析タイプをすべて注入します。LLMは新たな分析を行うのではなく、構造化データを自然言語に変換する翻訳機として機能します。

#### この設計がもたらす実践的メリット

- **出力の監査が可能** — LLMの総評が「メディアはポジティブ（+0.35）」と述べていれば、実際のスコアと照合できる。嘘をついていれば即座にわかる
- **LLM非依存のコア価値** — LLMが停止・劣化しても、スコア・乖離・トピック分析は正常に動作する。総評は「おまけ」であり、ツールの本質的価値はコードで保証されている
- **プロンプトの安定性** — LLMに「考えさせる」のではなく「読み上げさせる」ので、プロンプトエンジニアリングの難易度が大幅に下がる。モデルを差し替えても出力品質が崩れにくい
- **テスト可能性** — 分析ロジックはすべて決定的なコードなので、ユニットテストで網羅的に検証できる。LLM部分はテスト対象外としても、ツール全体の信頼性は担保される

### メモリ管理

```python
@st.cache_resource
def load_llm() -> Llama:
    ...
```

`@st.cache_resource`を使い、モデルはアプリ起動中に1回だけロードされます。Streamlitのリラン（ボタン押下等）のたびに再ロードされることはありません。

## 4. 著作権・プライバシーへの配慮

### APIコストゼロ・ローカル完結の設計

| ソース             | 方式                   | コスト |
| ------------------ | ---------------------- | ------ |
| Googleニュース     | RSS（公開フィード）    | 無料   |
| BlueSky            | AT Protocol（公開API） | 無料   |
| はてなブックマーク | 公開JSON API           | 無料   |
| 感情分析           | ローカルBERT推論       | 無料   |
| 総評生成           | ローカルLLM推論        | 無料   |

有料APIへの依存をゼロにし、個人開発者でも継続的に運用できるようにしています。

#### なぜローカル完結にこだわるか

外部API依存のリスクは、個人開発・小規模プロジェクトにおいて致命的です。

| リスク | 具体例 | 本プロジェクトでの回避策 |
|--------|--------|------------------------|
| 料金改定 | Twitter API v2の突然の有料化（2023年）、Reddit APIの料金改定 | 認証不要の公開フィード/APIのみ使用 |
| サービス終了 | Google翻訳API無料枠廃止 | ローカルモデルで推論完結 |
| レート制限 | OpenAI APIのTPM制限で分析が中断 | ローカル推論なので制限なし |
| データ送信 | ユーザーの検索キーワードが外部に送信される | 全処理がローカルで完結 |
| 可用性 | API障害で分析不能 | ネットワーク断でも過去履歴は閲覧可能 |

#### トレードオフの認識

ローカル完結には当然コストがあります。

- **ハードウェア要件** — RAM 12GB以上、ディスク6GB以上が必要。クラウドAPIなら低スペックマシンでも動く
- **モデル品質** — GPT-4やClaude 3.5と比較すると、ELYZA-8B Q4の日本語生成品質は劣る。ただし本ツールではLLMは「説明」のみなので、品質差の影響は限定的
- **初回セットアップ** — モデルダウンロード（4.5GB）に時間がかかる。SaaS APIならアカウント作成だけで即利用可能

これらのトレードオフを受け入れた上で、「月額課金なし・外部依存なし・データ主権確保」という継続運用性を優先しています。個人開発ツールにおいて「3ヶ月後も同じように動く」ことの価値は、生成品質の差よりも大きいと判断しました。

### 出典の明記

- はてなブックマークのコメント表示時には「出典: はてなブックマーク」を表示
- 各記事・投稿には元URLへのリンクを付与
- BlueSky投稿には著者ハンドルを表示

### サーバー負荷への配慮

はてなブックマークAPIへのアクセスは以下のように制限しています。

- ブックマーク数上位5件のみに限定
- リクエスト間に0.5秒のスリープを挿入
- フレーズ検索で不要なリクエストを削減

### データの局所性

すべての分析データはローカルの`data/`ディレクトリに保存され、外部サーバーへの送信は一切ありません。ユーザーのプライバシーと分析対象のデータ主権を守る設計です。

## 5. トピック別感情分析 — 「なぜ」を説明するための設計

### 課題

ソース全体の感情集計（pos/neg）とソース間の差分（乖離）まではできていたが、「なぜその差が生まれているのか」という粒度の分析が不足していました。

例えば「MacBook Pro M4 Pro」で分析したとき、メディアはポジ寄り、はてブはネガ寄りという結果が出ても、「何についてポジティブで、何についてネガティブなのか」がわからなければインサイトとしての価値が低い。

### 解決アプローチ — 既存出力の組み合わせ

新たなモデルやライブラリを追加せず、既存のパイプライン出力を組み合わせるだけで実現しました。

```
既存: BERT感情分析済み記事リスト（各記事にlabel付き）
既存: Janomeキーワード抽出結果（単語リスト）
→ 組合せ: 各キーワードが含まれる記事のラベルを集約 → トピック別Net Sentiment Score
```

この「既存資産の掛け合わせで新しい分析軸を生む」アプローチは、依存関係を増やさず、テスト容易性も維持できる点で優れています。

### 実装のポイント

- キーワードが含まれる記事の感情ラベルをカウントし、`(pos - neg) / (pos + neg)` でNet Sentiment Score（-1.0〜+1.0）を算出
- 出現数が2件未満のトピックはノイズとして除外
- 結果はLLMプロンプトに注入し、「どの話題がポジ／ネガに寄与しているか」を説明させる

### なぜLDA等のトピックモデルを使わないか

LDA（Latent Dirichlet Allocation）やBERTopicのようなトピックモデルも検討しましたが、以下の理由で不採用としました。

- 記事数が20〜30件程度ではトピックモデルの学習が不安定
- 追加の依存ライブラリとメモリ消費が増える
- 既存のキーワード抽出（Janome）と感情ラベル（BERT）の組み合わせで十分実用的な結果が得られた

## 6. 分析タイプ判定 — 数値に「意味」を付与する

### 課題

スコアや乖離値を出すところまではできていたが、「この数値パターンが何を意味するのか」はユーザーが解釈する必要がありました。同じ「乖離0.6」でも、それが「メディアとSNSで評価方向が逆」なのか「単にスコアの絶対値が違う」のかでは意味が異なります。

### 設計判断 — 相対評価ベースの採用

当初はトピック集中型の判定に固定閾値（例: スコア < -0.6）を使う案がありましたが、以下の問題がありました。

- データ量が少ないとスコアが極端に振れやすく、誤判定が増える
- トピック分布によって「集中」の基準が変わる

#### 固定閾値で何が起きるか — 具体例

「生成AI」で検索した場合と「マイナンバー」で検索した場合を比較します。

```
「生成AI」のトピック別スコア:
  AI: +0.3, 技術: +0.2, 規制: -0.4, 雇用: -0.6, 活用: +0.1
  → 固定閾値 -0.6 だと「雇用」がギリギリ該当

「マイナンバー」のトピック別スコア:
  情報: -0.2, セキュリティ: -0.3, 利便性: +0.1, 漏洩: -0.4, 制度: -0.1
  → 固定閾値 -0.6 だと何も該当しない（全体的にネガ寄りなのに）
```

「マイナンバー」のケースでは、全体がネガティブ寄りの中で「漏洩」が特に突出しているにもかかわらず、固定閾値では検出できません。相対評価なら「この分布の中で外れ値か」を判定するため、全体の水準に関係なく突出したトピックを検出できます。

#### 相対評価の実装

**平均±1σ（標準偏差）** による相対評価を採用しました。

```python
mean = sum(scores) / len(scores)
variance = sum((s - mean) ** 2 for s in scores) / len(scores)
std = variance ** 0.5

# 平均から1σ以上離れたトピックを「集中」とみなす
neg_outliers = [t for t in topics if t["net_score"] < mean - std and t["count"] >= 5]
```

これにより、データの分布に応じて判定基準が自動調整され、安定した分類が可能になります。

#### なぜ1σか

- 正規分布を仮定すると、1σ外は約16%。トピック数10〜20の中で1〜3件が該当する粒度
- 2σだと厳しすぎて実データでほぼ検出されない
- 0.5σだと緩すぎて半数近くが「集中」になり、意味をなさない

実データでの検証を経て、1σが「ユーザーが見て納得感のある」検出粒度であることを確認しています。

### 判定ルール一覧

| タイプ | 条件 | 設計意図 |
|--------|------|----------|
| 構造的乖離 | 最大乖離 > 0.5 かつ 符号逆転 | 単なる差ではなく「方向が逆」を検出 |
| トピック集中型 | 平均±1σ かつ count≥5 | 相対評価 + 最低サンプル数で安定化 |
| 中立支配 | 全ソースneutral > 60% | 「判断材料が少ない」状態の明示 |
| 感情一致 | 全ソース同方向 かつ 乖離 < 0.2 | 合意形成の検出 |
| 混在型 | 上記非該当 | デフォルト（無理に分類しない） |

### LLMプロンプトへの注入

分析タイプはLLMプロンプトに `■ 分析タイプ` として注入されます。これにより、LLMは「このデータは構造的乖離パターンである」という前提のもとで説明を生成でき、出力の方向性が安定します。

### チューニングで得た教訓

実際の運用テストで以下の問題が発見され、対策を行いました。

- **サンプル数が少ないソースの除外** — はてブの検索結果が1件だけの場合、スコア+1.00となり他ソースとの乖離が極端になる。5件未満のソースは乖離判定から除外するようにした
- **因果関係を断定しない表現** — 「批判が集中」という表現は、キーワード含有とそのキーワードへの批判を混同する。「ネガティブ文脈で多く出現」という事実のみを述べる表現に変更した
- **トピック集中型の最低出現数引き上げ** — count≧3では偶然の一致で誤判定が起きるため、count≧5に引き上げた

## 7. キーワード抽出の品質改善

### 数字トークンの除外

Janomeは「30」「10」などの数字を名詞（数詞）として解析します。「書き出し30分から10分へ」のような文から「30」「10」がキーワードとして抽出されても意味がないため、正規表現 `^[\d,.\-+%０-９]+$` で除外しています。

### メディア名の動的除外

GoogleニュースRSSのタイトルは「記事タイトル - メディア名」形式です。特定メディアの記事が多いと、そのメディア名がキーワード上位に来てしまう問題がありました（例: 「ゴリミー」）。

静的なストップワードリストでは対応できない（メディアは無数にある）ため、タイトル末尾の` - `以降を動的に抽出し、`extra_stop_words`として除外する方式を採用しました。

```python
news_media_names = set()
for title in news_titles:
    if " - " in title:
        media = title.rsplit(" - ", 1)[-1].strip()
        if media:
            news_media_names.add(media)
            # Janomeが個別トークンに分割するため、各トークンも追加
            for token in media.split():
                if len(token) > 1:
                    news_media_names.add(token)
```

### 代表意見抽出の改善

当初の実装では、「positiveスコアが最も高い記事」と「negativeスコアが最も高い記事」を独立にソートして取得していました。データが1件しかない場合、その1件が両方のTop-1になるのは当然です。

ラベルが`positive`の記事からのみポジ代表を、`negative`の記事からのみネガ代表を抽出するように修正しました。これにより、同一コメントがポジ・ネガ両方に表示される問題を根本的に解消しています。

## 8. アーキテクチャリファクタリング — 責務分離とDI導入

### 課題

初期実装では`app.py`が551行に膨れ上がり、以下の責務が混在していました。

- UIレンダリング
- データ収集のオーケストレーション
- 感情分析の実行と統計量計算
- キーワード抽出（同一処理の3回重複呼び出し）
- 履歴保存
- エラーハンドリング

テストを書こうとすると、Streamlit・LLM・ネットワーク・ファイルI/Oすべてが絡み、単体テストが事実上不可能な状態でした。

### 設計判断 — レイヤードアーキテクチャ + DI

以下の層に分離しました。

| レイヤー | ファイル | 責務 |
|----------|----------|------|
| エントリーポイント | `app.py` (73行) | Composition Root、ルーティング |
| オーケストレーション | `orchestrator.py` | フェーズ制御、エラーハンドリング |
| UI | `ui/components.py`, `ui/pages.py` | 描画ロジック |
| コアロジック | `services/analysis_pipeline.py` | 分析パイプライン |
| インターフェース | `protocols.py` | DI用Protocol定義 |
| アダプター | `adapters.py` | Protocol具象実装 |
| 例外 | `exceptions.py` | カスタム例外階層 |

### ProtocolベースのDI

Python 3.12の`typing.Protocol`を使い、構造的部分型（Structural Subtyping）でインターフェースを定義しています。

```python
# src/protocols.py
class DataCollectorProtocol(Protocol):
    def collect(self, keyword: str) -> tuple[...]: ...

class ReportGeneratorProtocol(Protocol):
    def generate(self, result: AnalysisResult) -> str: ...
```

オーケストレーターはProtocolにのみ依存し、具象クラスを知りません。

```python
# src/orchestrator.py
class AnalysisOrchestrator:
    def __init__(
        self,
        analyzer: SentimentAnalyzerProtocol,
        collector: DataCollectorProtocol,
        report_generator: ReportGeneratorProtocol,
        history_repository: HistoryRepositoryProtocol,
    ) -> None: ...
```

app.py（Composition Root）で具象を組み立てて注入します。

```python
# app.py
orchestrator = AnalysisOrchestrator(
    analyzer=_get_analyzer(),
    collector=MultiSourceCollector(bsky_handle, bsky_password),
    report_generator=LLMReportGenerator(),
    history_repository=history_repo,
)
```

テスト時はモックを渡すだけで、外部依存ゼロでロジックを検証できます。

```python
# テスト例
orchestrator = AnalysisOrchestrator(
    analyzer=MockAnalyzer(),
    collector=MockCollector(),
    report_generator=MockReportGenerator(),
    history_repository=MockHistoryRepository(),
)
```

#### DIがテストにどう効いているか — 実証

本プロジェクトのテストスイートは、この設計の恩恵を具体的に示しています。

**パイプライン統合テスト（test_analysis_pipeline.py）** では、BERTモデル（約1.2GB、ロード10秒以上）の代わりに30行のモックを注入するだけで、パイプライン全体の動作を0.3秒で検証できます。

```python
# 実際のテストコード — GPU/モデルファイル不要で統合テスト
class MockAnalyzer:
    def analyze_batch(self, texts):
        # タイトルの内容で決定的に感情を返す
        return [{"positive": 0.8, "negative": 0.1, "label": "positive"} if "良い" in t else ...]

def test_multi_source_divergence_detected(self):
    news = _make_articles(["良い話題"] * 8, "news")
    sns = _make_articles(["悪い話題"] * 8, "bluesky")
    result = run_analysis(..., analyzer=MockAnalyzer())  # ← 注入
    assert result.divergences[0][2] > 0.3  # 乖離が検出される
```

この構成により達成していること:

| 指標 | 値 |
|------|----|
| 高速テスト（`-m "not slow"`）の実行時間 | 約30秒（59テスト） |
| BERTモデル不要で検証可能なカバレッジ | コアロジック70%超 |
| テスト追加時に必要な外部セットアップ | なし |

Protocolを使わず具象クラスに直接依存していた場合、パイプラインのテストには3つのBERTモデル（計1.2GB）のダウンロードとGPU/CPU推論が必須となり、CI環境の構築コストが跳ね上がります。

### なぜDIフレームワークを使わないか

`dependency-injector`や`inject`等のDIフレームワークも検討しましたが、以下の理由で不採用としました。

- 依存ライブラリを増やしたくない（ローカル完結の原則）
- Protocol + コンストラクタ注入で十分シンプルに実現できる
- Streamlitの`@st.cache_resource`との相性を考慮すると、手動組み立ての方が制御しやすい

### 型安全性の強化

分析結果の受け渡しを`list[dict]`から型付きモデルに変更しました。

```python
# src/models/analysis_result.py
class AnalyzedArticle(BaseModel):
    title: str
    url: str
    source: str
    positive: float
    negative: float
    label: str  # "positive" | "neutral" | "negative"

class SourceAnalysis(BaseModel):
    results: list[AnalyzedArticle]
    stats: SourceStats | None
    keywords: list[tuple[str, int]]
    samples: dict[str, list[str]]
    net_score: float  # Net Sentiment Score (-1.0〜+1.0)

class AnalysisResult(BaseModel):
    keyword: str
    news: SourceAnalysis
    bsky: SourceAnalysis
    hatena: SourceAnalysis
    topic_sentiments: dict[str, list[dict]]
    analysis_types: list[dict]
    ...
```

これにより、IDEの補完が効き、型不一致がmypy等で検出可能になりました。

## 9. エラーハンドリングと構造化ログ — ユーザビリティとObservability

### 課題

初期実装では以下の問題がありました。

- 例外発生時に生のPythonトレースバックが画面に表示される
- ユーザーが「何をすればよいか」わからない
- `print()` によるデバッグ出力が散在し、後から調査できない
- フェーズ（収集・分析・要約）ごとの所要時間やメタデータが記録されない

### 設計判断 — structlog + stdlib logging 統合

#### なぜstructlogか

| 候補 | メリット | デメリット | 判定 |
|------|----------|------------|------|
| stdlib loggingのみ | 標準ライブラリ、追加依存なし | 構造化が煩雑、コンテキストバインドがない | 不採用 |
| **structlog** | **構造化ログ、コンテキストバインド、stdlib統合** | **依存追加** | **採用** |
| loguru | 簡潔なAPI | stdlibとの統合が弱い、シングルトン設計 | 不採用 |

structlogの決め手は以下の3点です。

1. **コンテキストバインド** — `logger.bind(phase="collect", keyword=keyword)` でフェーズやキーワードを一度バインドすれば、以降のログに自動付与される
2. **stdlibブリッジ** — structlogのログをstdlib loggingのHandler経由で出力できるため、既存のTimedRotatingFileHandler等をそのまま活用可能
3. **出力形式の切り替え** — 同じログをコンソールには人間可読形式、ファイルにはJSON形式で出力できる

#### ログ設計の全体像

```python
# src/logging_config.py
def setup_logging() -> None:
    # ファイル: JSON形式、DEBUG以上、日付ローテーション7日保持
    file_handler = TimedRotatingFileHandler(
        "data/logs/app.log", when="midnight", backupCount=7
    )
    file_handler.setLevel(logging.DEBUG)

    # コンソール: 人間可読形式、INFO以上
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # structlog → stdlib ブリッジ
    structlog.configure(
        processors=[..., ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
```

| 出力先 | レベル | 形式 | 用途 |
|--------|--------|------|------|
| `data/logs/app.log` | DEBUG | JSON | 障害調査、パフォーマンス分析 |
| コンソール | INFO | 人間可読 | 開発時の動作確認 |
| 画面 | — | — | 技術的詳細は表示しない |

#### フェーズ別ログ設計

分析プロセスの各フェーズでログレベルとメタデータを制御しています。

| フェーズ | ログレベル | メタデータ例 | 設計意図 |
|----------|------------|------------|----------|
| 収集 (collect) | INFO/WARNING/ERROR | source, keyword, article_count, filtered_media | ソース別の取得件数と失敗原因を追跡 |
| 分析 (analyze) | INFO | total_articles, analysis_types, elapsed_sec | パイプライン全体の所要時間と結果概要 |
| 要約 (report) | INFO/WARNING | input_tokens, output_tokens, elapsed_sec | LLM推論のパフォーマンス計測 |
| 保存 (save) | INFO/DEBUG | total_entries | 履歴の蓄積状況 |

```python
# オーケストレーターでのフェーズ別ログ例
def _collect(self, keyword: str) -> tuple | None:
    log = logger.bind(phase="collect", keyword=keyword)
    start = time.perf_counter()
    ...
    elapsed = time.perf_counter() - start
    log.info(
        "データ収集完了",
        news_count=len(news),
        bsky_count=len(sns),
        hatena_count=len(hatena),
        elapsed_sec=round(elapsed, 2),
    )
```

```python
# LLM推論のパフォーマンス計測例 (reporter.py)
log = logger.bind(phase="report", keyword=keyword)
log.info("LLM推論開始", input_tokens=token_count)
start = time.perf_counter()
output = llm(prompt, ...)
elapsed = time.perf_counter() - start
log.info("LLM推論完了", output_tokens=output_tokens, elapsed_sec=round(elapsed, 2))
```

#### ログレベルの使い分け基準

| レベル | 用途 | 例 |
|--------|------|----|
| DEBUG | 開発時のみ有用な詳細 | キーワードtop5、ファイルパス、履歴読み込み件数 |
| INFO | 運用上有用なイベント | フェーズ開始/完了、取得件数、所要時間 |
| WARNING | 処理は続行するが注意が必要 | BlueSky取得失敗、トークン超過リトライ |
| ERROR | 処理失敗 | APIエラー、モデルロード失敗、履歴保存失敗 |

#### JSONログの出力例

`data/logs/app.log` には以下のようなJSONが1行1イベントで記録されます。

```json
{"event": "データ収集完了", "phase": "collect", "keyword": "生成AI", "news_count": 28, "bsky_count": 15, "hatena_count": 42, "elapsed_sec": 3.41, "logger": "src.orchestrator", "level": "info", "timestamp": "2024-01-15T10:23:45.123456+09:00"}
{"event": "LLM推論完了", "phase": "report", "keyword": "生成AI", "input_tokens": 1284, "output_tokens": 487, "elapsed_sec": 45.2, "logger": "src.reporter", "level": "info", "timestamp": "2024-01-15T10:24:30.456789+09:00"}
```

この形式により、`jq` やログ分析ツールでのフィルタリング・集計が容易です。

```bash
# 収集フェーズのログのみ抽出
jq 'select(.phase == "collect")' data/logs/app.log

# LLM推論の平均所要時間を確認
jq 'select(.event == "LLM推論完了") | .elapsed_sec' data/logs/app.log
```

### カスタム例外階層

```python
TrendInsightError (基底)
├── DataCollectionError      # ネットワーク障害等
├── BlueskyAuthError         # BlueSky認証失敗
├── AnalysisPipelineError    # 感情分析処理エラー
├── ModelLoadError           # AIモデルロード失敗
├── ReportGenerationError    # AI総評生成失敗（メモリ不足等）
├── HistorySaveError         # 履歴保存失敗
└── HistoryLoadError         # 履歴読み込み失敗
```

各例外は`user_message`（何が起きたか）と`user_hint`（何をすべきか）を持ちます。

```python
class ReportGenerationError(TrendInsightError):
    user_message = "AI総評レポートの生成に失敗しました。"
    user_hint = "メモリ不足の可能性があります（推奨: 12GB以上）。他のアプリケーションを閉じて再度お試しください。"
```

#### オーケストレーターでのフェーズ別ハンドリング

```python
def _collect(self, keyword: str) -> tuple | None:
    log = logger.bind(phase="collect", keyword=keyword)
    try:
        ...
    except BlueskyAuthError as e:
        _show_error(e)  # ユーザー向けメッセージ表示
        ...
    except DataCollectionError as e:
        _show_error(e)
        return None
    except Exception as e:
        log.error("予期しないエラー", error=str(e), exc_info=True)  # 構造化ログにスタックトレース
        _show_error(DataCollectionError(str(e)))  # 画面にはユーザー向けメッセージ
        return None
```

### 設計のポイント

- **フェーズごとの独立したハンドリング** — データ収集失敗でも分析済み結果は表示、AI総評失敗でも他セクションは正常表示
- **グレースフルデグラデーション** — BlueSky認証失敗時はメディア+はてブの2ソースで続行
- **例外変換はアダプター層で実施** — コアロジックは例外を投げるだけ、ユーザー向け変換はアダプターが担当
