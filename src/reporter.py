"""AI総評レポート生成モジュール - 軽量LLMによるインサイト生成."""

from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.analyzer import compute_net_score
from src.services.insight import compute_divergences

MODEL_REPO = "elyza/Llama-3-ELYZA-JP-8B-GGUF"
MODEL_FILE = "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
MODEL_DIR = Path("models")


@st.cache_resource
def load_llm() -> Llama:
    """LLMモデルをダウンロード・ロードする（起動時1回のみ）.

    Returns:
        Llamaインスタンス.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILE

    if not model_path.exists():
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(MODEL_DIR),
        )
        model_path = Path(downloaded)

    return Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,  # CPU only
        verbose=False,
    )


def build_prompt(
    keyword: str,
    news_stats: dict[str, float],
    news_keywords: list[tuple[str, int]],
    news_count: int = 0,
    news_samples: dict[str, list[str]] | None = None,
    bsky_stats: dict[str, float] | None = None,
    bsky_keywords: list[tuple[str, int]] | None = None,
    bsky_count: int = 0,
    bsky_samples: dict[str, list[str]] | None = None,
    hatena_stats: dict[str, float] | None = None,
    hatena_keywords: list[tuple[str, int]] | None = None,
    hatena_count: int = 0,
    hatena_samples: dict[str, list[str]] | None = None,
    topic_sentiments: dict[str, list[dict]] | None = None,
    analysis_types: list[dict] | None = None,
) -> str:
    """LLMに渡すプロンプトを構築する.

    Args:
        keyword: 検索キーワード.
        news_stats: メディアの感情比率.
        news_keywords: メディアの頻出語.
        news_count: メディア記事数.
        news_samples: メディアの代表意見.
        bsky_stats: BlueSkyの感情比率.
        bsky_keywords: BlueSkyの頻出語.
        bsky_count: BlueSky投稿数.
        bsky_samples: BlueSkyの代表意見.
        hatena_stats: はてブの感情比率.
        hatena_keywords: はてブの頻出語.
        hatena_count: はてブコメント数.
        hatena_samples: はてブの代表意見.

    Returns:
        構造化されたプロンプト文字列.
    """
    def format_stats(stats: dict[str, float]) -> str:
        score = compute_net_score(stats)
        return (
            f"スコア {score:+.2f}"
            f"（ポジ{stats['positive']:.0%} / 中立{stats['neutral']:.0%}"
            f" / ネガ{stats['negative']:.0%}）"
        )

    def format_keywords(kws: list[tuple[str, int]], top_n: int = 5) -> str:
        return "、".join(w for w, _ in kws[:top_n])

    def format_samples(samples: dict[str, list[str]] | None) -> str:
        if not samples:
            return ""
        lines = []
        for s in samples.get("positive", []):
            lines.append(f"    [ポジ] {s}")
        for s in samples.get("negative", []):
            lines.append(f"    [ネガ] {s}")
        return "\n".join(lines)

    # ソース別データ
    sections = []
    sections.append(f"■ メディア（Googleニュース {news_count}件）")
    sections.append(f"  感情: {format_stats(news_stats)}")
    sections.append(f"  頻出語: {format_keywords(news_keywords)}")
    samples_text = format_samples(news_samples)
    if samples_text:
        sections.append(f"  代表意見:\n{samples_text}")

    if bsky_stats and bsky_keywords:
        sections.append(f"■ SNS（BlueSky {bsky_count}件）")
        sections.append(f"  感情: {format_stats(bsky_stats)}")
        sections.append(f"  頻出語: {format_keywords(bsky_keywords)}")
        samples_text = format_samples(bsky_samples)
        if samples_text:
            sections.append(f"  代表意見:\n{samples_text}")

    if hatena_stats and hatena_keywords:
        sections.append(f"■ はてなブックマーク（コメント {hatena_count}件）")
        sections.append(f"  感情: {format_stats(hatena_stats)}")
        sections.append(f"  頻出語: {format_keywords(hatena_keywords)}")
        samples_text = format_samples(hatena_samples)
        if samples_text:
            sections.append(f"  代表意見:\n{samples_text}")

    data_section = "\n".join(sections)

    # 乖離情報の事前計算
    source_names = {"news": "メディア", "bsky": "BlueSky", "hatena": "はてブ"}
    net_scores: dict[str, float] = {"news": compute_net_score(news_stats)}
    if bsky_stats:
        net_scores["bsky"] = compute_net_score(bsky_stats)
    if hatena_stats:
        net_scores["hatena"] = compute_net_score(hatena_stats)

    divergence_text = ""
    if len(net_scores) >= 2:
        divergences = compute_divergences(net_scores)
        div_lines = [
            f"  {source_names[a]} vs {source_names[b]}: {g:.2f}（{l}）"
            for a, b, g, l in divergences
        ]
        divergence_text = "\n■ ソース間の乖離\n" + "\n".join(div_lines)

    # トピック別感情データ
    topic_text = ""
    if topic_sentiments:
        topic_lines = ["\n■ トピック別感情（話題×感情）"]
        source_label = {"メディア": "メディア", "BlueSky": "SNS", "はてブ": "はてブ"}
        for src, topics in topic_sentiments.items():
            if not topics:
                continue
            label = source_label.get(src, src)
            top = topics[:5]
            items = [f"{t['topic']}({t['net_score']:+.2f})" for t in top]
            topic_lines.append(f"  {label}: " + "、".join(items))
        topic_text = "\n".join(topic_lines)

    # 分析タイプ情報
    type_text = ""
    if analysis_types:
        type_labels = [
            f"{at['emoji']} {at['label']}（{at['reason']}）" for at in analysis_types
        ]
        type_text = "\n■ 分析タイプ\n  " + "\n  ".join(type_labels)

    prompt = f"""以下は「{keyword}」に関する複数ソースの感情分析データです。

{data_section}
{divergence_text}
{topic_text}
{type_text}

上記データに基づき、総合インサイトを日本語で作成してください。

以下を必ず含めてください：
- 各ソースのスコア値を引用すること
- 最も乖離が大きいソースの組み合わせとその数値を明示すること
- トピック別感情を引用し、どの話題がポジティブ／ネガティブに寄与しているか説明すること
- 代表意見を根拠として使い、なぜ差が生まれているか説明すること

■ 総合インサイト
① 概要（乖離の有無と程度）
② 数値根拠（スコアと乖離幅）
③ トピック分析（どの話題がポジ／ネガに寄与しているか）
④ 結論（このトピックの空気感）

■ 総合インサイト
①"""

    return prompt


def generate_report(
    keyword: str,
    news_stats: dict[str, float],
    news_keywords: list[tuple[str, int]],
    news_count: int = 0,
    news_samples: dict[str, list[str]] | None = None,
    bsky_stats: dict[str, float] | None = None,
    bsky_keywords: list[tuple[str, int]] | None = None,
    bsky_count: int = 0,
    bsky_samples: dict[str, list[str]] | None = None,
    hatena_stats: dict[str, float] | None = None,
    hatena_keywords: list[tuple[str, int]] | None = None,
    hatena_count: int = 0,
    hatena_samples: dict[str, list[str]] | None = None,
    topic_sentiments: dict[str, list[dict]] | None = None,
    analysis_types: list[dict] | None = None,
) -> str:
    """LLMを使って総評レポートを生成する.

    Args:
        keyword: 検索キーワード.
        news_stats: メディアの感情比率.
        news_keywords: メディアの頻出語.
        news_count: メディア記事数.
        news_samples: メディアの代表意見.
        bsky_stats: BlueSkyの感情比率.
        bsky_keywords: BlueSkyの頻出語.
        bsky_count: BlueSky投稿数.
        bsky_samples: BlueSkyの代表意見.
        hatena_stats: はてブの感情比率.
        hatena_keywords: はてブの頻出語.
        hatena_count: はてブコメント数.
        hatena_samples: はてブの代表意見.

    Returns:
        生成されたレポートテキスト.
    """
    llm = load_llm()
    prompt = build_prompt(
        keyword, news_stats, news_keywords, news_count, news_samples,
        bsky_stats, bsky_keywords, bsky_count, bsky_samples,
        hatena_stats, hatena_keywords, hatena_count, hatena_samples,
        topic_sentiments=topic_sentiments,
        analysis_types=analysis_types,
    )

    # プロンプトがn_ctxを超えないようトークン数をチェックし、超過時は代表意見を削ってリトライ
    max_input_tokens = 2048 - 512 - 64  # n_ctx - max_tokens - 余裕
    token_count = len(llm.tokenize(prompt.encode("utf-8")))
    if token_count > max_input_tokens:
        # 代表意見を削除してリトライ
        prompt = build_prompt(
            keyword, news_stats, news_keywords, news_count, None,
            bsky_stats, bsky_keywords, bsky_count, None,
            hatena_stats, hatena_keywords, hatena_count, None,
            topic_sentiments=topic_sentiments,
            analysis_types=analysis_types,
        )

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["\n\n\n", "---", "以上"],
    )

    generated = output["choices"][0]["text"].strip()
    # プロンプトで①から始めているので、先頭に①を付与して返す
    result = f"① {generated}" if not generated.startswith("①") else generated
    # ①②③④の前に改行を挿入（LLMが改行なしで出力するケース対策）
    for marker in ["②", "③", "④"]:
        result = result.replace(marker, f"\n\n{marker}")
    return result
