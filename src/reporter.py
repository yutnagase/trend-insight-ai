"""AI総評レポート生成モジュール - 軽量LLMによるインサイト生成."""

from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

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
    bsky_stats: dict[str, float] | None = None,
    bsky_keywords: list[tuple[str, int]] | None = None,
    hatena_stats: dict[str, float] | None = None,
    hatena_keywords: list[tuple[str, int]] | None = None,
) -> str:
    """LLMに渡すプロンプトを構築する.

    Args:
        keyword: 検索キーワード.
        news_stats: メディアの感情比率.
        news_keywords: メディアの頻出語.
        bsky_stats: BlueSkyの感情比率.
        bsky_keywords: BlueSkyの頻出語.
        hatena_stats: はてブの感情比率.
        hatena_keywords: はてブの頻出語.

    Returns:
        構造化されたプロンプト文字列.
    """
    def format_stats(stats: dict[str, float]) -> str:
        return (
            f"ポジティブ{stats['positive']:.0%} / "
            f"中立{stats['neutral']:.0%} / "
            f"ネガティブ{stats['negative']:.0%}"
        )

    def format_keywords(kws: list[tuple[str, int]], top_n: int = 5) -> str:
        return "、".join(w for w, _ in kws[:top_n])

    sections = []
    sections.append(f"■ メディア（Googleニュース30件）")
    sections.append(f"  感情: {format_stats(news_stats)}")
    sections.append(f"  頻出語: {format_keywords(news_keywords)}")

    if bsky_stats and bsky_keywords:
        sections.append(f"■ SNS（BlueSky）")
        sections.append(f"  感情: {format_stats(bsky_stats)}")
        sections.append(f"  頻出語: {format_keywords(bsky_keywords)}")

    if hatena_stats and hatena_keywords:
        sections.append(f"■ はてなブックマーク（第三者コメント）")
        sections.append(f"  感情: {format_stats(hatena_stats)}")
        sections.append(f"  頻出語: {format_keywords(hatena_keywords)}")

    data_section = "\n".join(sections)

    prompt = f"""以下は「{keyword}」に関する複数ソースの感情分析データです。

{data_section}

上記データに基づき、以下の観点で3〜5行の総評レポートを日本語で作成してください。
- メディア報道と世論（SNS・はてブ）の間に温度差やギャップがあるか
- 各ソースで注目されているポイントの違い
- このトピックに対する世の中の空気感の総合的な読み解き

総評:"""

    return prompt


def generate_report(
    keyword: str,
    news_stats: dict[str, float],
    news_keywords: list[tuple[str, int]],
    bsky_stats: dict[str, float] | None = None,
    bsky_keywords: list[tuple[str, int]] | None = None,
    hatena_stats: dict[str, float] | None = None,
    hatena_keywords: list[tuple[str, int]] | None = None,
) -> str:
    """LLMを使って総評レポートを生成する.

    Args:
        keyword: 検索キーワード.
        news_stats: メディアの感情比率.
        news_keywords: メディアの頻出語.
        bsky_stats: BlueSkyの感情比率.
        bsky_keywords: BlueSkyの頻出語.
        hatena_stats: はてブの感情比率.
        hatena_keywords: はてブの頻出語.

    Returns:
        生成されたレポートテキスト.
    """
    llm = load_llm()
    prompt = build_prompt(
        keyword, news_stats, news_keywords,
        bsky_stats, bsky_keywords,
        hatena_stats, hatena_keywords,
    )

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["\n\n\n", "---", "以上"],
    )

    return output["choices"][0]["text"].strip()
