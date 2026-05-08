"""TrendInsight AI - メディア・SNS・はてブの多角的トレンド感情分析アプリ."""

import logging
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src.analyzer import SentimentAnalyzer
from src.reporter import generate_report
from src.services.analysis_pipeline import collect_data, run_analysis
from src.services.history import load_history, save_history
from src.services.text_processor import create_tokenizer
from src.ui.pages import render_archived_analysis, render_live_analysis

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


@st.cache_resource
def _get_analyzer() -> SentimentAnalyzer:
    """SentimentAnalyzerをStreamlitキャッシュで保持する."""
    return SentimentAnalyzer()


@st.cache_resource
def _get_tokenizer():
    """Janomeトークナイザーをキャッシュで保持する."""
    return create_tokenizer()


def main() -> None:
    """Streamlit UIのメインエントリポイント."""
    st.set_page_config(page_title="TrendInsight AI", page_icon="📊", layout="wide")
    st.title("📊 TrendInsight AI")
    st.caption("メディア・SNS・はてブの多角的視点から世の中の空気感を読み取る")

    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")

        st.subheader("BlueSky認証")
        bsky_handle = st.text_input(
            "ハンドル",
            value=os.getenv("BLUESKY_HANDLE", ""),
            placeholder="yourname.bsky.social",
        )
        bsky_password = st.text_input(
            "アプリパスワード",
            value=os.getenv("BLUESKY_APP_PASSWORD", ""),
            type="password",
        )
        st.markdown("[アプリパスワード生成](https://bsky.app/settings/app-passwords)")

        st.divider()
        st.subheader("📂 過去の分析を参照")
        history = load_history()
        history_options = ["（最新の分析）"] + [
            f"{datetime.fromisoformat(h['timestamp']).strftime('%Y-%m-%d %H:%M')} [{h['keyword']}]"
            for h in history
        ]
        selected_history = st.selectbox("履歴を選択", history_options, index=0)

    # メイン: キーワード入力
    keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")
    run_clicked = st.button("分析開始", disabled=not keyword)

    # 過去データ表示モード
    if selected_history != "（最新の分析）" and not run_clicked:
        selected_idx = history_options.index(selected_history) - 1
        render_archived_analysis(history[selected_idx])
        st.stop()

    if run_clicked:
        _run_analysis(keyword, bsky_handle, bsky_password)


def _run_analysis(keyword: str, bsky_handle: str, bsky_password: str) -> None:
    """分析メインフローを実行する."""
    # --- データ収集 ---
    with st.spinner("📰 データを収集中..."):
        news_articles, sns_articles, hatena_articles, hatena_entry_data = collect_data(
            keyword, bsky_handle, bsky_password
        )

    if not news_articles and not sns_articles:
        st.warning("記事・投稿が見つかりませんでした。")
        return

    if not news_articles:
        st.warning("ニュース記事の取得に失敗しました。")
    if not sns_articles and bsky_handle:
        st.warning("BlueSky投稿の取得に失敗しました。")
    elif not bsky_handle:
        st.info("💡 サイドバーでBlueSky認証を設定すると、SNSの声も分析できます。")

    # --- 分析パイプライン実行 ---
    analyzer = _get_analyzer()
    progress_bar = st.progress(0, text="分析中...")

    def on_progress(label: str, current: int, total: int) -> None:
        progress_bar.progress(current / total, text=f"{label}を分析中...")

    result = run_analysis(
        keyword=keyword,
        news_articles=news_articles,
        sns_articles=sns_articles,
        hatena_articles=hatena_articles,
        hatena_entry_data=hatena_entry_data,
        analyzer=analyzer,
        progress_callback=on_progress,
    )
    progress_bar.empty()

    # --- AI総評レポート生成 ---
    with st.spinner("🧠 AIが総評レポートを生成中（初回はモデルダウンロードのため数分かかります）..."):
        try:
            result.ai_report = generate_report(
                keyword=keyword,
                news_stats=result.news.stats.model_dump() if result.news.stats else {},
                news_keywords=result.news.keywords,
                news_count=len(result.news.results),
                news_samples=result.news.samples or None,
                bsky_stats=result.bsky.stats.model_dump() if result.bsky.stats else None,
                bsky_keywords=result.bsky.keywords or None,
                bsky_count=len(result.bsky.results),
                bsky_samples=result.bsky.samples or None,
                hatena_stats=result.hatena.stats.model_dump() if result.hatena.stats else None,
                hatena_keywords=result.hatena.keywords or None,
                hatena_count=len(result.hatena.results),
                hatena_samples=result.hatena.samples or None,
                topic_sentiments=result.topic_sentiments,
                analysis_types=result.analysis_types,
            )
        except Exception as e:
            logging.getLogger(__name__).error("AI総評生成失敗: %s", e, exc_info=True)
            st.warning(f"AI総評レポートの生成に失敗しました: {e}")

    # --- 結果表示 ---
    render_live_analysis(result)

    # --- 履歴保存 ---
    save_history(
        keyword,
        [r.model_dump() for r in result.news.results],
        [r.model_dump() for r in result.bsky.results],
        [r.model_dump() for r in result.hatena.results],
        result.wordcloud_images,
        result.ai_report,
    )
    st.info("💾 分析結果を履歴に保存しました。")


if __name__ == "__main__":
    main()
