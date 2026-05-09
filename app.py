"""TrendInsight AI - メディア・SNS・はてブの多角的トレンド感情分析アプリ."""

import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.adapters import JsonHistoryRepository, LLMReportGenerator, MultiSourceCollector
from src.analyzer import SentimentAnalyzer
from src.exceptions import HistoryLoadError
from src.orchestrator import AnalysisOrchestrator
from src.ui.pages import render_archived_analysis

load_dotenv()


def _setup_logging() -> None:
    """ロギング設定: コンソール + 日付ローテーションファイル出力."""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    # ファイルハンドラ: 日付で自動ローテーション、7日分保持
    file_handler = TimedRotatingFileHandler(
        log_dir / "app.log",
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # コンソールハンドラ: INFO以上のみ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


_setup_logging()


@st.cache_resource
def _get_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


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
            "ハンドル", value=os.getenv("BLUESKY_HANDLE", ""), placeholder="yourname.bsky.social"
        )
        bsky_password = st.text_input(
            "アプリパスワード", value=os.getenv("BLUESKY_APP_PASSWORD", ""), type="password"
        )
        st.markdown("[アプリパスワード生成](https://bsky.app/settings/app-passwords)")
        st.divider()
        st.subheader("📂 過去の分析を参照")
        history_repo = JsonHistoryRepository()
        try:
            history = history_repo.load()
        except HistoryLoadError as e:
            st.warning(f"⚠️ {e.user_message}")
            st.caption(e.user_hint)
            history = []
        history_options = ["（最新の分析）"] + [
            f"{datetime.fromisoformat(h['timestamp']).strftime('%Y-%m-%d %H:%M')} [{h['keyword']}]"
            for h in history
        ]
        selected_history = st.selectbox("履歴を選択", history_options, index=0)

    # キーワード入力
    keyword = st.text_input("分析キーワードを入力", placeholder="例: 生成AI")
    run_clicked = st.button("分析開始", disabled=not keyword)

    # ルーティング
    if selected_history != "（最新の分析）" and not run_clicked:
        selected_idx = history_options.index(selected_history) - 1
        render_archived_analysis(history[selected_idx])
        st.stop()

    if run_clicked:
        # Composition Root: 依存の組み立てと注入
        orchestrator = AnalysisOrchestrator(
            analyzer=_get_analyzer(),
            collector=MultiSourceCollector(bsky_handle, bsky_password),
            report_generator=LLMReportGenerator(),
            history_repository=history_repo,
        )
        orchestrator.execute(keyword)


if __name__ == "__main__":
    main()
