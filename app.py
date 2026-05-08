"""TrendInsight AI - メディア・SNS・はてブの多角的トレンド感情分析アプリ."""

import logging
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src.analyzer import SentimentAnalyzer
from src.orchestrator import AnalysisOrchestrator
from src.services.history import load_history
from src.ui.pages import render_archived_analysis

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


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
        history = load_history()
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
        orchestrator = AnalysisOrchestrator(analyzer=_get_analyzer())
        orchestrator.execute(keyword, bsky_handle, bsky_password)


if __name__ == "__main__":
    main()
