"""構造化ログ設定 - structlog + stdlib logging統合."""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import structlog

LOG_DIR = Path("data/logs")


def setup_logging() -> None:
    """structlog + stdlib loggingを統合設定する.

    出力先:
    - コンソール: 人間可読形式 (INFO以上)
    - ファイル: JSON形式 (DEBUG以上, 日付ローテーション7日保持)
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- stdlib logging ハンドラ設定 ---
    # ファイルハンドラ: JSON形式
    file_handler = TimedRotatingFileHandler(
        LOG_DIR / "app.log",
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # コンソールハンドラ: 人間可読形式
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # --- structlog 設定 ---
    # 共通プロセッサ
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # structlog → stdlib logging へブリッジ
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # stdlib formatter: structlogのプロセッサで整形
    # ファイル: JSON
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    file_handler.setFormatter(json_formatter)

    # コンソール: 人間可読
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
    )
    console_handler.setFormatter(console_formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """フェーズ名付きの構造化ロガーを取得する."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
