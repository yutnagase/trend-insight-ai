"""ワードクラウド生成サービス."""

from pathlib import Path

import structlog
from wordcloud import WordCloud

logger = structlog.get_logger(__name__)

FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
IMAGES_DIR = Path("data/images")


def generate_wordcloud(word_freq: list[tuple[str, int]]) -> WordCloud | None:
    """頻出単語からワードクラウドを生成する.

    Args:
        word_freq: (単語, 出現回数)のリスト.

    Returns:
        生成されたWordCloudオブジェクト、またはデータ不足時None.
    """
    if not word_freq:
        return None
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    )
    wc.generate_from_frequencies(dict(word_freq))
    return wc


def save_wordcloud_image(wc: WordCloud, timestamp: str, source: str) -> str:
    """ワードクラウド画像をタイムスタンプ付きで保存する.

    Args:
        wc: WordCloudオブジェクト.
        timestamp: ISO形式のタイムスタンプ.
        source: ソース名（news, bsky, hatena）.

    Returns:
        保存先のファイルパス.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp.replace(":", "").replace("-", "").replace("T", "_")[:15]
    filename = f"wordcloud_{source}_{ts}.png"
    filepath = IMAGES_DIR / filename
    wc.to_file(str(filepath))
    logger.debug("ワードクラウド保存", source=source, path=str(filepath))
    return str(filepath)
