"""テキスト処理サービス - 形態素解析・キーワード抽出."""

from collections import Counter

from janome.tokenizer import Tokenizer

# ストップワード（助詞・助動詞・不要語）
STOP_WORDS: set[str] = {
    "の", "に", "は", "が", "を", "で", "と", "も", "た", "だ", "する", "いる",
    "ある", "こと", "それ", "これ", "ない", "なる", "れる", "られる", "よう",
    "さん", "ため", "から", "まで", "など", "について", "として", "における",
    "Yahoo", "ニュース", "新聞", "速報", "記事", "配信", "発表",
    "https", "http", "www", "com", "jp",
}


def create_tokenizer() -> Tokenizer:
    """Janomeトークナイザーを生成する."""
    return Tokenizer()


def extract_keywords(
    titles: list[str], search_keyword: str, tokenizer: Tokenizer | None = None
) -> list[tuple[str, int]]:
    """テキスト群から名詞を抽出し頻出順に返す.

    Args:
        titles: テキストのリスト.
        search_keyword: 除外する検索キーワード.
        tokenizer: Tokenizerインスタンス（省略時は内部生成）.

    Returns:
        (単語, 出現回数)のリスト（頻出順）.
    """
    tok = tokenizer or create_tokenizer()
    stop = STOP_WORDS | {search_keyword}
    words: list[str] = []
    for title in titles:
        for token in tok.tokenize(title):
            part = token.part_of_speech.split(",")[0]
            surface = token.surface
            if part == "名詞" and len(surface) > 1 and surface not in stop:
                words.append(surface)
    return Counter(words).most_common()
