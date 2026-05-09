"""分析パイプライン - 感情分析から統計量算出までのオーケストレーション."""

import logging
from datetime import datetime

from src.analyzer import compute_net_score, compute_sentiment_stats, select_representative
from src.models.analysis_result import (
    AnalysisResult,
    AnalyzedArticle,
    SourceAnalysis,
    SourceStats,
)
from src.models.article import Article
from src.protocols import SentimentAnalyzerProtocol
from src.services.analysis_type import detect_analysis_types
from src.services.insight import compute_divergences
from src.services.text_processor import (
    create_tokenizer,
    extract_keywords,
    extract_media_names,
)
from src.services.topic_sentiment import compute_topic_sentiments
from src.services.wordcloud_generator import generate_wordcloud, save_wordcloud_image

logger = logging.getLogger(__name__)


def _analyze_articles(
    articles: list[Article],
    analyzer: SentimentAnalyzerProtocol,
) -> list[AnalyzedArticle]:
    """Articleリストに感情分析を適用する."""
    results: list[AnalyzedArticle] = []
    for article in articles:
        scores = analyzer.analyze(article.title)
        results.append(
            AnalyzedArticle(
                title=article.title,
                url=article.url,
                source=article.source,
                author=article.author,
                positive=scores["positive"],
                negative=scores["negative"],
                label=scores["label"],
                metadata=article.metadata,
            )
        )
    return results


def _build_source_analysis(
    results: list[AnalyzedArticle],
    titles: list[str],
    search_keyword: str,
    tokenizer,
    extra_stop_words: set[str] | None = None,
) -> SourceAnalysis:
    """ソース別の統計量・キーワード・代表意見を一括算出する."""
    if not results:
        return SourceAnalysis()

    results_dicts = [r.model_dump() for r in results]
    stats_dict = compute_sentiment_stats(results_dicts)
    keywords = extract_keywords(
        titles, search_keyword, tokenizer=tokenizer, extra_stop_words=extra_stop_words
    )
    samples = select_representative(results_dicts)

    return SourceAnalysis(
        results=results,
        stats=SourceStats(**stats_dict),
        keywords=keywords,
        samples=samples,
        net_score=compute_net_score(stats_dict),
    )



def run_analysis(
    keyword: str,
    news_articles: list[Article],
    sns_articles: list[Article],
    hatena_articles: list[Article],
    hatena_entry_data: list[dict],
    analyzer: SentimentAnalyzerProtocol,
    progress_callback=None,
) -> AnalysisResult:
    """分析パイプラインを実行し、構造化された結果を返す.

    Args:
        keyword: 検索キーワード.
        news_articles: ニュース記事リスト.
        sns_articles: BlueSky投稿リスト.
        hatena_articles: はてブコメントリスト.
        hatena_entry_data: はてブエントリデータ.
        analyzer: 感情分析器.
        progress_callback: 進捗通知用コールバック (label, current, total).

    Returns:
        AnalysisResult: 全分析結果を格納した型付きオブジェクト.
    """
    tokenizer = create_tokenizer()

    # --- 感情分析 ---
    def analyze_with_progress(articles, label):
        results = []
        total = len(articles)
        for i, article in enumerate(articles):
            scores = analyzer.analyze(article.title)
            results.append(
                AnalyzedArticle(
                    title=article.title,
                    url=article.url,
                    source=article.source,
                    author=article.author,
                    positive=scores["positive"],
                    negative=scores["negative"],
                    label=scores["label"],
                    metadata=article.metadata,
                )
            )
            if progress_callback:
                progress_callback(label, i + 1, total)
        return results

    news_results = analyze_with_progress(news_articles, "メディア記事")
    sns_results = analyze_with_progress(sns_articles, "SNS投稿")
    hatena_results = analyze_with_progress(hatena_articles, "はてブコメント")

    # --- ソース別統計量（キーワード抽出は1回のみ） ---
    news_titles = [r.title for r in news_results]
    bsky_titles = [r.title for r in sns_results]
    hatena_texts = [r.title for r in hatena_results]

    news_media_names = extract_media_names(news_titles)

    news_analysis = _build_source_analysis(
        news_results, news_titles, keyword, tokenizer, extra_stop_words=news_media_names
    )
    bsky_analysis = _build_source_analysis(
        sns_results, bsky_titles, keyword, tokenizer
    )
    hatena_analysis = _build_source_analysis(
        hatena_results, hatena_texts, keyword, tokenizer
    )

    # --- トピック別感情分析 ---
    all_topic_sentiments: dict[str, list[dict]] = {}
    source_configs = [
        ("メディア", news_analysis),
        ("BlueSky", bsky_analysis),
        ("はてブ", hatena_analysis),
    ]
    for src_name, src_analysis in source_configs:
        if src_analysis.results and src_analysis.keywords:
            results_dicts = [r.model_dump() for r in src_analysis.results]
            topics = compute_topic_sentiments(results_dicts, src_analysis.keywords)
            all_topic_sentiments[src_name] = topics

    # --- 乖離検出 ---
    net_scores: dict[str, float] = {"news": news_analysis.net_score}
    neutral_ratios: dict[str, float] = {}
    if news_analysis.stats:
        neutral_ratios["news"] = news_analysis.stats.neutral
    if bsky_analysis.stats:
        net_scores["bsky"] = bsky_analysis.net_score
        neutral_ratios["bsky"] = bsky_analysis.stats.neutral
    if hatena_analysis.stats:
        net_scores["hatena"] = hatena_analysis.net_score
        neutral_ratios["hatena"] = hatena_analysis.stats.neutral

    divergences = compute_divergences(net_scores) if len(net_scores) >= 2 else []

    # --- 分析タイプ判定 ---
    combined_topics = [t for topics in all_topic_sentiments.values() for t in topics]
    max_div = divergences[0][2] if divergences else 0.0

    analysis_types = detect_analysis_types(
        net_scores=net_scores,
        max_divergence=max_div,
        topic_sentiments=combined_topics,
        neutral_ratios=neutral_ratios,
        sample_counts={
            "news": len(news_results),
            "bsky": len(sns_results),
            "hatena": len(hatena_results),
        },
    )

    # --- ワードクラウド生成 ---
    timestamp = datetime.now().isoformat()
    wordcloud_images: dict[str, str] = {}

    wc_sources = [
        ("news", news_analysis.keywords),
        ("bsky", bsky_analysis.keywords),
        ("hatena", hatena_analysis.keywords),
    ]
    for source_key, kws in wc_sources:
        if kws:
            wc = generate_wordcloud(kws)
            if wc:
                path = save_wordcloud_image(wc, timestamp, source_key)
                wordcloud_images[source_key] = path

    logger.info("分析完了: タイプ=%s", [at["label"] for at in analysis_types])

    return AnalysisResult(
        keyword=keyword,
        news=news_analysis,
        bsky=bsky_analysis,
        hatena=hatena_analysis,
        hatena_entry_data=hatena_entry_data,
        topic_sentiments=all_topic_sentiments,
        analysis_types=analysis_types,
        divergences=divergences,
        wordcloud_images=wordcloud_images,
    )
