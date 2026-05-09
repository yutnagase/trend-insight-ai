"""アプリケーション例外定義 - エラー種別ごとのユーザー向けメッセージ."""


class TrendInsightError(Exception):
    """アプリケーション基底例外."""

    user_message: str = "予期しないエラーが発生しました。"
    user_hint: str = "時間をおいて再度お試しください。"


class DataCollectionError(TrendInsightError):
    """データ収集フェーズのエラー."""

    user_message = "データの収集中にエラーが発生しました。"
    user_hint = "ネットワーク接続を確認し、再度お試しください。"


class BlueskyAuthError(TrendInsightError):
    """BlueSky認証エラー."""

    user_message = "BlueSkyへの認証に失敗しました。"
    user_hint = (
        "サイドバーのハンドルとアプリパスワードを確認してください。"
        "アプリパスワードは https://bsky.app/settings/app-passwords で再生成できます。"
    )


class AnalysisPipelineError(TrendInsightError):
    """感情分析パイプラインのエラー."""

    user_message = "感情分析の処理中にエラーが発生しました。"
    user_hint = "分析対象のデータに問題がある可能性があります。別のキーワードでお試しください。"


class ModelLoadError(TrendInsightError):
    """AIモデルのロード/ダウンロードエラー."""

    user_message = "AIモデルの読み込みに失敗しました。"
    user_hint = (
        "初回起動時はモデルダウンロード（約4.5GB）にネットワーク接続が必要です。"
        "ディスク空き容量（6GB以上）とネットワーク接続を確認してください。"
    )


class ReportGenerationError(TrendInsightError):
    """AI総評レポート生成エラー."""

    user_message = "AI総評レポートの生成に失敗しました。"
    user_hint = (
        "メモリ不足の可能性があります（推奨: 12GB以上）。"
        "他のアプリケーションを閉じて再度お試しください。"
    )


class HistorySaveError(TrendInsightError):
    """履歴保存エラー."""

    user_message = "分析結果の保存に失敗しました。"
    user_hint = (
        "data/ ディレクトリの書き込み権限を確認してください。分析結果自体は画面に表示されています。"
    )


class HistoryLoadError(TrendInsightError):
    """履歴読み込みエラー."""

    user_message = "過去の分析履歴の読み込みに失敗しました。"
    user_hint = "data/analysis_history.json が破損している可能性があります。ファイルを削除すると復旧します。"
