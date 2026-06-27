from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]
WATCHLIST_PATH = ROOT_DIR / "data" / "jp_watchlist.csv"
HIGH_CONVICTION_THEMES = {
    "AI",
    "半導体",
    "防衛",
    "フィジカルAI",
    "防衛・フィジカルAI",
    "蓄電池",
    "電線",
    "電力",
}
TECHNICAL_SCORE_RANGE = (-4, 9)
FUNDAMENTAL_SCORE_RANGE = (-4, 7)
WATCHLIST_SCORE_RANGE = (0, 12)


@dataclass(frozen=True)
class TradePlan:
    bias: str
    score: int
    primary_entry_price: float
    secondary_entry_low: float | None
    secondary_entry_high: float | None
    stop_price: float
    first_target_price: float
    second_target_price: float
    trailing_exit_price: float
    risk_per_share: float
    reward_risk_first: float
    reward_risk_second: float
    entry: str
    stop_loss: str
    targets: str
    trailing_exit: str
    entry_story: str
    exit_story: str
    thesis: list[str]
    risks: list[str]


@dataclass(frozen=True)
class WatchlistProfile:
    name: str
    code: str
    theme: str | None
    revenue_growth: float | None
    profit_growth: float | None
    eps_growth_1y: float | None
    eps_growth_2y: float | None
    per_actual: float | None
    per_1y: float | None
    per_2y: float | None
    per_3y: float | None
    buy_story: str | None


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    require_sma_200: bool = True
    max_rsi: float = 75
    require_volume: bool = False
    stop_atr: float = 2.0
    target_r: float = 2.0
    trail: str = "ema_or_low"


def fetch_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price data found for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = "date"
    return data


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    financials = _safe_financials(stock)

    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "trailing_eps": info.get("trailingEps"),
        "forward_eps": info.get("forwardEps"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "debt_to_equity": info.get("debtToEquity"),
        "free_cashflow": info.get("freeCashflow"),
        "dividend_yield": info.get("dividendYield"),
        "recommendation": info.get("recommendationKey"),
        "financials_available": financials,
    }


def fetch_company_news(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception:
        return []

    articles: list[dict[str, Any]] = []
    for item in raw_news[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title")
        if not title:
            continue

        provider = content.get("provider") or item.get("publisher") or {}
        provider_name = provider.get("displayName") if isinstance(provider, dict) else provider
        link = content.get("canonicalUrl") or content.get("clickThroughUrl") or item.get("link")
        if isinstance(link, dict):
            link = link.get("url")
        published_at = content.get("pubDate") or item.get("providerPublishTime")
        articles.append(
            {
                "title": title,
                "publisher": provider_name or "Unknown",
                "published_at": _format_news_time(published_at),
                "url": link,
                "summary": content.get("summary") or item.get("summary"),
            }
        )

    return articles


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]

    result["sma_20"] = close.rolling(20, min_periods=5).mean()
    result["sma_50"] = close.rolling(50, min_periods=10).mean()
    result["sma_200"] = close.rolling(200, min_periods=50).mean()
    result["ema_21"] = close.ewm(span=21, adjust=False).mean()
    result["high_20"] = high.rolling(20, min_periods=5).max()
    result["high_55"] = high.rolling(55, min_periods=10).max()
    result["low_20"] = low.rolling(20, min_periods=5).min()
    result["volume_sma_20"] = result["Volume"].rolling(20, min_periods=5).mean()

    result["rsi_14"] = _rsi(close, 14)
    result["atr_14"] = _atr(high, low, close, 14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    return result


def build_analysis(ticker: str, period: str = "2y", risk_pct: float = 1.0) -> dict[str, Any]:
    prices = fetch_price_history(ticker, period)
    technicals = add_technical_indicators(prices)
    latest_close = float(technicals.dropna(subset=["Close"]).iloc[-1]["Close"])
    fundamentals = fetch_fundamentals(ticker)
    news = fetch_company_news(ticker)
    watchlist_profile = find_watchlist_profile(ticker)
    technical_raw, technical_reasons, technical_risks = score_technicals(technicals)
    fundamental_raw, fundamental_reasons, fundamental_risks = score_fundamentals(fundamentals)
    watchlist_raw, watchlist_reasons, watchlist_risks = score_watchlist_profile(watchlist_profile)
    technical_score = normalize_score(technical_raw, *TECHNICAL_SCORE_RANGE)
    fundamental_score = normalize_score(fundamental_raw, *FUNDAMENTAL_SCORE_RANGE)
    watchlist_score = normalize_score(watchlist_raw, *WATCHLIST_SCORE_RANGE)
    per_forecast = estimate_future_per(latest_close, fundamentals, watchlist_profile)
    total_score = weighted_total_score(technical_score, fundamental_score, watchlist_score, watchlist_profile)
    plan = build_trade_plan(
        technicals,
        total_score,
        technical_reasons + fundamental_reasons + watchlist_reasons,
        technical_risks + fundamental_risks + watchlist_risks,
        risk_pct,
        watchlist_profile,
        per_forecast,
    )

    return {
        "ticker": ticker,
        "prices": prices,
        "technicals": technicals,
        "fundamentals": fundamentals,
        "news": news,
        "technical_score": technical_score,
        "technical_raw_score": technical_raw,
        "technical_score_range": TECHNICAL_SCORE_RANGE,
        "fundamental_score": fundamental_score,
        "fundamental_raw_score": fundamental_raw,
        "fundamental_score_range": FUNDAMENTAL_SCORE_RANGE,
        "watchlist_score": watchlist_score,
        "watchlist_raw_score": watchlist_raw,
        "watchlist_score_range": WATCHLIST_SCORE_RANGE,
        "watchlist_profile": watchlist_profile,
        "per_forecast": per_forecast,
        "total_score": total_score,
        "trade_plan": plan,
    }


def score_technicals(data: pd.DataFrame) -> tuple[int, list[str], list[str]]:
    latest = data.dropna(subset=["Close"]).iloc[-1]
    previous = data.dropna(subset=["Close"]).iloc[-2]
    score = 0
    reasons: list[str] = []
    risks: list[str] = []

    if pd.notna(latest["sma_200"]) and latest["Close"] > latest["sma_200"]:
        score += 2
        reasons.append("株価が200日移動平均を上回り、長期トレンドは上向きです。")
    elif pd.notna(latest["sma_200"]):
        score -= 2
        risks.append("株価が200日移動平均を下回り、長期トレンドが弱い状態です。")
    else:
        risks.append("上場期間または取得期間が短く、200日移動平均による長期判定は未評価です。")

    if pd.notna(latest["sma_50"]) and pd.notna(latest["sma_200"]) and latest["sma_50"] > latest["sma_200"]:
        score += 2
        reasons.append("50日移動平均が200日移動平均を上回り、中長期の上昇基調があります。")
    elif pd.notna(latest["sma_50"]) and pd.notna(latest["sma_200"]):
        score -= 1
        risks.append("50日移動平均が200日移動平均を下回り、上昇トレンドの確認が弱いです。")

    if pd.notna(latest["high_55"]) and latest["Close"] > latest["high_55"] * 0.98:
        score += 2
        reasons.append("55日高値圏にあり、ブレイクアウト候補として見られます。")
    elif pd.notna(latest["sma_50"]) and latest["Close"] > latest["sma_50"] and latest["Close"] > latest["ema_21"]:
        score += 1
        reasons.append("株価が21日EMAと50日移動平均の上にあり、押し目継続の形です。")
    else:
        risks.append("高値更新や押し目反発の形がまだ十分ではありません。")

    if pd.notna(latest["rsi_14"]) and 45 <= latest["rsi_14"] <= 70:
        score += 1
        reasons.append("RSIは過熱しすぎず、トレンド追随しやすい範囲です。")
    elif pd.notna(latest["rsi_14"]) and latest["rsi_14"] > 75:
        score -= 1
        risks.append("RSIが高く、短期的な過熱感があります。")

    if pd.notna(latest["macd_hist"]) and pd.notna(previous["macd_hist"]) and latest["macd_hist"] > previous["macd_hist"]:
        score += 1
        reasons.append("MACDの勢いが改善しています。")
    else:
        risks.append("MACDの勢いはまだ強くありません。")

    if pd.notna(latest["volume_sma_20"]) and latest["Volume"] > latest["volume_sma_20"] * 1.2:
        score += 1
        reasons.append("出来高が20日平均を上回り、買いの参加が増えています。")

    return score, reasons, risks


def score_fundamentals(fundamentals: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    risks: list[str] = []

    if _gt(fundamentals.get("revenue_growth"), 0.05):
        score += 2
        reasons.append("売上成長率がプラスで、事業の伸びが確認できます。")
    elif _lt(fundamentals.get("revenue_growth"), 0):
        score -= 1
        risks.append("売上成長率がマイナスで、成長面に懸念があります。")

    if _gt(fundamentals.get("earnings_growth"), 0.05):
        score += 2
        reasons.append("利益成長率がプラスで、業績モメンタムがあります。")
    elif _lt(fundamentals.get("earnings_growth"), 0):
        score -= 1
        risks.append("利益成長率がマイナスで、業績の勢いが弱いです。")

    if _gt(fundamentals.get("profit_margin"), 0.1):
        score += 1
        reasons.append("利益率が一定水準を超えており、収益性があります。")
    elif _lt(fundamentals.get("profit_margin"), 0):
        score -= 1
        risks.append("利益率がマイナスで、収益性に注意が必要です。")

    if _gt(fundamentals.get("return_on_equity"), 0.12):
        score += 1
        reasons.append("ROEが高めで、資本効率が良好です。")

    if _gt(fundamentals.get("debt_to_equity"), 150):
        score -= 1
        risks.append("D/Eレシオが高く、財務レバレッジに注意が必要です。")

    if _gt(fundamentals.get("free_cashflow"), 0):
        score += 1
        reasons.append("フリーキャッシュフローがプラスです。")
    elif fundamentals.get("free_cashflow") is not None:
        score -= 1
        risks.append("フリーキャッシュフローがマイナスです。")

    if fundamentals.get("trailing_pe") is None and fundamentals.get("forward_pe") is None:
        risks.append("PER情報が取得できず、バリュエーション評価は限定的です。")

    return score, reasons, risks


def find_watchlist_profile(ticker: str) -> WatchlistProfile | None:
    if not WATCHLIST_PATH.exists():
        return None

    code = _normalize_ticker_code(ticker)
    watchlist = pd.read_csv(WATCHLIST_PATH, dtype={"code": str})
    matched = watchlist[watchlist["code"].astype(str).str.upper() == code]
    if matched.empty:
        return None

    row = matched.iloc[0]
    return WatchlistProfile(
        name=str(row["name"]),
        code=str(row["code"]),
        theme=_optional_text(row.get("theme")),
        revenue_growth=_optional_float(row.get("revenue_growth")),
        profit_growth=_optional_float(row.get("profit_growth")),
        eps_growth_1y=_optional_float(row.get("eps_growth_1y")),
        eps_growth_2y=_optional_float(row.get("eps_growth_2y")),
        per_actual=_optional_float(row.get("per_actual")),
        per_1y=_optional_float(row.get("per_1y")),
        per_2y=_optional_float(row.get("per_2y")),
        per_3y=_optional_float(row.get("per_3y")),
        buy_story=_optional_text(row.get("buy_story")),
    )


def score_watchlist_profile(profile: WatchlistProfile | None) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    risks: list[str] = []

    if profile is None:
        risks.append("監視リストに該当銘柄がないため、独自の成長予想・買うストーリー評価は未反映です。")
        return 0, reasons, risks

    if profile.theme:
        reasons.append(f"監視リストのテーマは「{profile.theme}」です。")
        if any(theme in profile.theme for theme in HIGH_CONVICTION_THEMES):
            score += 1
            reasons.append("成長テーマ・国策/構造変化テーマに該当します。")

    if _gt(profile.revenue_growth, 0.1):
        score += 2
        reasons.append("来期売上成長率が10%を超えています。")
    elif _lt(profile.revenue_growth, 0.1):
        risks.append("来期売上成長率が10%未満で、成長株としては物足りない可能性があります。")

    if _gt(profile.profit_growth, 0.1):
        score += 2
        reasons.append("来期純利益成長率が10%を超えています。")
    elif _lt(profile.profit_growth, 0.1):
        risks.append("来期純利益成長率が10%未満です。")

    if _gt(profile.eps_growth_1y, 0.1):
        score += 2
        reasons.append("1期先EPS成長率が10%を超えています。")
    elif _lt(profile.eps_growth_1y, 0.1):
        risks.append("1期先EPS成長率が10%未満です。")

    if all(value is not None for value in [profile.per_actual, profile.per_1y, profile.per_2y, profile.per_3y]):
        if profile.per_1y < profile.per_actual and profile.per_2y <= profile.per_1y and profile.per_3y <= profile.per_2y:
            score += 2
            reasons.append("3期先まで予想PERが低下しており、利益成長による割安化が見込まれます。")
        else:
            risks.append("予想PERの低下が一直線ではなく、将来の割安化シナリオに確認が必要です。")

    if profile.per_3y is not None:
        if profile.per_3y <= 15:
            score += 2
            reasons.append("3期先PERが15倍以下で、成長後のバリュエーションに余地があります。")
        elif profile.per_3y <= 25:
            score += 1
            reasons.append("3期先PERは25倍以下で、過度な割高さは抑えられています。")
        else:
            risks.append("3期先PERが高く、かなり強い成長継続が必要です。")

    if profile.buy_story:
        score += 1
        reasons.append(f"買うストーリー: {profile.buy_story}")
    else:
        risks.append("買うストーリーが未記入です。ストーリーが作れない場合は見送り優先です。")

    risks.append("ストーリーが崩れたら、ファンダメンタルズでもチャートでも一度ポジション解消を優先します。")

    return score, reasons, risks


def estimate_future_per(
    close: float,
    fundamentals: dict[str, Any],
    profile: WatchlistProfile | None,
) -> dict[str, Any]:
    yahoo = estimate_yahoo_future_per(close, fundamentals)
    watchlist = None
    if profile is not None:
        watchlist = {
            "source": "監視リストの3期先予想EPS/PER",
            "current": profile.per_actual,
            "year_1": profile.per_1y,
            "year_2": profile.per_2y,
            "year_3": profile.per_3y,
            "growth_used": profile.eps_growth_1y,
        }

    merged = merge_per_forecasts(yahoo, watchlist)
    return {
        "source": merged["source"],
        "current": merged["current"],
        "year_1": merged["year_1"],
        "year_2": merged["year_2"],
        "year_3": merged["year_3"],
        "growth_used": yahoo["growth_used"],
        "yahoo": yahoo,
        "watchlist": watchlist,
        "merged": merged,
    }


def estimate_yahoo_future_per(close: float, fundamentals: dict[str, Any]) -> dict[str, Any]:
    current = _optional_float(fundamentals.get("trailing_pe"))
    year_1 = _optional_float(fundamentals.get("forward_pe"))
    trailing_eps = _optional_float(fundamentals.get("trailing_eps"))
    forward_eps = _optional_float(fundamentals.get("forward_eps"))
    growth = _optional_float(fundamentals.get("earnings_growth"))
    if growth is None:
        growth = _optional_float(fundamentals.get("revenue_growth"))
    if growth is None:
        growth = 0.0

    growth = max(min(growth, 1.0), -0.5)
    base_eps = forward_eps or trailing_eps
    year_2 = None
    year_3 = None
    if base_eps is not None and base_eps > 0:
        eps_2y = base_eps * ((1 + growth) ** 1)
        eps_3y = base_eps * ((1 + growth) ** 2)
        year_2 = close / eps_2y if eps_2y > 0 else None
        year_3 = close / eps_3y if eps_3y > 0 else None

    return {
        "source": "Yahoo FinanceのEPSと成長率からの概算",
        "current": current,
        "year_1": year_1,
        "year_2": year_2,
        "year_3": year_3,
        "growth_used": growth,
    }


def merge_per_forecasts(yahoo: dict[str, Any], watchlist: dict[str, Any] | None) -> dict[str, Any]:
    if watchlist is None:
        return {**yahoo, "source": "Yahoo FinanceのEPSと成長率からの概算"}

    return {
        "source": "Yahoo概算と監視リスト予想の平均。片方が欠ける年は取得できた側を採用",
        "current": _mean_available(yahoo.get("current"), watchlist.get("current")),
        "year_1": _mean_available(yahoo.get("year_1"), watchlist.get("year_1")),
        "year_2": _mean_available(yahoo.get("year_2"), watchlist.get("year_2")),
        "year_3": _mean_available(yahoo.get("year_3"), watchlist.get("year_3")),
        "growth_used": yahoo.get("growth_used"),
    }


def normalize_score(score: int, min_score: int, max_score: int) -> int:
    normalized = (score - min_score) / (max_score - min_score) * 100
    return int(round(max(0, min(100, normalized))))


def weighted_total_score(
    technical_score: int,
    fundamental_score: int,
    watchlist_score: int,
    profile: WatchlistProfile | None,
) -> int:
    if profile is None:
        return int(round((technical_score * 0.55) + (fundamental_score * 0.45)))
    return int(round((technical_score * 0.40) + (fundamental_score * 0.35) + (watchlist_score * 0.25)))


def build_trade_plan(
    data: pd.DataFrame,
    total_score: int,
    thesis: list[str],
    risks: list[str],
    risk_pct: float,
    profile: WatchlistProfile | None,
    per_forecast: dict[str, Any],
) -> TradePlan:
    latest = data.dropna(subset=["Close"]).iloc[-1]
    close = float(latest["Close"])
    atr = _number_or(latest["atr_14"], close * 0.03)
    high_55 = _number_or(latest["high_55"], float(data["High"].tail(55).max()))
    low_20 = _number_or(latest["low_20"], float(data["Low"].tail(20).min()))
    ema_21 = _number_or(latest["ema_21"], close)
    sma_50 = _number_or(latest["sma_50"], close)
    risk_per_share = max(atr * 2, close - low_20, close * 0.05)
    stop = close - risk_per_share
    target_2r = close + risk_per_share * 2
    target_3r = close + risk_per_share * 3
    breakout_entry = high_55 * 1.002
    pullback_entry_low = max(close - atr, min(ema_21, sma_50))
    pullback_entry_high = max(ema_21, sma_50)

    if total_score >= 70:
        bias = "買い候補"
    elif total_score >= 50:
        bias = "監視候補"
    else:
        bias = "見送り寄り"

    if close >= min(ema_21, sma_50):
        secondary_entry_low = pullback_entry_low
        secondary_entry_high = pullback_entry_high
        entry = (
            f"高値追随なら {breakout_entry:.2f} 以上で出来高増を確認。"
            f" 押し目なら {pullback_entry_low:.2f} から {pullback_entry_high:.2f} の反発確認を待つ。"
        )
    else:
        secondary_entry_low = None
        secondary_entry_high = None
        entry = (
            f"高値追随なら {breakout_entry:.2f} 以上で出来高増を確認。"
            f" 押し目狙いは見送り、まず21日EMAまたは50日SMAの回復を待つ。"
        )
    stop_loss = (
        f"初期損切り目安は {stop:.2f}。"
        f" 1回の損失は資金の {risk_pct:.1f}% 以内に収める前提。"
    )
    targets = f"利確目安は2R={target_2r:.2f}、3R={target_3r:.2f}。半分利確後は残りを伸ばす。"
    trailing_exit = (
        f"含み益が出た後は21日EMA割れ、または直近20日安値 {low_20:.2f} 割れで撤退を検討。"
    )
    entry_story = build_entry_story(profile, per_forecast, breakout_entry, secondary_entry_low, secondary_entry_high)
    exit_story = build_exit_story(profile, per_forecast, stop, target_2r, target_3r, low_20)

    return TradePlan(
        bias=bias,
        score=total_score,
        primary_entry_price=breakout_entry,
        secondary_entry_low=secondary_entry_low,
        secondary_entry_high=secondary_entry_high,
        stop_price=stop,
        first_target_price=target_2r,
        second_target_price=target_3r,
        trailing_exit_price=low_20,
        risk_per_share=risk_per_share,
        reward_risk_first=2.0,
        reward_risk_second=3.0,
        entry=entry,
        stop_loss=stop_loss,
        targets=targets,
        trailing_exit=trailing_exit,
        entry_story=entry_story,
        exit_story=exit_story,
        thesis=thesis[:8],
        risks=risks[:8],
    )


def build_entry_story(
    profile: WatchlistProfile | None,
    per_forecast: dict[str, Any],
    breakout_entry: float,
    pullback_low: float | None,
    pullback_high: float | None,
) -> str:
    parts: list[str] = []
    if profile is not None:
        if profile.theme:
            parts.append(f"テーマは「{profile.theme}」。")
        if profile.buy_story:
            parts.append(f"監視リスト上の買う理由は「{profile.buy_story}」。")
        if _gt(profile.revenue_growth, 0.1) and _gt(profile.profit_growth, 0.1):
            parts.append("来期の売上・純利益がともに10%以上伸びる前提で、成長ストーリーがあります。")
    merged = per_forecast.get("merged", per_forecast)
    if merged.get("year_3") is not None:
        per_label = "統合した"
        if per_forecast.get("watchlist") is None:
            per_label = "Yahoo概算による"
        parts.append(f"{per_label}3年後PERは{merged['year_3']:.2f}倍で、利益成長による割安化を確認します。")
    parts.append(f"価格面では{breakout_entry:.2f}以上の高値ブレイクを主シナリオにします。")
    if pullback_low is not None and pullback_high is not None:
        parts.append(f"押し目で入る場合は{pullback_low:.2f}から{pullback_high:.2f}の反発確認を待ちます。")
    else:
        parts.append("押し目狙いは急がず、移動平均線の回復を待ちます。")
    return "".join(parts)


def build_exit_story(
    profile: WatchlistProfile | None,
    per_forecast: dict[str, Any],
    stop: float,
    target_2r: float,
    target_3r: float,
    trailing_exit: float,
) -> str:
    story_break = "買うストーリーが崩れた場合"
    if profile is not None and profile.buy_story:
        story_break = f"「{profile.buy_story}」というストーリーが崩れた場合"
    parts = [
        f"初期損切りは{stop:.2f}で、想定と違ったら小さく撤退します。",
        f"利益が伸びたら{target_2r:.2f}で一部利確、{target_3r:.2f}で追加利確を検討します。",
        f"残りは20日安値{trailing_exit:.2f}割れ、または21日EMA割れでトレーリング撤退します。",
        f"{story_break}は、価格に関係なくポジション縮小または解消を優先します。",
    ]
    merged = per_forecast.get("merged", per_forecast)
    if merged.get("year_3") is not None and merged["year_3"] > 30:
        parts.append("3年後PERが高めに残るため、成長鈍化や決算ミス時は早めに利益を守ります。")
    return "".join(parts)


BACKTEST_STRATEGIES = [
    StrategyConfig(name="Base"),
    StrategyConfig(name="Volume Confirm", require_volume=True),
    StrategyConfig(name="RSI < 70", max_rsi=70),
    StrategyConfig(name="Tighter Stop", stop_atr=1.5),
    StrategyConfig(name="3R Target", target_r=3.0),
    StrategyConfig(name="Conservative", require_volume=True, max_rsi=70, stop_atr=1.5),
]


def run_backtest(tickers: list[str], period: str = "5y", compare_strategies: bool = False) -> dict[str, Any]:
    if compare_strategies:
        return run_backtest_comparison(tickers, period)

    return run_backtest_for_strategy(tickers, period, BACKTEST_STRATEGIES[0])


def run_backtest_comparison(tickers: list[str], period: str = "5y") -> dict[str, Any]:
    strategies = []
    best = None
    for config in BACKTEST_STRATEGIES:
        result = run_backtest_for_strategy(tickers, period, config)
        strategies.append({"name": config.name, **result["summary"]})
        if result["summary"]["tested"] > 0 and result["summary"]["trades"] > 0:
            if best is None or strategy_rank(result["summary"]) > strategy_rank(best):
                best = {"name": config.name, **result["summary"]}

    base = strategies[0] if strategies else None
    suggestions = build_backtest_suggestions(base, best)
    return {"summary": base, "results": run_backtest_for_strategy(tickers, period, BACKTEST_STRATEGIES[0])["results"], "strategies": strategies, "best": best, "suggestions": suggestions}


def run_backtest_for_strategy(tickers: list[str], period: str, config: StrategyConfig) -> dict[str, Any]:
    results = []
    for ticker in tickers:
        try:
            prices = fetch_price_history(ticker, period)
            data = add_technical_indicators(prices)
            result = backtest_trend_strategy(ticker, data, config)
            results.append(result)
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})

    valid = [row for row in results if "error" not in row and row["trades"] > 0]
    if not valid:
        summary = {
            "tickers": len(tickers),
            "tested": 0,
            "trades": 0,
            "win_rate": None,
            "avg_return": None,
            "total_return": None,
            "max_drawdown": None,
        }
    else:
        trades = sum(row["trades"] for row in valid)
        wins = sum(row["wins"] for row in valid)
        avg_return = sum(row["avg_return"] * row["trades"] for row in valid) / trades
        total_return = sum(row["total_return"] for row in valid) / len(valid)
        max_drawdown = min(row["max_drawdown"] for row in valid)
        summary = {
            "tickers": len(tickers),
            "tested": len(valid),
            "trades": trades,
            "win_rate": wins / trades if trades else None,
            "avg_return": avg_return,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
        }

    return {"summary": summary, "results": results}


def backtest_trend_strategy(ticker: str, data: pd.DataFrame, config: StrategyConfig | None = None) -> dict[str, Any]:
    config = config or BACKTEST_STRATEGIES[0]
    frame = data.dropna(subset=["Close", "high_55", "low_20", "ema_21", "atr_14"]).copy()
    if len(frame) < 80:
        return {"ticker": ticker, "trades": 0, "wins": 0, "win_rate": None, "avg_return": 0.0, "total_return": 0.0, "max_drawdown": 0.0}

    in_position = False
    entry_price = 0.0
    stop_price = 0.0
    half_sold = False
    trades: list[float] = []
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0

    for i in range(1, len(frame)):
        prev = frame.iloc[i - 1]
        row = frame.iloc[i]
        close = float(row["Close"])
        previous_high = float(prev["high_55"])

        if not in_position:
            trend_ok = pd.isna(row["sma_200"]) or close > float(row["sma_200"]) or not config.require_sma_200
            breakout = close > previous_high
            rsi_ok = pd.isna(row["rsi_14"]) or float(row["rsi_14"]) < config.max_rsi
            volume_ok = not config.require_volume or (
                pd.notna(row["volume_sma_20"]) and float(row["Volume"]) > float(row["volume_sma_20"]) * 1.2
            )
            if trend_ok and breakout and rsi_ok and volume_ok:
                in_position = True
                entry_price = close
                stop_price = max(entry_price - float(row["atr_14"]) * config.stop_atr, float(row["low_20"]))
                half_sold = False
            continue

        risk = max(entry_price - stop_price, entry_price * 0.03)
        target = entry_price + risk * config.target_r
        exit_price = None
        if close <= stop_price:
            exit_price = close
        elif not half_sold and close >= target:
            half_sold = True
            stop_price = entry_price
        elif half_sold and should_trailing_exit(close, row, config):
            exit_price = close
        elif should_trailing_exit(close, row, config):
            exit_price = close

        if exit_price is not None:
            trade_return = (exit_price - entry_price) / entry_price
            if half_sold:
                trade_return = (target - entry_price) / entry_price * 0.5 + trade_return * 0.5
            trades.append(trade_return)
            equity *= 1 + trade_return
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, equity / peak - 1)
            in_position = False

    wins = sum(1 for value in trades if value > 0)
    return {
        "ticker": ticker,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) if trades else None,
        "avg_return": sum(trades) / len(trades) if trades else 0.0,
        "total_return": equity - 1,
        "max_drawdown": max_drawdown,
    }


def should_trailing_exit(close: float, row: pd.Series, config: StrategyConfig) -> bool:
    if config.trail == "ema":
        return close < float(row["ema_21"])
    if config.trail == "low":
        return close < float(row["low_20"])
    return close < float(row["ema_21"]) or close < float(row["low_20"])


def strategy_rank(summary: dict[str, Any]) -> float:
    avg_return = summary.get("avg_return") or 0
    max_drawdown = abs(summary.get("max_drawdown") or 0)
    win_rate = summary.get("win_rate") or 0
    trades = summary.get("trades") or 0
    trade_penalty = 0.03 if trades < 5 else 0
    return avg_return * 3 + win_rate * 0.2 - max_drawdown * 0.7 - trade_penalty


def build_backtest_suggestions(base: dict[str, Any] | None, best: dict[str, Any] | None) -> list[str]:
    if not base or not best:
        return ["十分な取引数がありません。銘柄数や検証期間を増やしてから比較してください。"]

    suggestions = []
    if best["name"] != base["name"]:
        suggestions.append(f"今回の銘柄群では「{best['name']}」がベース戦略よりバランス良好です。")
    if (best.get("max_drawdown") or 0) > -0.1:
        suggestions.append("最大ドローダウンが比較的浅いため、現在のリスク管理は機能しています。")
    else:
        suggestions.append("最大ドローダウンが深めです。出来高確認、RSI上限、損切り幅の短縮を優先して検討してください。")
    if (best.get("win_rate") or 0) < 0.45:
        suggestions.append("勝率が低めです。高値ブレイク直後だけでなく、押し目反発条件を別ルールとして検証する余地があります。")
    if (best.get("avg_return") or 0) <= 0:
        suggestions.append("平均損益がプラスでないため、この銘柄群ではトレンド条件をさらに絞る必要があります。")
    else:
        suggestions.append("平均損益がプラスなら、勝率よりも損小利大の維持を優先してください。")
    return suggestions


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _safe_financials(stock: yf.Ticker) -> bool:
    try:
        return not stock.financials.empty
    except Exception:
        return False


def _gt(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and value > threshold


def _lt(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and value < threshold


def _normalize_ticker_code(ticker: str) -> str:
    return ticker.upper().replace(".T", "").strip()


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number_or(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and pd.notna(value):
        return float(value)
    return fallback


def _mean_available(*values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and pd.notna(value)]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _format_news_time(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, str):
        return value
    return str(value)
