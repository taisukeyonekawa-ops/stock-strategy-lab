from __future__ import annotations

from datetime import datetime
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

import pandas as pd
import yfinance as yf

from src.stock_analysis import (
    ROOT_DIR,
    WATCHLIST_PATH,
    add_technical_indicators,
    fetch_fundamentals,
    fetch_price_history,
    find_watchlist_profile,
)


STOCK_LIST_PATH = ROOT_DIR / "data" / "stock_list.csv"

STOCK_LIST_COLUMNS = [
    "name",
    "code",
    "current_price",
    "pts_price",
    "revenue_current",
    "revenue_next",
    "profit_current",
    "profit_next",
    "eps_current",
    "eps_next",
    "eps_next2",
    "per_current",
    "per_next",
    "per_next2",
    "rsi",
    "updated_at",
    "memo",
]


def load_stock_list() -> pd.DataFrame:
    if STOCK_LIST_PATH.exists():
        frame = pd.read_csv(STOCK_LIST_PATH)
        return normalize_stock_list(frame)

    seeded = seed_stock_list()
    save_stock_list(seeded)
    return seeded


def save_stock_list(frame: pd.DataFrame) -> None:
    STOCK_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_stock_list(frame)
    normalized.to_csv(STOCK_LIST_PATH, index=False)


def seed_stock_list() -> pd.DataFrame:
    if WATCHLIST_PATH.exists():
        watchlist = pd.read_csv(WATCHLIST_PATH, dtype={"code": str})
        rows: list[dict[str, Any]] = []
        for _, row in watchlist.iterrows():
            code = _clean_code(row.get("code"))
            price = _optional_float(row.get("price"))
            per_current = _optional_float(row.get("per_actual"))
            per_next = _optional_float(row.get("per_1y"))
            per_next2 = _optional_float(row.get("per_2y"))
            rows.append(
                {
                    "name": _optional_text(row.get("name")) or code,
                    "code": code,
                    "current_price": price,
                    "pts_price": None,
                    "revenue_current": None,
                    "revenue_next": None,
                    "profit_current": None,
                    "profit_next": None,
                    "eps_current": _derive_eps(price, per_current),
                    "eps_next": _derive_eps(price, per_next),
                    "eps_next2": _derive_eps(price, per_next2),
                    "per_current": per_current,
                    "per_next": per_next,
                    "per_next2": per_next2,
                    "rsi": None,
                    "updated_at": None,
                    "memo": _optional_text(row.get("buy_story")) or _optional_text(row.get("theme")),
                }
            )
        return normalize_stock_list(pd.DataFrame(rows))

    return normalize_stock_list(pd.DataFrame(columns=STOCK_LIST_COLUMNS))


def normalize_stock_list(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in STOCK_LIST_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result = result[STOCK_LIST_COLUMNS]

    result["name"] = result["name"].map(_optional_text)
    result["code"] = result["code"].map(_clean_code)
    for column in [
        "current_price",
        "pts_price",
        "revenue_current",
        "revenue_next",
        "profit_current",
        "profit_next",
        "eps_current",
        "eps_next",
        "eps_next2",
        "per_current",
        "per_next",
        "per_next2",
        "rsi",
    ]:
        result[column] = result[column].map(_optional_float)
    result["updated_at"] = result["updated_at"].map(_optional_text)
    result["memo"] = result["memo"].map(_optional_text)
    result = result[result["code"].notna() & (result["code"].str.strip() != "")]
    result = result.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return result


def add_stock_row(frame: pd.DataFrame, code: str, name: str | None = None, memo: str | None = None) -> pd.DataFrame:
    normalized = normalize_stock_list(frame)
    code = _clean_code(code)
    if not code:
        raise ValueError("銘柄コードが空です。")
    if code in normalized["code"].tolist():
        raise ValueError(f"{code} はすでに一覧にあります。")

    try:
        snapshot = build_stock_snapshot(code, name=name, memo=memo)
    except Exception:
        snapshot = {
            "name": _optional_text(name) or code,
            "code": code,
            "current_price": None,
            "pts_price": None,
            "revenue_current": None,
            "revenue_next": None,
            "profit_current": None,
            "profit_next": None,
            "eps_current": None,
            "eps_next": None,
            "eps_next2": None,
            "per_current": None,
            "per_next": None,
            "per_next2": None,
            "rsi": None,
            "updated_at": _now_text(),
            "memo": _optional_text(memo),
        }
    normalized = pd.concat([normalized, pd.DataFrame([snapshot])], ignore_index=True)
    return normalize_stock_list(normalized)


def remove_stock_rows(frame: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    normalized = normalize_stock_list(frame)
    targets = {_clean_code(code) for code in codes if _clean_code(code)}
    if not targets:
        return normalized
    remaining = normalized[~normalized["code"].isin(targets)].reset_index(drop=True)
    return normalize_stock_list(remaining)


def refresh_stock_rows(frame: pd.DataFrame, codes: list[str] | None = None) -> pd.DataFrame:
    normalized = normalize_stock_list(frame)
    if normalized.empty:
        return normalized

    target_codes = {_clean_code(code) for code in (codes or normalized["code"].tolist()) if _clean_code(code)}
    if not target_codes:
        return normalized

    refreshed_rows: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        code = str(row["code"])
        if code in target_codes:
            try:
                refreshed_rows.append(
                    build_stock_snapshot(
                        code,
                        name=_optional_text(row.get("name")),
                        memo=_optional_text(row.get("memo")),
                    )
                )
            except Exception:
                fallback = row.to_dict()
                fallback["updated_at"] = _now_text()
                refreshed_rows.append(fallback)
        else:
            refreshed_rows.append(row.to_dict())

    return normalize_stock_list(pd.DataFrame(refreshed_rows))


def sort_stock_list(frame: pd.DataFrame, sort_key: str, ascending: bool = True) -> pd.DataFrame:
    normalized = normalize_stock_list(frame)
    if normalized.empty:
        return normalized
    if sort_key not in normalized.columns:
        return normalized

    if sort_key in {"name", "code", "updated_at", "memo"}:
        sorted_frame = normalized.sort_values(
            by=sort_key,
            ascending=ascending,
            na_position="last",
            kind="mergesort",
        )
        return sorted_frame.reset_index(drop=True)

    numeric = pd.to_numeric(normalized[sort_key], errors="coerce")
    sorted_frame = normalized.assign(_sort_key=numeric).sort_values(
        by="_sort_key",
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )
    return sorted_frame.drop(columns=["_sort_key"]).reset_index(drop=True)


def build_stock_snapshot(code: str, name: str | None = None, memo: str | None = None) -> dict[str, Any]:
    code = _clean_code(code)
    ticker = resolve_ticker(code)
    stock = yf.Ticker(ticker)
    fundamentals = fetch_fundamentals(ticker)
    profile = find_watchlist_profile(code)

    prices = fetch_price_history(ticker, period="1y")
    technicals = add_technical_indicators(prices)
    latest = technicals.dropna(subset=["Close"]).iloc[-1]
    current_price = _optional_float(latest["Close"])
    if current_price is None:
        current_price = _optional_float(fundamentals.get("current_price"))

    pts_price = _extract_pts_price(fundamentals)
    rsi = _optional_float(latest.get("rsi_14"))

    revenue_current, revenue_next = _extract_revenue_estimates(stock, fundamentals, profile)
    profit_current, profit_next = _extract_profit_estimates(stock, fundamentals, profile)
    eps_current, eps_next, eps_next2, per_current, per_next, per_next2 = _extract_eps_and_per(
        current_price,
        fundamentals,
        profile,
    )

    return {
        "name": _optional_text(name) or _optional_text(fundamentals.get("name")) or code,
        "code": code,
        "current_price": current_price,
        "pts_price": pts_price,
        "revenue_current": revenue_current,
        "revenue_next": revenue_next,
        "profit_current": profit_current,
        "profit_next": profit_next,
        "eps_current": eps_current,
        "eps_next": eps_next,
        "eps_next2": eps_next2,
        "per_current": per_current,
        "per_next": per_next,
        "per_next2": per_next2,
        "rsi": rsi,
        "updated_at": _now_text(),
        "memo": _optional_text(memo) or _optional_text(profile.buy_story if profile else None) or _optional_text(profile.theme if profile else None),
    }


def resolve_ticker(code: str) -> str:
    code = _clean_code(code)
    if not code:
        raise ValueError("銘柄コードが空です。")

    candidates = [code]
    if not code.upper().endswith(".T"):
        candidates.append(f"{code}.T")

    for ticker in candidates:
        try:
            buffer = StringIO()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                data = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        except Exception:
            continue
        if not data.empty:
            return ticker
    return candidates[-1]


def _extract_revenue_estimates(
    stock: yf.Ticker,
    fundamentals: dict[str, Any],
    profile: Any | None,
) -> tuple[float | None, float | None]:
    current = _latest_statement_value(stock.income_stmt, ["Total Revenue", "Operating Revenue", "Revenue"])
    if current is None:
        current = _optional_float(fundamentals.get("total_revenue"))
    growth = _pick_growth(profile, fundamentals, "revenue_growth")
    next_value = _apply_growth(current, growth)
    return current, next_value


def _extract_profit_estimates(
    stock: yf.Ticker,
    fundamentals: dict[str, Any],
    profile: Any | None,
) -> tuple[float | None, float | None]:
    current = _latest_statement_value(
        stock.income_stmt,
        ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"],
    )
    if current is None:
        current = _optional_float(fundamentals.get("net_income"))
    growth = _pick_growth(profile, fundamentals, "profit_growth", fallback_key="earnings_growth")
    next_value = _apply_growth(current, growth)
    return current, next_value


def _extract_eps_and_per(
    current_price: float | None,
    fundamentals: dict[str, Any],
    profile: Any | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    if current_price is None:
        return None, None, None, None, None, None

    base_eps = _optional_float(fundamentals.get("forward_eps")) or _optional_float(fundamentals.get("trailing_eps"))
    growth = _pick_growth(profile, fundamentals, "eps_growth_1y", fallback_key="earnings_growth")

    if profile is not None and all(
        _optional_float(getattr(profile, field, None)) is not None
        for field in ("per_actual", "per_1y", "per_2y")
    ):
        per_current = _optional_float(profile.per_actual)
        per_next = _optional_float(profile.per_1y)
        per_next2 = _optional_float(profile.per_2y)
        eps_current = _derive_eps(current_price, per_current)
        eps_next = _derive_eps(current_price, per_next)
        eps_next2 = _derive_eps(current_price, per_next2)
        return eps_current, eps_next, eps_next2, per_current, per_next, per_next2

    if base_eps is None:
        return None, None, None, None, None, None

    eps_current = base_eps
    eps_next = _apply_growth(eps_current, growth)
    eps_next2 = _apply_growth(eps_next, growth) if eps_next is not None else None
    per_current = _per_from_price(current_price, eps_current)
    per_next = _per_from_price(current_price, eps_next)
    per_next2 = _per_from_price(current_price, eps_next2)
    return eps_current, eps_next, eps_next2, per_current, per_next, per_next2


def _latest_statement_value(frame: Any, labels: list[str]) -> float | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    for label in labels:
        if label not in frame.index:
            continue
        row = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
        if row.empty:
            continue
        row = row.sort_index()
        return _optional_float(row.iloc[-1])
    return None


def _extract_pts_price(fundamentals: dict[str, Any]) -> float | None:
    for key in ("post_market_price", "postMarketPrice", "pre_market_price", "preMarketPrice"):
        value = _optional_float(fundamentals.get(key))
        if value is not None:
            return value
    return None


def _pick_growth(profile: Any | None, fundamentals: dict[str, Any], profile_key: str, fallback_key: str | None = None) -> float | None:
    if profile is not None:
        value = _optional_float(getattr(profile, profile_key, None))
        if value is not None:
            return value
    value = _optional_float(fundamentals.get(profile_key))
    if value is not None:
        return value
    if fallback_key is not None:
        return _optional_float(fundamentals.get(fallback_key))
    return None


def _per_from_price(price: float | None, eps: float | None) -> float | None:
    if price is None or eps is None or eps <= 0:
        return None
    return price / eps


def _derive_eps(price: float | None, per: float | None) -> float | None:
    if price is None or per is None or per <= 0:
        return None
    return price / per


def _apply_growth(value: float | None, growth: float | None) -> float | None:
    if value is None or growth is None:
        return None
    return value * (1 + growth)


def _clean_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _now_text() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
