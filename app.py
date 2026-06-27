from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.stock_analysis import build_analysis, run_backtest
from src.stock_list import add_stock_row, load_stock_list, refresh_stock_rows, remove_stock_rows, save_stock_list, sort_stock_list


st.set_page_config(page_title="Stock Strategy Lab", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --bg: #0f1720;
        --panel: #151f2c;
        --panel-soft: #1c2a38;
        --text: #e8eef5;
        --muted: #8da1b6;
        --line: rgba(255, 255, 255, 0.1);
        --green: #2fd17c;
        --blue: #58a6ff;
        --amber: #f4bd50;
        --red: #ff6b6b;
    }
    .stApp {
        background:
            radial-gradient(circle at 16% 0%, rgba(88, 166, 255, 0.12), transparent 28%),
            linear-gradient(180deg, #0d141d 0%, #111a24 44%, #0d141d 100%);
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: #101923;
        border-right: 1px solid var(--line);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-weight: 700;
    }
    [data-testid="stMetricValue"] {
        color: var(--text);
    }
    [data-testid="stMetricDelta"] {
        color: var(--green);
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .hero {
        padding: 18px 0 4px;
    }
    .hero-title {
        font-size: 38px;
        font-weight: 760;
        color: var(--text);
        margin: 0;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 15px;
        margin-top: 6px;
    }
    .strategy-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 18px 0 10px;
    }
    .strategy-card {
        border: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
        border-radius: 8px;
        padding: 15px 16px;
        min-height: 126px;
    }
    .strategy-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .strategy-value {
        font-size: 30px;
        line-height: 1.1;
        font-weight: 780;
        margin: 9px 0 8px;
    }
    .strategy-note {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.35;
    }
    .accent-entry { color: var(--blue); }
    .accent-stop { color: var(--red); }
    .accent-target { color: var(--green); }
    .accent-trail { color: var(--amber); }
    .panel {
        border: 1px solid var(--line);
        background: rgba(21, 31, 44, 0.72);
        border-radius: 8px;
        padding: 18px;
    }
    .small-label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 2px;
    }
    .small-value {
        color: var(--text);
        font-size: 19px;
        font-weight: 720;
    }
    @media (max-width: 980px) {
        .strategy-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 640px) {
        .strategy-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 30px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_number(value: object, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "N/A"
        if abs(value) >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f}T{suffix}"
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B{suffix}"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M{suffix}"
        return f"{value:.2f}{suffix}"
    return str(value)


def format_percent(value: object) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2%}"
    return "N/A"


def format_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:,.2f}"


def format_score(score: int) -> str:
    return f"{score} / 100"


def format_raw_range(raw_score: int, score_range: tuple[int, int]) -> str:
    return f"素点 {raw_score}（最小 {score_range[0]} / 最大 {score_range[1]}）"


def price_card(label: str, value: str, note: str, accent: str) -> str:
    return f"""
    <div class="strategy-card">
        <div class="strategy-label">{label}</div>
        <div class="strategy-value {accent}">{value}</div>
        <div class="strategy-note">{note}</div>
    </div>
    """


def render_price_chart(analysis: dict) -> None:
    data = analysis["technicals"].dropna(subset=["Close"])
    plan = analysis["trade_plan"]
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#151f2c")
    ax.set_facecolor("#151f2c")
    ax.plot(data.index, data["Close"], label="Close", linewidth=2.0, color="#e8eef5")
    ax.plot(data.index, data["ema_21"], label="EMA 21", linewidth=1.15, color="#58a6ff")
    ax.plot(data.index, data["sma_50"], label="SMA 50", linewidth=1.15, color="#f4bd50")
    ax.plot(data.index, data["sma_200"], label="SMA 200", linewidth=1.15, color="#9b8cff")
    ax.axhline(plan.primary_entry_price, color="#58a6ff", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axhline(plan.stop_price, color="#ff6b6b", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axhline(plan.first_target_price, color="#2fd17c", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(plan.second_target_price, color="#2fd17c", linestyle=":", linewidth=1.0, alpha=0.85)
    ax.set_title(f"{analysis['ticker']} trend strategy", color="#e8eef5", pad=14)
    ax.grid(True, alpha=0.18, color="#8da1b6")
    ax.tick_params(colors="#8da1b6")
    for spine in ax.spines.values():
        spine.set_color("#2b3a49")
    legend = ax.legend(facecolor="#101923", edgecolor="#2b3a49")
    for text in legend.get_texts():
        text.set_color("#e8eef5")
    st.pyplot(fig)


def render_stock_list_manager() -> None:
    st.subheader("銘柄一覧管理")
    st.caption("銘柄はローカルCSVで管理します。保存、最新化、削除はそれぞれ明示操作です。")

    frame = load_stock_list()
    latest_update = frame["updated_at"].dropna().sort_values().iloc[-1] if not frame.empty and frame["updated_at"].notna().any() else "N/A"
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("登録銘柄", str(len(frame)))
    summary_col2.metric("最終更新", latest_update)
    summary_col3.metric("空欄数", str(int(frame.isna().sum().sum())) if not frame.empty else "0")

    sort_options = {
        "銘柄名": "name",
        "銘柄コード": "code",
        "現在の株価": "current_price",
        "夜間PTS": "pts_price",
        "売上高（今期）": "revenue_current",
        "売上高（来期）": "revenue_next",
        "純利益（今期）": "profit_current",
        "純利益（来期）": "profit_next",
        "EPS（今期予想）": "eps_current",
        "EPS（来期予想）": "eps_next",
        "EPS（再来期予想）": "eps_next2",
        "PER（今期予想）": "per_current",
        "PER（来期予想）": "per_next",
        "PER（再来期予想）": "per_next2",
        "RSI": "rsi",
        "更新日時": "updated_at",
    }

    control_col1, control_col2, control_col3 = st.columns([1.2, 0.8, 1.2])
    sort_label = control_col1.selectbox("並び替え", list(sort_options.keys()), index=2)
    ascending = control_col2.radio("順序", ["昇順", "降順"], horizontal=True, index=1) == "昇順"
    selected_codes = control_col3.multiselect("操作対象", options=frame["code"].tolist(), default=[])

    sorted_frame = sort_stock_list(frame, sort_options[sort_label], ascending=ascending)
    edited_frame = st.data_editor(
        sorted_frame,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["code", "updated_at"],
        key="stock_list_editor",
    )

    action_col1, action_col2, action_col3 = st.columns(3)
    if action_col1.button("変更を保存", type="primary"):
        save_stock_list(edited_frame)
        st.success("一覧を保存しました。")
        st.rerun()

    if action_col2.button("選択を最新化"):
        if not selected_codes:
            st.warning("最新化する銘柄を選んでください。")
        else:
            refreshed = refresh_stock_rows(edited_frame, selected_codes)
            save_stock_list(refreshed)
            st.success("選択した銘柄を最新化しました。")
            st.rerun()

    if action_col3.button("全件を最新化"):
        refreshed = refresh_stock_rows(edited_frame, None)
        save_stock_list(refreshed)
        st.success("全銘柄を最新化しました。")
        st.rerun()

    delete_confirm = st.checkbox("削除対象の削除を確認しました", value=False)
    if st.button("選択を削除", disabled=not delete_confirm):
        if not selected_codes:
            st.warning("削除対象を選んでください。")
        else:
            remaining = remove_stock_rows(edited_frame, selected_codes)
            save_stock_list(remaining)
            st.success("選択した銘柄を削除しました。")
            st.rerun()

    with st.expander("銘柄を追加", expanded=False):
        with st.form("add_stock_form", clear_on_submit=True):
            add_code = st.text_input("銘柄コード", placeholder="7203.T / AAPL / 421A")
            add_name = st.text_input("銘柄名", placeholder="任意")
            add_memo = st.text_input("メモ", placeholder="任意")
            submitted = st.form_submit_button("追加して保存", type="primary")
            if submitted:
                try:
                    updated = add_stock_row(edited_frame, add_code, add_name or None, add_memo or None)
                    save_stock_list(updated)
                    st.success("銘柄を追加しました。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"追加できませんでした: {exc}")


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Stock Strategy Lab</div>
        <div class="hero-sub">ファンダメンタルズとテクニカルを合わせて、エントリー価格・売却価格・損切りを具体化します。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("分析条件")
    ticker = st.text_input("ティッカー", value="AAPL", help="日本株は 7203.T のように入力")
    period = st.selectbox("分析期間", ["6mo", "1y", "2y", "5y"], index=2)
    risk_pct = st.slider("1回の許容損失", min_value=0.25, max_value=3.0, value=1.0, step=0.25, format="%.2f%%")
    backtest_tickers = st.text_input(
        "バックテスト銘柄",
        value=(
            "AAPL, MSFT, NVDA, GOOGL, AMZN, META, AVGO, AMD, TSLA, NFLX, "
            "421A.T, 6532.T, 8136.T, 8316.T, 8306.T, 7013.T, 7011.T, 5803.T, 6857.T, 5074.T"
        ),
    )
    st.button("分析する", type="primary")

render_stock_list_manager()

try:
    with st.spinner("株価とファンダメンタルズを取得しています..."):
        analysis = build_analysis(ticker.strip().upper(), period=period, risk_pct=risk_pct)
except Exception as exc:
    st.error(f"分析できませんでした: {exc}")
    st.stop()

fundamentals = analysis["fundamentals"]
technicals = analysis["technicals"].dropna(subset=["Close"])
latest = technicals.iloc[-1]
plan = analysis["trade_plan"]
per_forecast = analysis["per_forecast"]
news = analysis["news"]
best_practices = analysis["best_practices"]

st.subheader(f"{fundamentals['name']} ({analysis['ticker']})")
score_col, price_col, tech_col, fund_col = st.columns(4)
score_col.metric("総合判断", plan.bias, f"{format_score(plan.score)}")
price_col.metric("現在値", format_number(float(latest["Close"])))
tech_col.metric("テクニカル", format_score(analysis["technical_score"]))
fund_col.metric("ファンダメンタルズ", format_score(analysis["fundamental_score"]))

pullback_note = "押し目は一旦見送り"
if plan.secondary_entry_low is not None and plan.secondary_entry_high is not None:
    pullback_note = f"押し目候補 {format_price(plan.secondary_entry_low)} - {format_price(plan.secondary_entry_high)}"

st.markdown(
    f"""
    <div class="strategy-grid">
        {price_card("ENTRY", format_price(plan.primary_entry_price), f"高値追随の買い候補。{pullback_note}", "accent-entry")}
        {price_card("STOP", format_price(plan.stop_price), f"1株あたり想定リスク {format_price(plan.risk_per_share)}", "accent-stop")}
        {price_card("SELL 1", format_price(plan.first_target_price), "2R到達で一部利確を検討", "accent-target")}
        {price_card("SELL 2", format_price(plan.second_target_price), "3R到達、または勢い鈍化で追加利確", "accent-target")}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="panel">
        <div class="small-label">Trailing Exit</div>
        <div class="small-value">{format_price(plan.trailing_exit_price)} 割れ、または21日EMA割れで撤退を検討</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

watchlist_profile = analysis["watchlist_profile"]
if watchlist_profile is not None:
    st.subheader("監視リスト観点")
    wl1, wl2, wl3, wl4 = st.columns(4)
    wl1.metric("独自スコア", format_score(analysis["watchlist_score"]))
    wl2.metric("テーマ", watchlist_profile.theme or "N/A")
    wl3.metric("来期売上成長", format_percent(watchlist_profile.revenue_growth))
    wl4.metric("来期純利益成長", format_percent(watchlist_profile.profit_growth))
    wl5, wl6, wl7, wl8 = st.columns(4)
    wl5.metric("1期先EPS成長", format_percent(watchlist_profile.eps_growth_1y))
    wl6.metric("1期先PER", format_number(watchlist_profile.per_1y))
    wl7.metric("2期先PER", format_number(watchlist_profile.per_2y))
    wl8.metric("3期先PER", format_number(watchlist_profile.per_3y))
    if watchlist_profile.buy_story:
        st.markdown(
            f"""
            <div class="panel">
                <div class="small-label">Buy Story</div>
                <div class="small-value">{watchlist_profile.buy_story}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.subheader("監視リスト観点")
    st.info("この銘柄は読み込んだ日本株監視リストにはありません。10%以上の増収増益、3期先PER、買うストーリーの有無は未評価です。")

st.subheader("スコア内訳")
sr1, sr2, sr3 = st.columns(3)
sr1.metric("テクニカル", format_score(analysis["technical_score"]), format_raw_range(analysis["technical_raw_score"], analysis["technical_score_range"]))
sr2.metric("ファンダメンタルズ", format_score(analysis["fundamental_score"]), format_raw_range(analysis["fundamental_raw_score"], analysis["fundamental_score_range"]))
sr3.metric("監視リスト", format_score(analysis["watchlist_score"]), format_raw_range(analysis["watchlist_raw_score"], analysis["watchlist_score_range"]))

st.subheader("PER予測")
pe_tabs = st.tabs(["統合", "Yahoo概算", "監視リスト"])
with pe_tabs[0]:
    merged_per = per_forecast["merged"]
    pe1, pe2, pe3, pe4 = st.columns(4)
    pe1.metric("実績PER", format_number(merged_per["current"]))
    pe2.metric("1年後PER", format_number(merged_per["year_1"]))
    pe3.metric("2年後PER", format_number(merged_per["year_2"]))
    pe4.metric("3年後PER", format_number(merged_per["year_3"]))
    st.caption(f"統合方法: {merged_per['source']}")
with pe_tabs[1]:
    yahoo_per = per_forecast["yahoo"]
    yp1, yp2, yp3, yp4 = st.columns(4)
    yp1.metric("実績PER", format_number(yahoo_per["current"]))
    yp2.metric("1年後PER", format_number(yahoo_per["year_1"]))
    yp3.metric("2年後PER", format_number(yahoo_per["year_2"]))
    yp4.metric("3年後PER", format_number(yahoo_per["year_3"]))
    st.caption(f"Yahoo概算の成長率前提: {format_percent(yahoo_per['growth_used'])}")
with pe_tabs[2]:
    watchlist_per = per_forecast["watchlist"]
    if watchlist_per is None:
        st.info("監視リストに該当銘柄がないため、監視リスト側のPER予測はありません。")
    else:
        wp1, wp2, wp3, wp4 = st.columns(4)
        wp1.metric("実績PER", format_number(watchlist_per["current"]))
        wp2.metric("1年後PER", format_number(watchlist_per["year_1"]))
        wp3.metric("2年後PER", format_number(watchlist_per["year_2"]))
        wp4.metric("3年後PER", format_number(watchlist_per["year_3"]))
        st.caption(f"監視リストのEPS成長率前提: {format_percent(watchlist_per['growth_used'])}")

st.subheader("Entry / Exit Story")
story_col1, story_col2 = st.columns(2)
with story_col1:
    st.markdown(
        f"""
        <div class="panel">
            <div class="small-label">Entry Story</div>
            <div class="small-value">{plan.entry_story}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with story_col2:
    st.markdown(
        f"""
        <div class="panel">
            <div class="small-label">Exit Story</div>
            <div class="small-value">{plan.exit_story}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("最新ニュース")
if not news:
    st.info("日本語ニュースを取得できませんでした。ネットワーク状況やニュース配信状況を確認してください。")
else:
    news_source = news[0].get("source", "ニュース")
    st.caption(f"{news_source}を優先表示しています。日本語ニュースが取得できない場合のみYahoo Financeにフォールバックします。")
    for article in news:
        title = article["title"]
        publisher = article["publisher"]
        published_at = article["published_at"]
        url = article["url"]
        summary = article.get("summary")
        if url:
            st.markdown(f"**[{title}]({url})**")
        else:
            st.markdown(f"**{title}**")
        st.caption(f"{publisher} / {published_at}")
        if summary:
            st.write(summary)

st.subheader("調査済みベストプラクティス")
bp_tabs = st.tabs(["今回の確認", "ファンダメンタルズ", "テクニカル", "リスク管理"])
with bp_tabs[0]:
    bp_col1, bp_col2, bp_col3 = st.columns(3)
    with bp_col1:
        st.markdown("**強み**")
        for item in best_practices["strengths"] or ["明確な強みはまだ限定的です。"]:
            st.write(f"- {item}")
    with bp_col2:
        st.markdown("**注意点**")
        for item in best_practices["cautions"] or ["大きな注意点は検出されていません。"]:
            st.write(f"- {item}")
    with bp_col3:
        st.markdown("**次の確認**")
        for item in best_practices["actions"]:
            st.write(f"- {item}")
with bp_tabs[1]:
    for item in best_practices["fundamental"]:
        st.write(f"- {item}")
with bp_tabs[2]:
    for item in best_practices["technical"]:
        st.write(f"- {item}")
with bp_tabs[3]:
    for item in best_practices["risk"]:
        st.write(f"- {item}")

with st.expander("運用ルール"):
    st.write("- 投資ストーリーを組み立て、ストーリーが崩れたら売る")
    st.write("- 成長セクターに集中し、来期10%以上の増収増益を重視")
    st.write("- 3期先までPERを見て、将来の割安化がある銘柄を優先")
    st.write("- RSI70以上は短期過熱として警戒")
    st.write("- 信用取引は基本NG。買う場合も100株ずつなど分割して入る")
    st.write("- 損失許容を先に決め、risk/rewardが合う時だけupsideを狙う")

with st.expander("指標の説明"):
    st.write("**テクニカルスコア**: 200日線、50日線、55日高値、RSI、MACD、出来高から、トレンドに乗れているかを100点満点で評価します。")
    st.write("**ファンダメンタルズスコア**: 売上成長、利益成長、利益率、ROE、D/Eレシオ、FCFから、事業の質と財務の強さを100点満点で評価します。")
    st.write("**監視リストスコア**: 来期10%以上の増収増益、EPS成長、3期先PER低下、成長テーマ、買うストーリーの有無を100点満点で評価します。")
    st.write("**PER**: 株価が1株利益の何倍まで買われているかを示します。低いほど割安に見えますが、成長率や業種差も一緒に見る必要があります。")
    st.write("**2年後/3年後PER**: 現在株価が変わらない前提で、将来EPSが伸びた場合にPERがどこまで下がるかを見ます。下がるほど利益成長による割安化が期待できます。")
    st.write("**RSI**: 短期的な買われすぎ・売られすぎを見る指標です。70以上は過熱警戒、45から70程度はトレンド追随しやすい範囲として扱います。")
    st.write("**ATR**: 値動きの大きさを示す指標です。損切り幅やポジションサイズを決める時に使います。")
    st.write("**MACD**: 短期と中期の移動平均の差から勢いを見ます。ヒストグラム改善は上昇モメンタムの改善として扱います。")
    st.write("**FCF**: フリーキャッシュフローです。会計上の利益だけでなく、実際に自由に使える現金を生んでいるかを見ます。")

with st.expander("バックテスト"):
    st.caption("価格データだけで再現できるテクニカル売買ルールの検証です。ファンダメンタルズや監視リストの将来予想は、過去時点で利用できたとは限らないためバックテストには入れていません。")
    tickers_for_backtest = [item.strip().upper() for item in backtest_tickers.split(",") if item.strip()]
    run_backtest_now = st.button("バックテストを実行")
    if tickers_for_backtest and run_backtest_now:
        with st.spinner("バックテストを実行しています..."):
            backtest = run_backtest(tickers_for_backtest, period="5y", compare_strategies=True)
        summary = backtest["summary"]
        bt1, bt2, bt3, bt4 = st.columns(4)
        bt1.metric("検証銘柄", f"{summary['tested']} / {summary['tickers']}")
        bt2.metric("取引数", str(summary["trades"]))
        bt3.metric("勝率", format_percent(summary["win_rate"]))
        bt4.metric("平均損益", format_percent(summary["avg_return"]))
        bt5, bt6 = st.columns(2)
        bt5.metric("銘柄平均リターン", format_percent(summary["total_return"]))
        bt6.metric("最大ドローダウン", format_percent(summary["max_drawdown"]))

        st.subheader("改善候補の比較")
        strategy_rows = []
        for row in backtest["strategies"]:
            strategy_rows.append({
                "Strategy": row["name"],
                "Tested": row["tested"],
                "Trades": row["trades"],
                "Win Rate": format_percent(row["win_rate"]),
                "Avg Return": format_percent(row["avg_return"]),
                "Total Return": format_percent(row["total_return"]),
                "Max DD": format_percent(row["max_drawdown"]),
            })
        st.dataframe(pd.DataFrame(strategy_rows), use_container_width=True, hide_index=True)

        st.subheader("改善メモ")
        for suggestion in backtest["suggestions"]:
            st.write(f"- {suggestion}")

        st.subheader("ベース戦略の銘柄別結果")
        rows = []
        for row in backtest["results"]:
            if "error" in row:
                rows.append({"Ticker": row["ticker"], "Trades": "ERR", "Win Rate": row["error"], "Avg Return": "", "Total Return": "", "Max DD": ""})
            else:
                rows.append({
                    "Ticker": row["ticker"],
                    "Trades": row["trades"],
                    "Win Rate": format_percent(row["win_rate"]),
                    "Avg Return": format_percent(row["avg_return"]),
                    "Total Return": format_percent(row["total_return"]),
                    "Max DD": format_percent(row["max_drawdown"]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif tickers_for_backtest:
        st.info("銘柄を確認してから「バックテストを実行」を押してください。")

st.divider()

chart_col, plan_col = st.columns([1.45, 1])
with chart_col:
    render_price_chart(analysis)

with plan_col:
    st.subheader("戦略提案")
    st.markdown(f"**エントリー**  \n{plan.entry}")
    st.markdown(f"**損切り**  \n{plan.stop_loss}")
    st.markdown(f"**利確**  \n{plan.targets}")
    st.markdown(f"**出口管理**  \n{plan.trailing_exit}")

st.divider()

fund_col, tech_col = st.columns(2)
with fund_col:
    st.subheader("ファンダメンタルズ")
    f1, f2, f3 = st.columns(3)
    f1.metric("時価総額", format_number(fundamentals["market_cap"]))
    f2.metric("売上成長", format_percent(fundamentals["revenue_growth"]))
    f3.metric("利益成長", format_percent(fundamentals["earnings_growth"]))
    f4, f5, f6 = st.columns(3)
    f4.metric("PER", format_number(fundamentals["trailing_pe"]))
    f5.metric("ROE", format_percent(fundamentals["return_on_equity"]))
    f6.metric("FCF", format_number(fundamentals["free_cashflow"]))

with tech_col:
    st.subheader("テクニカル")
    t1, t2, t3 = st.columns(3)
    t1.metric("RSI 14", format_number(float(latest["rsi_14"])))
    t2.metric("ATR 14", format_number(float(latest["atr_14"])))
    t3.metric("MACD Hist", format_number(float(latest["macd_hist"])))
    t4, t5, t6 = st.columns(3)
    t4.metric("21日EMA", format_number(float(latest["ema_21"])))
    t5.metric("50日SMA", format_number(float(latest["sma_50"])))
    t6.metric("200日SMA", format_number(float(latest["sma_200"])))

reason_col, risk_col = st.columns(2)
with reason_col:
    st.subheader("根拠")
    for item in plan.thesis:
        st.write(f"- {item}")

with risk_col:
    st.subheader("注意点")
    for item in plan.risks:
        st.write(f"- {item}")

st.caption("これは投資助言ではなく、公開データに基づく分析支援です。最終判断はご自身の資金管理とリスク許容度に合わせて行ってください。")
