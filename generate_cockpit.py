#!/usr/bin/env python3
"""
Market Capitulation Cockpit — Zero-Noise Local HTML Dashboard Generator

Philosophy:
  "Once people lose all hope, that's when we look for buying opportunities.
   If there's a little ounce of hope, we keep waiting."

Fetches live daily OHLCV data via yfinance (zero news, zero social media, zero YouTube),
computes an objective 0-100 Capitulation / Despair Score across broad market + AI rotation
tickers, and renders a self-contained local HTML cockpit at:
  ~/code/trading/market_cockpit.html

Usage:
  python3 ~/code/trading/generate_cockpit.py
  python3 ~/code/trading/generate_cockpit.py --open
"""

import argparse
import datetime as dt
import html
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUTPUT_PATH = Path(__file__).resolve().parent / "docs" / "index.html"

UNIVERSE = [
    # Broad Market & Macro
    {"ticker": "SPY", "name": "S&P 500", "group": "Broad Market"},
    {"ticker": "QQQ", "name": "Nasdaq 100", "group": "Broad Market"},
    {"ticker": "IWM", "name": "Russell 2000", "group": "Broad Market"},
    {"ticker": "TLT", "name": "20Y+ Treasuries", "group": "Macro Hedge"},
    # AI Core & Semis
    {"ticker": "NVDA", "name": "Nvidia", "group": "AI Core"},
    {"ticker": "SMH", "name": "Semiconductors ETF", "group": "AI Core"},
    {"ticker": "AVGO", "name": "Broadcom", "group": "AI Core"},
    {"ticker": "AMD", "name": "AMD", "group": "AI Core"},
    {"ticker": "MSFT", "name": "Microsoft", "group": "AI Hyperscaler"},
    {"ticker": "META", "name": "Meta Platforms", "group": "AI Hyperscaler"},
    # AI Infrastructure & Power Rotation
    {"ticker": "CRWV", "name": "CoreWeave", "group": "AI Infra"},
    {"ticker": "VST", "name": "Vistra", "group": "AI Power"},
    {"ticker": "OKLO", "name": "Oklo", "group": "AI Power"},
    {"ticker": "CEG", "name": "Constellation Energy", "group": "AI Power"},
]

VOL_TICKERS = ["^VIX", "^VIX3M"]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_sparkline_svg(values: list[float], color: str) -> str:
    if len(values) < 2:
        return ""
    w, h = 112, 30
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    pts = []
    for i, v in enumerate(values):
        x = (i / (len(values) - 1)) * (w - 4) + 2
        y = h - 3 - ((v - lo) / span) * (h - 6)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    last_x, last_y = pts[-1].split(",")
    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" points="{poly}"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.3" fill="{color}"/>'
        f"</svg>"
    )


def clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def fetch_and_score() -> dict:
    all_tickers = [u["ticker"] for u in UNIVERSE] + VOL_TICKERS
    raw = yf.download(all_tickers, period="6mo", interval="1d", progress=False, auto_adjust=True)
    close = raw["Close"]
    vol = raw["Volume"]

    latest_dt = close.index[-1]
    latest_date_str = latest_dt.strftime("%Y-%m-%d")

    # VIX & Term Structure
    vix_s = close["^VIX"].dropna()
    vix_last = float(vix_s.iloc[-1])
    vix_prev = float(vix_s.iloc[-2]) if len(vix_s) > 1 else vix_last
    vix_peak_20d = float(vix_s.tail(20).max())
    vix_peak_date = vix_s.tail(20).idxmax().strftime("%b %d")

    vix3m_last = None
    vix_ratio = None
    if "^VIX3M" in close.columns and len(close["^VIX3M"].dropna()) > 0:
        vix3m_last = float(close["^VIX3M"].dropna().iloc[-1])
        if vix3m_last > 0:
            vix_ratio = vix_last / vix3m_last

    rows = []
    for item in UNIVERSE:
        t = item["ticker"]
        if t not in close.columns:
            continue
        s = close[t].dropna()
        v = vol[t].dropna() if t in vol.columns else pd.Series(dtype=float)
        if len(s) < 25:
            continue

        last = float(s.iloc[-1])
        prev = float(s.iloc[-2])
        chg_1d = (last / prev - 1) * 100
        chg_5d = (last / float(s.iloc[-6]) - 1) * 100 if len(s) >= 6 else 0.0

        hi_4m = float(s.tail(85).max())
        hi_date = s.tail(85).idxmax().strftime("%b %d")
        dd_hi = (last / hi_4m - 1) * 100

        lo_20d = float(s.tail(20).min())
        lo_date = s.tail(20).idxmin().strftime("%b %d")
        off_lo = (last / lo_20d - 1) * 100

        sma20 = float(s.rolling(20).mean().iloc[-1])
        sma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else sma20
        vs_sma50 = (last / sma50 - 1) * 100
        rsi14 = float(compute_rsi(s, 14).iloc[-1])

        vol_ratio = 1.0
        if len(v) >= 21 and float(v.iloc[-21:-1].mean()) > 0:
            vol_ratio = float(v.iloc[-1]) / float(v.iloc[-21:-1].mean())

        # Per-ticker Hope vs Despair status (for non-hedge equities)
        if item["group"] == "Macro Hedge":
            state = "HEDGE"
            state_class = "badge-neutral"
            action_note = "Duration buffer"
        else:
            # Despair requires deep discount AND no fresh euphoria/hope bounce
            is_deeply_oversold = rsi14 <= 32 and vs_sma50 <= -12
            is_capitulating_today = chg_1d <= -5 and vol_ratio >= 1.8 and vs_sma50 <= -10
            is_basing_quiet = (abs(chg_1d) < 3.5) and (off_lo > 4.0) and (off_lo < 15.0) and (vs_sma50 < -5)
            is_hope_rip = chg_1d >= 6.0 and vs_sma50 < -5

            if is_capitulating_today or is_deeply_oversold:
                state = "DESPAIR ZONE"
                state_class = "badge-despair"
                action_note = "Check 3-tranche entry"
            elif is_hope_rip:
                state = "HOPE SPIKE"
                state_class = "badge-hope"
                action_note = "Do NOT chase — wait for retest"
            elif is_basing_quiet:
                state = "BASING"
                state_class = "badge-base"
                action_note = "Held prior low — watch higher low"
            elif vs_sma50 < -5:
                state = "BLEEDING"
                state_class = "badge-wait"
                action_note = "Still has hope — keep waiting"
            else:
                state = "COMPLACENT"
                state_class = "badge-hope"
                action_note = "No fear — stay patient"

        spark_color = "#10b981" if chg_1d >= 0 else "#ef4444"
        rows.append({
            "ticker": t,
            "name": item["name"],
            "group": item["group"],
            "last": last,
            "chg_1d": chg_1d,
            "chg_5d": chg_5d,
            "hi_4m": hi_4m,
            "hi_date": hi_date,
            "dd_hi": dd_hi,
            "lo_20d": lo_20d,
            "lo_date": lo_date,
            "off_lo": off_lo,
            "vs_sma50": vs_sma50,
            "rsi14": rsi14,
            "vol_ratio": vol_ratio,
            "state": state,
            "state_class": state_class,
            "action_note": action_note,
            "spark_svg": build_sparkline_svg([float(x) for x in s.tail(25).tolist()], spark_color),
        })

    # Compute Market-Wide "Despair Index" (0 = Max Hope/Complacency, 100 = Total Loss of Hope / Capitulation)
    eq_rows = [r for r in rows if r["group"] != "Macro Hedge"]
    pct_below_50d = sum(1 for r in eq_rows if r["vs_sma50"] < 0) / max(len(eq_rows), 1) * 100
    pct_oversold = sum(1 for r in eq_rows if r["rsi14"] < 35) / max(len(eq_rows), 1) * 100
    avg_rsi = float(np.mean([r["rsi14"] for r in eq_rows])) if eq_rows else 50.0
    avg_dd = float(np.mean([r["dd_hi"] for r in eq_rows])) if eq_rows else 0.0
    heavy_red_vol = sum(1 for r in eq_rows if r["chg_1d"] <= -3.0 and r["vol_ratio"] >= 1.5)

    # Sub-scores (0-100 each)
    # 1. VIX Fear Score: VIX 15 -> 0, VIX 35 -> 100
    s_vix = clamp((vix_last - 15.0) / 20.0 * 100.0)
    # 2. VIX Term Structure Inversion: VIX/VIX3M 0.90 -> 0, 1.15 -> 100
    s_term = clamp(((vix_ratio or 0.95) - 0.90) / 0.25 * 100.0)
    # 3. Breadth Breakdown (% below 50d SMA)
    s_breadth = clamp(pct_below_50d)
    # 4. Momentum Exhaustion (Avg RSI: 55 -> 0, 25 -> 100)
    s_rsi = clamp((55.0 - avg_rsi) / 30.0 * 100.0)
    # 5. Active Flush Penalty/Bonus: if market just ripped green today (SPY > +1.5%), hope just returned -> dampens despair!
    spy_row = next((r for r in rows if r["ticker"] == "SPY"), None)
    qqq_row = next((r for r in rows if r["ticker"] == "QQQ"), None)
    spy_1d = spy_row["chg_1d"] if spy_row else 0.0
    hope_bounce_penalty = 18.0 if spy_1d >= 2.0 else (10.0 if spy_1d >= 1.0 else 0.0)

    despair_score = clamp(
        0.30 * s_vix
        + 0.20 * s_term
        + 0.25 * s_breadth
        + 0.25 * s_rsi
        - hope_bounce_penalty
    )

    # Regime verdict aligned with user's exact rule:
    # "Once people lose all hope -> look for buying opportunities; if there's an ounce of hope -> keep waiting."
    if despair_score >= 78 and spy_1d <= 0.5:
        verdict = "TOTAL DESPAIR — HUNT ENTRIES"
        verdict_sub = "VIX panic + washed-out breadth + no relief bounce. Start 3-tranche scaling on quality names holding prior lows."
        verdict_tone = "verdict-buy"
    elif despair_score >= 58:
        verdict = "STRESSED — STILL AN OUNCE OF HOPE (WAIT)"
        verdict_sub = (
            "Damage is real, but either VIX hasn't blown out (>35) or a relief rally just injected fresh hope. "
            "Do NOT chase green days — wait for either a true flush or a quiet higher-low retest."
        )
        verdict_tone = "verdict-wait"
    else:
        verdict = "HOPE INTACT — KEEP WAITING"
        verdict_sub = "Crowd still expects a quick rebound. Protect cash, ignore YouTube/X noise, and let the tape come to you."
        verdict_tone = "verdict-hope"

    # Historical comparison: how did Friday Nov 20 (or worst day in last 10d) look vs Today?
    vix_hist_dates = [d.strftime("%b %d") for d in vix_s.tail(15).index]
    vix_hist_vals = [round(float(x), 2) for x in vix_s.tail(15).tolist()]

    return {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "latest_date": latest_date_str,
        "despair_score": round(despair_score, 1),
        "verdict": verdict,
        "verdict_sub": verdict_sub,
        "verdict_tone": verdict_tone,
        "vix_last": vix_last,
        "vix_prev": vix_prev,
        "vix_chg": (vix_last / vix_prev - 1) * 100 if vix_prev else 0.0,
        "vix_peak_20d": vix_peak_20d,
        "vix_peak_date": vix_peak_date,
        "vix3m_last": vix3m_last,
        "vix_ratio": vix_ratio,
        "pct_below_50d": pct_below_50d,
        "pct_oversold": pct_oversold,
        "avg_rsi": avg_rsi,
        "avg_dd": avg_dd,
        "heavy_red_vol": heavy_red_vol,
        "spy_1d": spy_1d,
        "qqq_1d": qqq_row["chg_1d"] if qqq_row else 0.0,
        "hope_bounce_penalty": hope_bounce_penalty,
        "pillars": [
            {"label": "VIX Panic Level", "value": f"{vix_last:.2f}", "target": "≥ 35.0 = Panic", "score": round(s_vix, 1)},
            {
                "label": "VIX / VIX3M Inversion",
                "value": f"{vix_ratio:.2f}x" if vix_ratio else "N/A",
                "target": "> 1.05x = Acute Fear",
                "score": round(s_term, 1),
            },
            {"label": "Universe Below 50d SMA", "value": f"{pct_below_50d:.0f}%", "target": "≥ 85% = Washed Out", "score": round(s_breadth, 1)},
            {"label": "Avg RSI(14) Exhaustion", "value": f"{avg_rsi:.1f}", "target": "≤ 30 = Capitulation", "score": round(s_rsi, 1)},
        ],
        "rows": rows,
        "vix_hist_dates": vix_hist_dates,
        "vix_hist_vals": vix_hist_vals,
    }


def fmt_pct(val: float) -> str:
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def render_html(d: dict) -> str:
    score = d["despair_score"]
    # Build table rows
    trs = []
    for r in d["rows"]:
        c1d = "pos" if r["chg_1d"] > 0 else ("neg" if r["chg_1d"] < 0 else "flat")
        c5d = "pos" if r["chg_5d"] > 0 else ("neg" if r["chg_5d"] < 0 else "flat")
        c50 = "pos" if r["vs_sma50"] > 0 else "neg"
        vol_flag = " vol-hot" if r["vol_ratio"] >= 1.8 else ""
        trs.append(
            f'<tr data-group="{html.escape(r["group"])}" data-state="{html.escape(r["state"])}">'
            f'<th scope="row">'
            f'<div class="tk-cell"><span class="tk">{html.escape(r["ticker"])}</span>'
            f'<span class="tk-name">{html.escape(r["name"])}</span></div>'
            f"</th>"
            f'<td><span class="grp">{html.escape(r["group"])}</span></td>'
            f'<td class="num">${r["last"]:.2f}</td>'
            f'<td class="num {c1d}">{fmt_pct(r["chg_1d"])}</td>'
            f'<td class="num {c5d}">{fmt_pct(r["chg_5d"])}</td>'
            f'<td class="num neg">{r["dd_hi"]:.1f}% <span class="sub">({html.escape(r["hi_date"])})</span></td>'
            f'<td class="num">{r["off_lo"]:+.1f}% <span class="sub">({html.escape(r["lo_date"])})</span></td>'
            f'<td class="num {c50}">{r["vs_sma50"]:+.1f}%</td>'
            f'<td class="num">{r["rsi14"]:.1f}</td>'
            f'<td class="num{vol_flag}">{r["vol_ratio"]:.1f}x</td>'
            f'<td class="spark-td">{r["spark_svg"]}</td>'
            f'<td><span class="badge {r["state_class"]}">{html.escape(r["state"])}</span>'
            f'<div class="act-note">{html.escape(r["action_note"])}</div></td>'
            f"</tr>"
        )
    table_body = "\n".join(trs)

    # Pillars HTML
    pillar_cards = []
    for p in d["pillars"]:
        bar_w = max(4, min(100, p["score"]))
        pillar_cards.append(
            f'<div class="pillar">'
            f'<div class="pillar-top"><span class="pillar-label">{html.escape(p["label"])}</span>'
            f'<span class="pillar-val">{html.escape(p["value"])}</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="inline-size:{bar_w}%"></div></div>'
            f'<div class="pillar-sub">{html.escape(p["target"])} · Score {p["score"]:.0f}/100</div>'
            f"</div>"
        )
    pillars_html = "\n".join(pillar_cards)

    # VIX 15-day bar chart SVG
    v_vals = d["vix_hist_vals"]
    v_dates = d["vix_hist_dates"]
    v_max = max(max(v_vals, default=40), 40)
    v_bars = []
    bw = 26
    gap = 8
    svg_w = len(v_vals) * (bw + gap) + 20
    svg_h = 110
    # 35 panic line y
    y35 = svg_h - 24 - (35 / v_max) * (svg_h - 38)
    for i, (dt_lbl, val) in enumerate(zip(v_dates, v_vals)):
        bh = (val / v_max) * (svg_h - 38)
        x = 10 + i * (bw + gap)
        y = svg_h - 24 - bh
        fill = "#10b981" if val >= 35 else ("#f59e0b" if val >= 25 else "#64748b")
        v_bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{bh:.1f}" rx="4" fill="{fill}" opacity="0.9"/>'
            f'<text x="{x + bw/2:.1f}" y="{y - 4:.1f}" text-anchor="middle" class="svg-num">{val:.0f}</text>'
            f'<text x="{x + bw/2:.1f}" y="{svg_h - 8}" text-anchor="middle" class="svg-lbl">{html.escape(dt_lbl[-2:])}</text>'
        )
    vix_chart_svg = (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" class="vix-svg" role="img" aria-label="15-day VIX panic tracker">'
        f'<line x1="6" y1="{y35:.1f}" x2="{svg_w - 6}" y2="{y35:.1f}" stroke="#10b981" stroke-dasharray="4 4" stroke-width="1"/>'
        f'<text x="{svg_w - 8}" y="{y35 - 4:.1f}" text-anchor="end" class="svg-thresh">VIX 35 = Capitulation</text>'
        + "".join(v_bars)
        + "</svg>"
    )

    hope_penalty_html = (
        f'<div class="penalty-pill">Hope-Bounce Dampener Active: −{d["hope_bounce_penalty"]:.0f} pts '
        f'(SPY {fmt_pct(d["spy_1d"])} today — relief rallies restore hope; wait for retest)</div>'
        if d["hope_bounce_penalty"] > 0
        else '<div class="penalty-pill neutral">No relief-rally dampener today (SPY not spiking)</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="color-scheme" content="dark light" />
  <title>Market Capitulation Cockpit — {html.escape(d["latest_date"])}</title>
  <style>
    :root {{
      color-scheme: dark light;
      --bg-light: #f8fafc;
      --bg-dark: #090b10;
      --surface-light: #ffffff;
      --surface-dark: #11151f;
      --surface2-light: #f1f5f9;
      --surface2-dark: #181e2c;
      --border-light: #e2e8f0;
      --border-dark: #252e42;
      --text-light: #0f172a;
      --text-dark: #e6edf7;
      --muted-light: #64748b;
      --muted-dark: #8b9bb4;
      --accent-light: #059669;
      --accent-dark: #10b981;

      --bg: var(--bg-dark);
      --surface: var(--surface-dark);
      --surface2: var(--surface2-dark);
      --border: var(--border-dark);
      --text: var(--text-dark);
      --muted: var(--muted-dark);
      --accent: var(--accent-dark);
      --scrollbar-thumb: #334155;
      --scrollbar-track: #0f172a;
      scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
      accent-color: var(--accent);
    }}

    @media (prefers-color-scheme: light) {{
      :root {{
        --bg: var(--bg-light);
        --surface: var(--surface-light);
        --surface2: var(--surface2-light);
        --border: var(--border-light);
        --text: var(--text-light);
        --muted: var(--muted-light);
        --accent: var(--accent-light);
        --scrollbar-thumb: #94a3b8;
        --scrollbar-track: #e2e8f0;
      }}
    }}

    @supports (color: light-dark(white, black)) {{
      :root {{
        --bg: light-dark(var(--bg-light), var(--bg-dark));
        --surface: light-dark(var(--surface-light), var(--surface-dark));
        --surface2: light-dark(var(--surface2-light), var(--surface2-dark));
        --border: light-dark(var(--border-light), var(--border-dark));
        --text: light-dark(var(--text-light), var(--text-dark));
        --muted: light-dark(var(--muted-light), var(--muted-dark));
        --accent: light-dark(var(--accent-light), var(--accent-dark));
      }}
    }}

    @supports not (scrollbar-color: auto) {{
      .table-wrapper {{ scrollbar-width: auto; }}
      .table-wrapper::-webkit-scrollbar {{ width: 10px; height: 10px; }}
      .table-wrapper::-webkit-scrollbar-thumb {{ background: var(--scrollbar-thumb); border-radius: 6px; }}
      .table-wrapper::-webkit-scrollbar-track {{ background: var(--scrollbar-track); }}
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
      -webkit-font-smoothing: antialiased;
    }}

    .wrap {{
      max-inline-size: 1380px;
      margin-inline: auto;
      padding: 24px 20px 56px;
    }}

    header.top {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-end;
      gap: 14px;
      margin-block-end: 20px;
      padding-block-end: 16px;
      border-block-end: 1px solid var(--border);
    }}
    .brand h1 {{
      margin: 0;
      font-size: 1.45rem;
      letter-spacing: -0.02em;
    }}
    .brand p {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .meta {{
      font-size: 0.8rem;
      color: var(--muted);
      text-align: end;
    }}
    .meta code {{
      background: var(--surface2);
      padding: 2px 7px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 0.78rem;
    }}

    /* Verdict Banner */
    .hero {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 16px;
      margin-block-end: 20px;
    }}
    @media (max-width: 920px) {{
      .hero {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
    }}
    .verdict-card {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 14px;
      border-inline-start-width: 5px;
    }}
    .verdict-buy {{ border-inline-start-color: #10b981; }}
    .verdict-wait {{ border-inline-start-color: #f59e0b; }}
    .verdict-hope {{ border-inline-start-color: #ef4444; }}

    .score-row {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .verdict-tag {{
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .verdict-title {{
      font-size: 1.35rem;
      font-weight: 800;
      margin: 4px 0 0;
      letter-spacing: -0.015em;
    }}
    .verdict-buy .verdict-title {{ color: #10b981; }}
    .verdict-wait .verdict-title {{ color: #f59e0b; }}
    .verdict-hope .verdict-title {{ color: #ef4444; }}

    .big-score {{
      font-size: 2.5rem;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      line-height: 1;
    }}
    .big-score span {{
      font-size: 0.9rem;
      font-weight: 500;
      color: var(--muted);
    }}
    .verdict-sub {{
      margin: 0;
      font-size: 0.92rem;
      color: var(--muted);
    }}

    /* Gauge */
    .gauge-wrap {{ margin-block: 4px; }}
    .gauge-track {{
      position: relative;
      block-size: 12px;
      border-radius: 999px;
      background: linear-gradient(90deg, #ef4444 0%, #f59e0b 55%, #10b981 100%);
      opacity: 0.9;
    }}
    .gauge-pin {{
      position: absolute;
      inset-block-start: -4px;
      inline-size: 6px;
      block-size: 20px;
      background: var(--text);
      border-radius: 3px;
      transform: translateX(-50%);
      box-shadow: 0 0 0 2px var(--surface);
    }}
    .gauge-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      color: var(--muted);
      margin-block-start: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .penalty-pill {{
      font-size: 0.78rem;
      padding: 8px 12px;
      border-radius: 10px;
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.35);
      color: #f59e0b;
    }}
    .penalty-pill.neutral {{
      background: var(--surface2);
      border-color: var(--border);
      color: var(--muted);
    }}

    /* Pillars */
    .pillars {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }}
    @media (max-width: 520px) {{
      .pillars {{ grid-template-columns: 1fr; }}
    }}
    .pillar {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .pillar-top {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
    }}
    .pillar-label {{ font-size: 0.78rem; color: var(--muted); font-weight: 600; }}
    .pillar-val {{ font-size: 1.15rem; font-weight: 800; font-variant-numeric: tabular-nums; }}
    .bar-track {{
      block-size: 6px;
      background: var(--border);
      border-radius: 999px;
      overflow: hidden;
      margin-block: 8px 6px;
    }}
    .bar-fill {{
      block-size: 100%;
      background: var(--accent);
      border-radius: 999px;
    }}
    .pillar-sub {{ font-size: 0.72rem; color: var(--muted); }}

    /* VIX chart + Anti-Distraction Protocol */
    .mid-grid {{
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 16px;
      margin-block-end: 20px;
    }}
    @media (max-width: 920px) {{
      .mid-grid {{ grid-template-columns: 1fr; }}
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 0.92rem;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
    }}
    .vix-svg {{ inline-size: 100%; block-size: auto; display: block; }}
    .svg-num {{ font-size: 9px; fill: var(--text); font-weight: 700; }}
    .svg-lbl {{ font-size: 9px; fill: var(--muted); }}
    .svg-thresh {{ font-size: 9px; fill: #10b981; font-weight: 700; }}

    .rules-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 9px;
    }}
    .rules-list li {{
      font-size: 0.84rem;
      padding: 9px 12px;
      border-radius: 10px;
      background: var(--surface2);
      border: 1px solid var(--border);
    }}
    .rules-list strong {{ color: var(--text); }}

    /* Filter Toolbar */
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-block-end: 12px;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .fbtn {{
      background: var(--surface2);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .fbtn.active {{
      background: var(--accent);
      color: #ffffff;
      border-color: var(--accent);
    }}

    /* Responsive Table (modern-web-guidance: sticky headers + @container stacked cards) */
    .table-wrapper {{
      overflow: auto;
      max-inline-size: 100%;
      max-block-size: min(680px, 78vh);
      container-type: inline-size;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--surface);
    }}
    table {{
      inline-size: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.85rem;
      font-variant-numeric: tabular-nums;
    }}
    caption {{
      text-align: start;
      padding: 12px 16px;
      font-size: 0.78rem;
      color: var(--muted);
      border-block-end: 1px solid var(--border);
    }}
    thead {{
      position: sticky;
      inset-block-start: 0;
      z-index: 2;
      background: var(--surface2);
    }}
    thead th {{
      text-align: start;
      padding: 10px 12px;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      border-block-end: 1px solid var(--border);
      white-space: nowrap;
    }}
    thead th.num {{ text-align: end; }}
    tbody td, tbody th {{
      padding: 11px 12px;
      border-block-end: 1px solid var(--border);
      vertical-align: middle;
      white-space: nowrap;
    }}
    tbody tr:last-child td, tbody tr:last-child th {{ border-block-end: none; }}
    tbody th {{
      position: sticky;
      inset-inline-start: 0;
      z-index: 1;
      background: var(--surface);
      text-align: start;
    }}
    tbody tr:hover td, tbody tr:hover th {{ background: var(--surface2); }}

    .tk-cell {{ display: flex; flex-direction: column; }}
    .tk {{ font-weight: 800; font-size: 0.92rem; letter-spacing: 0.02em; }}
    .tk-name {{ font-size: 0.72rem; color: var(--muted); }}
    .grp {{
      font-size: 0.72rem;
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--muted);
    }}
    td.num {{ text-align: end; }}
    .pos {{ color: #10b981; font-weight: 600; }}
    .neg {{ color: #ef4444; font-weight: 600; }}
    .flat {{ color: var(--muted); }}
    .sub {{ font-size: 0.7rem; color: var(--muted); font-weight: 400; }}
    .vol-hot {{ color: #f59e0b; font-weight: 800; }}

    .badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.68rem;
      font-weight: 800;
      letter-spacing: 0.07em;
    }}
    .badge-despair {{ background: rgba(16,185,129,0.16); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }}
    .badge-base {{ background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.35); }}
    .badge-wait {{ background: rgba(245,158,11,0.14); color: #f59e0b; border: 1px solid rgba(245,158,11,0.35); }}
    .badge-hope {{ background: rgba(239,68,68,0.14); color: #ef4444; border: 1px solid rgba(239,68,68,0.35); }}
    .badge-neutral {{ background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }}
    .act-note {{ font-size: 0.72rem; color: var(--muted); margin-block-start: 3px; }}
    .spark {{ display: block; }}

    @container (width < 760px) {{
      table, thead, tbody, tr, th, td {{ display: block; }}
      thead {{ display: none; }}
      tr {{
        margin: 10px;
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
      }}
      tbody th {{
        position: sticky;
        inset-block-start: 0;
        background: var(--surface2);
        border-block-end: 1px solid var(--border);
      }}
      tbody td {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        white-space: normal;
      }}
      table {{
        --l2: "Bucket";
        --l3: "Last";
        --l4: "1D %";
        --l5: "5D %";
        --l6: "vs 4m High";
        --l7: "Off 20d Low";
        --l8: "vs 50d SMA";
        --l9: "RSI(14)";
        --l10: "Vol vs 20d";
        --l11: "25d Trend";
        --l12: "Regime";
      }}
      td:nth-child(2)::before {{ content: var(--l2); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(3)::before {{ content: var(--l3); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(4)::before {{ content: var(--l4); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(5)::before {{ content: var(--l5); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(6)::before {{ content: var(--l6); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(7)::before {{ content: var(--l7); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(8)::before {{ content: var(--l8); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(9)::before {{ content: var(--l9); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(10)::before {{ content: var(--l10); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(11)::before {{ content: var(--l11); color: var(--muted); font-size: 0.75rem; }}
      td:nth-child(12)::before {{ content: var(--l12); color: var(--muted); font-size: 0.75rem; }}
    }}

    footer {{
      margin-block-start: 20px;
      font-size: 0.76rem;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="top">
      <div class="brand">
        <h1>Market Capitulation Cockpit</h1>
        <p>Zero news. Zero YouTube. Zero memes. Buy only when hope is gone — wait while any ounce of hope remains.</p>
      </div>
      <div class="meta">
        <div>Market Data Through: <strong>{html.escape(d["latest_date"])}</strong></div>
        <div>Auto-updated Mon–Fri at <strong>12:00 PM PST</strong> (Cloud + Mobile Ready)</div>
      </div>
    </header>

    <section class="hero">
      <div class="card verdict-card {d["verdict_tone"]}">
        <div>
          <div class="verdict-tag">Current Regime Signal</div>
          <div class="score-row">
            <div class="verdict-title">{html.escape(d["verdict"])}</div>
            <div class="big-score">{score:.0f}<span>/100</span></div>
          </div>
        </div>
        <div class="gauge-wrap">
          <div class="gauge-track">
            <div class="gauge-pin" style="inset-inline-start:{score}%"></div>
          </div>
          <div class="gauge-labels">
            <span>0 · Max Hope (Wait)</span>
            <span>50 · Nervous</span>
            <span>78+ · Total Despair (Hunt)</span>
          </div>
        </div>
        <p class="verdict-sub">{html.escape(d["verdict_sub"])}</p>
        {hope_penalty_html}
      </div>

      <div class="card">
        <h2>4 Objective Despair Pillars</h2>
        <div class="pillars">
          {pillars_html}
        </div>
      </div>
    </section>

    <section class="mid-grid">
      <div class="card">
        <h2>15-Day VIX Fear Tape (20d Peak: {d["vix_peak_20d"]:.2f} on {html.escape(d["vix_peak_date"])})</h2>
        {vix_chart_svg}
      </div>
      <div class="card">
        <h2>Anti-Distraction Cockpit Rules</h2>
        <ul class="rules-list">
          <li><strong>Rule 1 — No Hope Buying:</strong> If Despair Score &lt; 78 or SPY just ripped green (+2%+), keep cash idle. Relief rallies keep hope alive.</li>
          <li><strong>Rule 2 — Peer Divergence Tell:</strong> Look for leaders flushing to new lows while peers (AVGO, AMD, VST) hold their prior lows.</li>
          <li><strong>Rule 3 — 3-Tranche Sizing:</strong> 33% at VIX ≥ 35 flush, 33% if it bleeds another −10%, 33% on higher-low confirmation. Never weeklies.</li>
          <li><strong>Rule 4 — Feed Blackout:</strong> Close X, YouTube, and 1-minute charts. Re-run this HTML once at noon and once after 4:05 PM ET.</li>
        </ul>
      </div>
    </section>

    <section>
      <div class="toolbar">
        <div class="filters" role="group" aria-label="Filter universe by bucket">
          <button class="fbtn active" data-filter="ALL">All ({len(d["rows"])})</button>
          <button class="fbtn" data-filter="Broad Market">Broad Market</button>
          <button class="fbtn" data-filter="AI Core">AI Core</button>
          <button class="fbtn" data-filter="AI Power">AI Power</button>
          <button class="fbtn" data-filter="AI Infra">AI Infra</button>
          <button class="fbtn" data-filter="DESPAIR">Despair / Basing Only</button>
        </div>
      </div>

      <div tabindex="0" role="region" aria-labelledby="market-table-caption" class="table-wrapper">
        <table>
          <caption id="market-table-caption">
            Broad Market &amp; AI Rotation Scorecard — {html.escape(d["latest_date"])} (Generated {html.escape(d["generated_at"])})
          </caption>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Bucket</th>
              <th class="num">Last</th>
              <th class="num">1D %</th>
              <th class="num">5D %</th>
              <th class="num">vs 4m High</th>
              <th class="num">Off 20d Low</th>
              <th class="num">vs 50d SMA</th>
              <th class="num">RSI(14)</th>
              <th class="num">Vol vs 20d</th>
              <th>25d Trend</th>
              <th>Hope / Despair State</th>
            </tr>
          </thead>
          <tbody id="scorecard-body">
            {table_body}
          </tbody>
        </table>
      </div>
    </section>

    <footer>
      <span>Source: yfinance daily OHLCV only (no news, no social sentiment feeds).</span>
      <span>SPY 1D: {fmt_pct(d["spy_1d"])} · QQQ 1D: {fmt_pct(d["qqq_1d"])} · VIX: {d["vix_last"]:.2f} ({fmt_pct(d["vix_chg"])})</span>
    </footer>
  </div>

  <script>
    const buttons = document.querySelectorAll('.fbtn');
    const rows = document.querySelectorAll('#scorecard-body tr');
    buttons.forEach((btn) => {{
      btn.addEventListener('click', () => {{
        buttons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const f = btn.dataset.filter;
        rows.forEach((tr) => {{
          if (f === 'ALL') {{
            tr.style.display = '';
          }} else if (f === 'DESPAIR') {{
            const st = tr.dataset.state || '';
            tr.style.display = (st === 'DESPAIR ZONE' || st === 'BASING') ? '' : 'none';
          }} else {{
            tr.style.display = tr.dataset.group === f ? '' : 'none';
          }}
        }});
      }});
    }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate local Market Capitulation Cockpit HTML")
    parser.add_argument("--open", action="store_true", help="Open generated HTML in default browser")
    args = parser.parse_args()

    data = fetch_and_score()
    html_out = render_html(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html_out, encoding="utf-8")
    print(f"✅ Generated {OUTPUT_PATH}")
    print(f"   Date: {data['latest_date']} | Despair Score: {data['despair_score']}/100 | Verdict: {data['verdict']}")
    print(f"   VIX: {data['vix_last']:.2f} (20d Peak: {data['vix_peak_20d']:.2f} on {data['vix_peak_date']}) | Below 50d SMA: {data['pct_below_50d']:.0f}%")

    if args.open:
        subprocess.run(["open", str(OUTPUT_PATH)], check=False)


if __name__ == "__main__":
    main()
