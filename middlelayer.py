"""
MiddleLayer — Ops Intelligence for Fintech
==========================================
Automated PnL calculation, anomaly detection,
and AI-powered explanations via Claude API.

Usage:
    python middlelayer.py --input trades.csv --output report.html
    python middlelayer.py --input trades.csv --slack-webhook <url>
"""

import csv
import json
import os
import sys
import argparse
from datetime import datetime
from typing import Optional
import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────

FX_RATES_TO_EUR = {
    "EUR": 1.0,
    "USD": 0.92,
    "GBP": 1.17,
    "SEK": 0.088,
    "NOK": 0.086,
    "DKK": 0.134,
    "CHF": 1.05,
    "PLN": 0.23,
}

ANOMALY_RULES = {
    "large_loss_eur":       -5000,   # Single trade loss > €5,000
    "large_gain_eur":        8000,   # Single trade gain > €8,000
    "price_deviation_pct":    5.0,   # Entry vs market price deviation > 5%
    "late_hour_threshold":      22,  # Trades after 22:00
    "early_hour_threshold":      6,  # Trades before 06:00
    "consecutive_failures":      2,  # Failed trades on same instrument
    "fx_spike_pct":              4.0,# FX rate move > 4% vs day average
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_trades(filepath: str) -> list[dict]:
    """Load and parse trade CSV file."""
    trades = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                "id":           row["trade_id"],
                "timestamp":    row["timestamp"],
                "instrument":   row["instrument"],
                "type":         row["trade_type"],        # BUY / SELL
                "currency":     row["currency"],
                "quantity":     float(row["quantity"]),
                "entry_price":  float(row["entry_price"]),
                "market_price": float(row["market_price"]),
                "status":       row["status"],            # SETTLED / FAILED / PENDING
                "desk":         row["desk"],
                "counterparty": row["counterparty"],
            })
    return trades

# ── PnL Engine ────────────────────────────────────────────────────────────────

def calculate_pnl(trade: dict) -> dict:
    """Calculate mark-to-market PnL for a single trade."""
    qty   = trade["quantity"]
    entry = trade["entry_price"]
    mkt   = trade["market_price"]
    fx    = FX_RATES_TO_EUR.get(trade["currency"], 1.0)

    if trade["type"] == "BUY":
        raw_pnl = (mkt - entry) * qty
    else:
        raw_pnl = (entry - mkt) * qty

    pnl_eur = raw_pnl * fx
    price_deviation_pct = abs((mkt - entry) / entry * 100) if entry != 0 else 0

    return {
        **trade,
        "pnl_local":            round(raw_pnl, 2),
        "pnl_eur":              round(pnl_eur, 2),
        "price_deviation_pct":  round(price_deviation_pct, 2),
        "fx_rate":              fx,
    }

def run_pnl_engine(trades: list[dict]) -> list[dict]:
    """Run PnL calculation across all trades."""
    results = []
    for t in trades:
        if t["status"] == "SETTLED":
            results.append(calculate_pnl(t))
        else:
            results.append({**t, "pnl_local": 0, "pnl_eur": 0,
                             "price_deviation_pct": 0, "fx_rate": 0})
    return results

# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(trades: list[dict]) -> list[dict]:
    """Flag trades that breach anomaly rules."""
    flagged = []

    # Track consecutive failures per instrument
    failure_counts: dict[str, int] = {}

    for t in trades:
        flags = []

        # Rule 1: Large loss
        if t["pnl_eur"] < ANOMALY_RULES["large_loss_eur"]:
            flags.append(f"LARGE_LOSS: €{t['pnl_eur']:,.0f}")

        # Rule 2: Large gain (unusual, may indicate data error)
        if t["pnl_eur"] > ANOMALY_RULES["large_gain_eur"]:
            flags.append(f"LARGE_GAIN: €{t['pnl_eur']:,.0f}")

        # Rule 3: Price deviation
        if t["price_deviation_pct"] > ANOMALY_RULES["price_deviation_pct"]:
            flags.append(f"PRICE_DEVIATION: {t['price_deviation_pct']:.1f}%")

        # Rule 4: Off-hours trading
        try:
            hour = int(t["timestamp"].split("T")[1].split(":")[0])
            if hour >= ANOMALY_RULES["late_hour_threshold"] or \
               hour < ANOMALY_RULES["early_hour_threshold"]:
                flags.append(f"OFF_HOURS: {hour:02d}:00")
        except (IndexError, ValueError):
            pass

        # Rule 5: Failed trade tracking
        if t["status"] == "FAILED":
            key = t["instrument"]
            failure_counts[key] = failure_counts.get(key, 0) + 1
            if failure_counts[key] >= ANOMALY_RULES["consecutive_failures"]:
                flags.append(f"CONSECUTIVE_FAILURES: {failure_counts[key]}x on {key}")

        if flags:
            flagged.append({**t, "flags": flags, "flag_count": len(flags)})

    return sorted(flagged, key=lambda x: x["flag_count"], reverse=True)

# ── Claude API — AI Explanations ──────────────────────────────────────────────

def get_ai_explanation(flagged_trades: list[dict], summary: dict) -> str:
    """
    Send flagged trades to Claude API.
    Returns plain-English analysis for ops team.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️  ANTHROPIC_API_KEY not set. Skipping AI analysis."

    client = anthropic.Anthropic(api_key=api_key)

    trades_summary = json.dumps([{
        "id":          t["id"],
        "instrument":  t["instrument"],
        "type":        t["type"],
        "currency":    t["currency"],
        "pnl_eur":     t["pnl_eur"],
        "deviation":   t["price_deviation_pct"],
        "flags":       t["flags"],
        "status":      t["status"],
        "desk":        t["desk"],
        "counterparty":t["counterparty"],
        "timestamp":   t["timestamp"],
    } for t in flagged_trades[:10]], indent=2)

    prompt = f"""You are a senior Middle Office analyst at a fintech company.
Review the following flagged trades and provide a concise daily operations briefing.

PORTFOLIO SUMMARY:
- Total trades: {summary['total_trades']}
- Settled: {summary['settled']}
- Failed: {summary['failed']}
- Total PnL: €{summary['total_pnl_eur']:,.2f}
- Flagged trades: {summary['flagged_count']}

FLAGGED TRADES (top 10):
{trades_summary}

Write a structured briefing with:
1. EXECUTIVE SUMMARY (2-3 sentences, key risk points)
2. TOP CONCERNS (bullet list, max 5, each with trade ID and specific action)
3. DESK BREAKDOWN (which desk needs attention and why)
4. RECOMMENDED ACTIONS (concrete next steps for the ops team)

Keep language clear and actionable. Avoid jargon. This will be read by a CFO."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

# ── Summary Statistics ─────────────────────────────────────────────────────────

def build_summary(trades: list[dict], flagged: list[dict]) -> dict:
    """Build portfolio-level summary statistics."""
    settled = [t for t in trades if t["status"] == "SETTLED"]
    failed  = [t for t in trades if t["status"] == "FAILED"]

    total_pnl = sum(t["pnl_eur"] for t in settled)

    by_desk = {}
    for t in settled:
        desk = t["desk"]
        by_desk[desk] = by_desk.get(desk, 0) + t["pnl_eur"]

    by_currency = {}
    for t in settled:
        ccy = t["currency"]
        by_currency[ccy] = by_currency.get(ccy, 0) + t["pnl_eur"]

    return {
        "total_trades":    len(trades),
        "settled":         len(settled),
        "failed":          len(failed),
        "pending":         len([t for t in trades if t["status"] == "PENDING"]),
        "total_pnl_eur":   round(total_pnl, 2),
        "flagged_count":   len(flagged),
        "by_desk":         {k: round(v, 2) for k, v in by_desk.items()},
        "by_currency":     {k: round(v, 2) for k, v in by_currency.items()},
        "best_trade":      max(settled, key=lambda x: x["pnl_eur"], default=None),
        "worst_trade":     min(settled, key=lambda x: x["pnl_eur"], default=None),
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ── HTML Dashboard ────────────────────────────────────────────────────────────

def generate_html_report(trades: list[dict], flagged: list[dict],
                          summary: dict, ai_analysis: str) -> str:
    """Generate a clean HTML dashboard report."""

    pnl_color = "#166534" if summary["total_pnl_eur"] >= 0 else "#991B1B"
    pnl_sign  = "+" if summary["total_pnl_eur"] >= 0 else ""

    desk_rows = "".join(
        f'<tr><td>{desk}</td><td style="color:{"#166534" if pnl>=0 else "#991B1B"};font-weight:500">'
        f'{"+" if pnl>=0 else ""}€{pnl:,.2f}</td></tr>'
        for desk, pnl in summary["by_desk"].items()
    )

    flagged_rows = ""
    for t in flagged[:15]:
        severity = "high" if t["flag_count"] >= 2 else "medium"
        color    = "#FEE2E2" if severity == "high" else "#FEF3C7"
        badge    = "#991B1B" if severity == "high" else "#92400E"
        flags_str = " · ".join(t["flags"])
        pnl_str   = f'{"+" if t["pnl_eur"]>=0 else ""}€{t["pnl_eur"]:,.2f}'
        flagged_rows += f"""
        <tr style="background:{color}">
          <td><strong>{t['id']}</strong></td>
          <td>{t['instrument']}</td>
          <td>{t['type']}</td>
          <td style="color:{pnl_color};font-weight:500">{pnl_str}</td>
          <td><span style="background:{badge};color:white;padding:2px 8px;
              border-radius:4px;font-size:11px">{severity.upper()}</span></td>
          <td style="font-size:12px;color:#374151">{flags_str}</td>
        </tr>"""

    ai_html = ai_analysis.replace("\n", "<br>") if ai_analysis else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MiddleLayer — Daily PnL Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #F8FAFC; color: #1E293B; font-size: 14px; }}
    .header {{ background: #1B3A6B; color: white; padding: 24px 32px;
               display: flex; justify-content: space-between; align-items: center; }}
    .header h1 {{ font-size: 22px; font-weight: 600; letter-spacing: -0.3px; }}
    .header .meta {{ font-size: 12px; opacity: 0.75; text-align: right; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 32px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 24px; }}
    .metric {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px;
               padding: 16px; text-align: center; }}
    .metric .value {{ font-size: 26px; font-weight: 600; margin-bottom: 4px; }}
    .metric .label {{ font-size: 11px; color: #64748B; text-transform: uppercase;
                      letter-spacing: 0.05em; }}
    .card {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px;
             padding: 20px; margin-bottom: 20px; }}
    .card h2 {{ font-size: 14px; font-weight: 600; color: #1E293B; margin-bottom: 16px;
                text-transform: uppercase; letter-spacing: 0.05em; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; padding: 8px 12px; background: #F1F5F9;
          font-weight: 500; color: #475569; font-size: 11px;
          text-transform: uppercase; letter-spacing: 0.05em; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #F1F5F9; }}
    tr:last-child td {{ border-bottom: none; }}
    .ai-box {{ background: #EFF6FF; border-left: 4px solid #3B82F6;
               border-radius: 0 8px 8px 0; padding: 16px 20px;
               font-size: 13px; line-height: 1.7; color: #1E3A5F; }}
    .footer {{ text-align: center; padding: 24px; font-size: 11px; color: #94A3B8; }}
    .status-ok {{ color: #166534; font-weight: 500; }}
    .status-fail {{ color: #991B1B; font-weight: 500; }}
  </style>
</head>
<body>

<div class="header">
  <div>
    <h1>MiddleLayer</h1>
    <div style="font-size:13px;opacity:0.85;margin-top:4px">
      Ops Intelligence for Fintech — Daily PnL Report
    </div>
  </div>
  <div class="meta">
    Generated: {summary['generated_at']}<br>
    Trades analysed: {summary['total_trades']}
  </div>
</div>

<div class="container">

  <!-- Metric Cards -->
  <div class="metrics">
    <div class="metric">
      <div class="value" style="color:{pnl_color}">{pnl_sign}€{summary['total_pnl_eur']:,.0f}</div>
      <div class="label">Total PnL (EUR)</div>
    </div>
    <div class="metric">
      <div class="value">{summary['total_trades']}</div>
      <div class="label">Total Trades</div>
    </div>
    <div class="metric">
      <div class="value" style="color:#166534">{summary['settled']}</div>
      <div class="label">Settled</div>
    </div>
    <div class="metric">
      <div class="value" style="color:#991B1B">{summary['failed']}</div>
      <div class="label">Failed</div>
    </div>
    <div class="metric">
      <div class="value" style="color:#92400E">{summary['flagged_count']}</div>
      <div class="label">Flagged</div>
    </div>
  </div>

  <!-- AI Analysis -->
  <div class="card">
    <h2>AI Operations Briefing</h2>
    <div class="ai-box">{ai_html}</div>
  </div>

  <!-- Flagged Trades -->
  <div class="card">
    <h2>Flagged Trades — Requires Attention</h2>
    <table>
      <thead>
        <tr>
          <th>Trade ID</th><th>Instrument</th><th>Type</th>
          <th>PnL (EUR)</th><th>Severity</th><th>Flags</th>
        </tr>
      </thead>
      <tbody>{flagged_rows}</tbody>
    </table>
  </div>

  <!-- Desk Breakdown -->
  <div class="card">
    <h2>PnL by Desk</h2>
    <table>
      <thead><tr><th>Desk</th><th>PnL (EUR)</th></tr></thead>
      <tbody>{desk_rows}</tbody>
    </table>
  </div>

</div>

<div class="footer">
  MiddleLayer v1.0 · Built with Python + Claude API ·
  github.com/onurmertsozer/middlelayer
</div>

</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiddleLayer — Ops Intelligence")
    parser.add_argument("--input",  default="trades.csv",   help="Trade CSV file")
    parser.add_argument("--output", default="report.html",  help="HTML report output")
    parser.add_argument("--json",   action="store_true",    help="Also output JSON")
    args = parser.parse_args()

    print("MiddleLayer — starting analysis...")
    print(f"  Input:  {args.input}")

    trades  = load_trades(args.input)
    trades  = run_pnl_engine(trades)
    flagged = detect_anomalies(trades)
    summary = build_summary(trades, flagged)

    print(f"  Trades loaded:  {summary['total_trades']}")
    print(f"  Settled:        {summary['settled']}")
    print(f"  Failed:         {summary['failed']}")
    print(f"  Flagged:        {summary['flagged_count']}")
    print(f"  Total PnL:      €{summary['total_pnl_eur']:,.2f}")
    print("  Calling Claude API for AI analysis...")

    ai_analysis = get_ai_explanation(flagged, summary)

    html = generate_html_report(trades, flagged, summary, ai_analysis)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved:   {args.output}")

    if args.json:
        with open("report.json", "w") as f:
            json.dump({"summary": summary, "flagged": flagged[:20]}, f, indent=2)
        print("  JSON saved:     report.json")

    print("\nDone. Open report.html in your browser.")

if __name__ == "__main__":
    main()
