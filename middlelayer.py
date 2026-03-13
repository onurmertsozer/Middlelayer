"""
MiddleLayer — Ops Intelligence for Fintech
==========================================
Automated PnL calculation, anomaly detection,
and AI-powered explanations via Claude API.
"""

import yaml
import urllib.request
import csv
import json
import os
import argparse
from datetime import datetime
import anthropic

# ── Configuration & Rates ─────────────────────────────────────────────────────

FX_RATES_TO_EUR = {
    "EUR": 1.0, "USD": 0.92, "GBP": 1.17, "SEK": 0.088,
    "NOK": 0.086, "DKK": 0.134, "CHF": 1.05, "PLN": 0.23,
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    """Loads operational thresholds from a YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ Warning: {config_path} not found. Using default internal rules.")
        return {
            "analysis_rules": {
                "large_loss_threshold": -5000,
                "large_gain_threshold": 8000,
                "price_deviation_pct": 5.0,
                "trading_hours": {"start": 6, "end": 22},
                "consecutive_failures": 2
            }
        }

def load_trades(filepath: str) -> list[dict]:
    trades = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                "id":           row["trade_id"],
                "timestamp":    row["timestamp"],
                "instrument":   row["instrument"],
                "type":         row["trade_type"],
                "currency":     row["currency"],
                "quantity":     float(row["quantity"]),
                "entry_price":  float(row["entry_price"]),
                "market_price": float(row["market_price"]),
                "status":       row["status"],
                "desk":         row["desk"],
                "counterparty": row["counterparty"],
            })
    return trades

# ── PnL Engine ────────────────────────────────────────────────────────────────

def calculate_pnl(trade: dict) -> dict:
    qty, entry, mkt = trade["quantity"], trade["entry_price"], trade["market_price"]
    fx = FX_RATES_TO_EUR.get(trade["currency"], 1.0)

    raw_pnl = (mkt - entry) * qty if trade["type"] == "BUY" else (entry - mkt) * qty
    pnl_eur = raw_pnl * fx
    price_deviation_pct = abs((mkt - entry) / entry * 100) if entry != 0 else 0

    return {
        **trade, "pnl_local": round(raw_pnl, 2), "pnl_eur": round(pnl_eur, 2),
        "price_deviation_pct": round(price_deviation_pct, 2), "fx_rate": fx,
    }

def run_pnl_engine(trades: list[dict]) -> list[dict]:
    return [calculate_pnl(t) if t["status"] == "SETTLED" else {**t, "pnl_local": 0, "pnl_eur": 0, "price_deviation_pct": 0, "fx_rate": 0} for t in trades]

# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(trades: list[dict], config: dict) -> list[dict]:
    rules = config.get("analysis_rules", {})
    flagged = []
    failure_counts = {}

    for t in trades:
        flags = []
        # Threshold checks using config values
        if t["pnl_eur"] < rules.get("large_loss_threshold", -5000): 
            flags.append(f"LARGE_LOSS: €{t['pnl_eur']:,.0f}")
        
        if t["pnl_eur"] > rules.get("large_gain_threshold", 8000): 
            flags.append(f"LARGE_GAIN: €{t['pnl_eur']:,.0f}")
        
        if t["price_deviation_pct"] > rules.get("price_deviation_pct", 5.0): 
            flags.append(f"PRICE_DEVIATION: {t['price_deviation_pct']:.1f}%")
        
        # Working hours check
        try:
            dt = datetime.fromisoformat(t["timestamp"])
            hours = rules.get("trading_hours", {"start": 6, "end": 22})
            if dt.hour >= hours["end"] or dt.hour < hours["start"]:
                flags.append(f"OFF_HOURS: {dt.hour:02d}:00")
        except ValueError:
            pass

        # Failure check
        if t["status"] == "FAILED":
            key = t["instrument"]
            failure_counts[key] = failure_counts.get(key, 0) + 1
            if failure_counts[key] >= rules.get("consecutive_failures", 2):
                flags.append(f"CONSECUTIVE_FAILURES: {failure_counts[key]}x on {key}")

        if flags:
            flagged.append({**t, "flags": flags, "flag_count": len(flags)})

    return sorted(flagged, key=lambda x: x["flag_count"], reverse=True)

# ── Claude API — AI Explanations ──────────────────────────────────────────────

def get_ai_explanation(flagged_trades: list[dict], summary: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️ ANTHROPIC_API_KEY not set. Skipping AI analysis."

    client = anthropic.Anthropic(api_key=api_key)

    trades_summary = json.dumps([{
        "id": t["id"], "instrument": t["instrument"], "type": t["type"], "pnl_eur": t["pnl_eur"],
        "flags": t["flags"], "status": t["status"], "counterparty": t["counterparty"]
    } for t in flagged_trades[:10]], indent=2)

    prompt = f"""You are a senior Middle Office analyst. Provide a concise operations briefing.
PORTFOLIO SUMMARY:
- Total trades: {summary['total_trades']}
- Total PnL: €{summary['total_pnl_eur']:,.2f}
FLAGGED TRADES:
{trades_summary}

Write a structured briefing:
1. EXECUTIVE SUMMARY (2 sentences)
2. TOP CONCERNS (bullet list, max 3)
3. RECOMMENDED ACTIONS"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# ── Summary Statistics ─────────────────────────────────────────────────────────

def build_summary(trades: list[dict], flagged: list[dict]) -> dict:
    settled = [t for t in trades if t["status"] == "SETTLED"]
    failed  = [t for t in trades if t["status"] == "FAILED"]
    total_pnl = sum(t["pnl_eur"] for t in settled)

    by_desk = {}
    for t in settled:
        desk = t["desk"]
        by_desk[desk] = by_desk.get(desk, 0) + t["pnl_eur"]

    return {
        "total_trades":    len(trades),
        "settled":          len(settled),
        "failed":           len(failed),
        "total_pnl_eur":    round(total_pnl, 2),
        "flagged_count":    len(flagged),
        "by_desk":          {k: round(v, 2) for k, v in by_desk.items()},
        "generated_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ── Slack Integration ─────────────────────────────────────────────────────────

def send_slack_alert(webhook_url: str, summary: dict, flagged: list[dict]):
    if not webhook_url: return

    color = "#991B1B" if summary["flagged_count"] > 0 else "#166534"
    flagged_text = ""
    for t in flagged[:5]:
        flags_str = ", ".join(t['flags'])
        flagged_text += f"• *{t['id']}* ({t['instrument']}): €{t['pnl_eur']:,.2f} ➔ {flags_str}\n"
    
    if not flagged_text: flagged_text = "All clear. ✅"

    slack_data = {
        "attachments": [{
            "color": color,
            "title": "📊 MiddleLayer Daily Ops Briefing",
            "text": (
                f"*Total Trades:* {summary['total_trades']} | *PnL:* €{summary['total_pnl_eur']:,.2f}\n"
                f"*Anomalies:* {summary['flagged_count']}\n\n*⚠️ Top Concerns:*\n{flagged_text}"
            ),
            "footer": "MiddleLayer Ops Intelligence",
            "ts": int(datetime.now().timestamp())
        }]
    }

    req = urllib.request.Request(
        webhook_url, 
        data=json.dumps(slack_data).encode("utf-8"), 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        urllib.request.urlopen(req)
        print("  ✅ Slack alert sent successfully!")
    except Exception:
        print("  ⚠️ Failed to send Slack alert.")

# ── HTML Dashboard ────────────────────────────────────────────────────────────

def generate_html_report(trades: list[dict], flagged: list[dict], summary: dict, ai_analysis: str) -> str:
    pnl_color = "#166534" if summary["total_pnl_eur"] >= 0 else "#991B1B"
    desk_rows = "".join(f'<tr><td>{desk}</td><td style="color:{"#166534" if pnl>=0 else "#991B1B"}">€{pnl:,.2f}</td></tr>' for desk, pnl in summary["by_desk"].items())
    flagged_rows = "".join(f'<tr style="background:{"#FEE2E2" if t["flag_count"]>=2 else "#FEF3C7"}"><td><strong>{t["id"]}</strong></td><td>{t["instrument"]}</td><td>{t["type"]}</td><td>€{t["pnl_eur"]:,.2f}</td><td>{"HIGH" if t["flag_count"]>=2 else "MEDIUM"}</td><td>{" · ".join(t["flags"])}</td></tr>' for t in flagged[:15])

    return f"""<html><body style="font-family:sans-serif;background:#F8FAFC;padding:20px;">
    <h1 style="color:#1B3A6B">MiddleLayer Dashboard</h1>
    <p>Generated: {summary['generated_at']}</p>
    <div style="display:flex;gap:10px;margin-bottom:20px;">
        <div style="background:white;padding:20px;border:1px solid #DDD;"><h3>€{summary['total_pnl_eur']:,.0f}</h3>Total PnL</div>
        <div style="background:white;padding:20px;border:1px solid #DDD;"><h3>{summary['flagged_count']}</h3>Flagged</div>
    </div>
    <div style="background:#EFF6FF;padding:15px;border-left:5px solid #3B82F6;margin-bottom:20px;">
        <h2>AI Operations Briefing</h2>{ai_analysis.replace(chr(10), '<br>')}
    </div>
    <h2>Flagged Trades</h2>
    <table border="1" style="width:100%;border-collapse:collapse;">
        <tr style="background:#EEE"><th>ID</th><th>Instrument</th><th>Type</th><th>PnL</th><th>Severity</th><th>Flags</th></tr>
        {flagged_rows}
    </table>
    </body></html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiddleLayer — Ops Intelligence")
    parser.add_argument("--input",  default="trades.csv",   help="Trade CSV file path")
    parser.add_argument("--output", default="report.html",  help="HTML report output path")
    parser.add_argument("--config", default="config.yaml",  help="Path to config.yaml")
    parser.add_argument("--slack-webhook", default=None,    help="Slack Webhook URL")
    args = parser.parse_args()

    print("MiddleLayer — starting analysis...")
    
    # 1. Load configuration and data
    config = load_config(args.config)
    trades = load_trades(args.input)
    
    # 2. Process engine
    trades  = run_pnl_engine(trades)
    flagged = detect_anomalies(trades, config)
    summary = build_summary(trades, flagged)

    # 3. Notifications
    if args.slack_webhook:
        send_slack_alert(args.slack_webhook, summary, flagged)

    # 4. AI & Report
    print("  Calling AI for analysis...")
    ai_analysis = get_ai_explanation(flagged, summary)
    html = generate_html_report(trades, flagged, summary, ai_analysis)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"✅ Report successfully generated: {args.output}")

if __name__ == "__main__":
    main()
