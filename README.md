# MiddleLayer
### Ops Intelligence for Fintech — Automated PnL Monitoring & AI-Powered Risk Flagging
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onurmertsozer/middlelayer/blob/main/middlelayer.ipynb)

### 📈 Project Roadmap (In Progress)
- [x] Phase 1: Core PnL Engine & CSV Ingestion
- [x] Phase 2: Slack Notification Integration
- [x] Phase 3: External Configuration (YAML)
- [ ] Phase 4: Interactive Data Viz (Chart.js)
- [ ] Phase 5: Database Persistence (PostgreSQL)
---

## The Problem

Early-stage fintech companies and trading desks run on spreadsheets until something breaks.
By the time a Middle Office team is hired, manual reconciliation has already caused missed flags,
delayed escalations, and avoidable losses.

**MiddleLayer fills that gap.**

It gives a lean ops team the visibility of a fully-staffed Middle Office — automated,
real-time, and explained in plain English by AI.

---

## What It Does

```
trades.csv  ──►  PnL Engine  ──►  Anomaly Detector  ──►  Claude AI  ──►  HTML Report
                 (MtM calc)       (5 rule sets)          (briefing)       (dashboard)
```

**1. Mark-to-Market PnL Calculation**
- Supports Equities, FX, Crypto, Commodities
- Multi-currency normalisation to EUR (USD, GBP, SEK, NOK, DKK, CHF, PLN)
- Long and short position handling

**2. Automated Anomaly Detection — 5 Rule Sets**
| Rule | Threshold | What it catches |
|------|-----------|-----------------|
| Large loss | < €5,000 | Single-trade outsized loss |
| Large gain | > €8,000 | Potential data error or fat finger |
| Price deviation | > 5% | Entry vs. market price mismatch |
| Off-hours trading | Before 06:00 or after 22:00 | Unauthorised or unusual activity |
| Consecutive failures | 2+ on same instrument | Counterparty or system issue |

**3. AI Operations Briefing (Claude API)**
Each flagged trade gets a plain-English explanation:
- What triggered the flag
- Why it matters operationally
- Recommended action for the ops team

**4. HTML Dashboard**
- Executive summary metrics (Total PnL, Settled, Failed, Flagged)
- Colour-coded severity (High / Medium)
- PnL breakdown by desk and currency
- Exportable for daily standup or CFO review

---

## Scenario

> *A Baltic fintech (Series A, 60 employees) processes 50+ trades per day across
> Equities, FX, and Crypto desks. The CFO and Ops Lead review PnL manually each morning.
> MiddleLayer runs automatically at 07:00, delivers a dashboard to their inbox,
> and flags anything that needs human attention — before the trading day begins.*

---

## Quick Start

### 1. Install dependencies
```bash
pip install anthropic
```

### 2. Set your API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run with sample data
```bash
python middlelayer.py --input trades.csv --output report.html
```

### 4. Open the report
```
open report.html
```

### 5. Optional: also export JSON
```bash
python middlelayer.py --input trades.csv --output report.html --json
```

---

## Sample Output

```
MiddleLayer — starting analysis...
  Input:          trades.csv
  Trades loaded:  50
  Settled:        45
  Failed:         5
  Flagged:        12
  Total PnL:      €14,821.40
  Calling Claude API for AI analysis...
  Report saved:   report.html

Done. Open report.html in your browser.
```

**AI Briefing example:**
> *"EXECUTIVE SUMMARY: Today's portfolio shows a net gain of €14,821 across 45 settled trades,
> but 3 consecutive SPOT GOLD failures from HSBC require immediate counterparty investigation.
> The Crypto desk's BTC/EUR position at 23:08 represents significant off-hours risk exposure.*
>
> *TOP CONCERNS:*
> *• TRD-037: BTC/EUR off-hours gain of €4,275 — verify authorisation*
> *• TRD-041/042/043: 3 consecutive SPOT GOLD failures — escalate to HSBC desk*
> *• TRD-039: NVIDIA off-hours sale at 01:15 — confirm trader instruction..."*

---

## File Structure

```
MiddleLayer/
├── middlelayer.py      # Core engine — PnL, anomaly detection, Claude API, HTML output
├── trades.csv          # Sample data — 50 synthetic trades (Baltic/Nordic markets)
├── report.html         # Generated dashboard (run the script to create)
└── README.md           # This file
```

---


### Features
- **Configurable Rules:** All anomaly thresholds are managed via `config.yaml`.
- **Slack Alerts:** Real-time notifications for critical trade flags.
- ## 🔔 Real-time Monitoring.
- <img width="350" height="108" alt="Screenshot 2026-03-13 at 16 58 44" src="https://github.com/user-attachments/assets/204b545f-6b1c-4707-b203-b7563c2e5a38" />

- **AI Briefing:** Automated insights powered by Claude 3.5 Sonnet.

### Roadmap
- [x] Slack / email delivery for daily reports
- [x] Configurable anomaly thresholds via `config.yaml`
- [ ] Multi-day trend analysis

---

## Built With

- **Python 3.11+** — PnL engine and anomaly detection
- **Claude API (claude-sonnet-4)** — AI operations briefing
- **No external dependencies** beyond `anthropic` SDK

---

## About

Built as a portfolio project demonstrating AI-augmented financial operations tooling.
Scenario based on real Middle Office workflows in Baltic and Nordic fintech environments.

*Author: Onur Mert Sözer — [linkedin.com/in/onurmert1](https://linkedin.com/in/onurmert1)*
*GitHub: [github.com/onurmertsozer](https://github.com/onurmertsozer)*
