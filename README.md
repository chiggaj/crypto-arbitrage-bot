# Crypto Arbitrage Bot

A Python-based arbitrage scanner that monitors price discrepancies across multiple crypto exchanges (Coinbase, Kraken, Gemini, etc.) and logs profitable spread opportunities. Designed for automated execution, risk analysis, and detailed logging.

---

## ⚙️ Features
- Monitors multiple exchanges using APIs (via `ccxt`).
- Detects 2-leg arbitrage spreads (buy on one, sell on another).
- Logs opportunities above configurable thresholds (e.g., 0.4%).
- Outputs structured CSV logs for later analysis.
- Alerts for profitability potential (PnL tracking).
- Configurable per-exchange capital allocation.

---

## 📦 Installation
Clone this repository and install dependencies:
```bash
git clone git@github.com:chiggaj/crypto-arbitrage-bot.git
cd crypto-arbitrage-bot
pip install -r requirements.txt
