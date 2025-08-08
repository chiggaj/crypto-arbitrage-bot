import ccxt
import asyncio
import traceback
import datetime
import os
import csv
from pathlib import Path
import smtplib, ssl
from email.message import EmailMessage
from collections import defaultdict
from dotenv import load_dotenv

# ---------- .env ----------
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=465
# SMTP_USER=youremail@gmail.com
# SMTP_PASS=your_app_password
# SMS_TO=7045551234@vtext.com  (or @vzwpix.com for MMS)
load_dotenv()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMS_TO    = os.getenv("SMS_TO")

# ---------- Config ----------
capital = 5000                   # USD simulated capital
min_profit_threshold = 1         # Skip tiny profits (< $1) to reduce noise
spread_alert_threshold = 0.4     # % spread to trigger an alert (pre-fees)
alert_cooldown_seconds = 300     # 5 minutes per coin

# Exchanges
exchanges = {
    "coinbase": ccxt.coinbase(),
    "kraken":   ccxt.kraken(),
    "gemini":   ccxt.gemini(),
}

# Estimated taker fees (rough defaults)
exchange_fees = {
    "coinbase": 0.0050,  # 0.50%
    "kraken":   0.0026,  # 0.26%
    "gemini":   0.0035,  # 0.35%
}

# Per-exchange symbols (None if unsupported). POL/XLM skipped on Gemini.
symbols = {
    "BTC":  {"coinbase": "BTC/USD",   "kraken": "XBT/USD",     "gemini": "BTC/USD"},
    "ETH":  {"coinbase": "ETH/USD",   "kraken": "ETH/USD",     "gemini": "ETH/USD"},
    "SOL":  {"coinbase": "SOL/USD",   "kraken": "SOL/USD",     "gemini": "SOL/USD"},
    "POL":  {"coinbase": "POL/USD",   "kraken": "POL/USD",     "gemini": None},
    "XRP":  {"coinbase": "XRP/USD",   "kraken": "XRP/USD",     "gemini": "XRP/USD"},
    "ADA":  {"coinbase": "ADA/USD",   "kraken": "ADA/USD",     "gemini": "ADA/USD"},
    "XLM":  {"coinbase": "XLM/USD",   "kraken": "XLM/USD",     "gemini": None},
    "AAVE": {"coinbase": "AAVE/USD",  "kraken": "AAVE/USD",    "gemini": "AAVE/USD"},
}

# ---------- SMS (Email-to-SMS) ----------
def send_sms(message: str):
    if not (SMTP_USER and SMTP_PASS and SMS_TO):
        print("⚠️  SMS not configured (missing SMTP_USER/SMTP_PASS/SMS_TO). Skipping alert.")
        return
    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = SMS_TO
    msg["Subject"] = ""
    msg.set_content(message[:150])
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("📲 SMS alert sent.")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

last_alert_ts = defaultdict(lambda: 0)  # per-coin cooldown

# ---------- Logging ----------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def log_row(row: dict):
    """Append a row to today's CSV (auto create with header)."""
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    fpath = LOG_DIR / f"arbitrage_{date_str}.csv"
    write_header = not fpath.exists()
    with fpath.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp","coin","buy_ex","sell_ex",
                "buy_price","sell_price","spread_pct",
                "gross_profit","fees","net_profit"
            ]
        )
        if write_header:
            w.writeheader()
        w.writerow(row)

# ---------- Helpers ----------
async def safe_fetch_ticker(ex: ccxt.Exchange, market: str):
    try:
        t = ex.fetch_ticker(market)
        return t.get("last")
    except Exception as e:
        print(f"❌ Error fetching {market} from {ex.id}: {e}")
        return None

def fmt_price(p):
    return f"${p:.4f}" if p is not None else "-"

# ---------- Main Loop ----------
async def run():
    # Load markets once
    for name, ex in exchanges.items():
        try:
            ex.load_markets()
        except Exception as e:
            print(f"⚠️ Could not load markets for {name}: {e}")

    while True:
        print("\n🔄 Fetching prices...\n")
        total_simulated_profit = 0.0
        now_ts = datetime.datetime.now().timestamp()

        for coin, per_ex_symbols in symbols.items():
            try:
                prices = {}
                for name, ex in exchanges.items():
                    sym = per_ex_symbols.get(name)
                    if not sym:
                        continue
                    if hasattr(ex, "markets") and ex.markets and sym not in ex.markets:
                        continue
                    price = await safe_fetch_ticker(ex, sym)
                    if price is not None:
                        prices[name] = price

                if len(prices) < 2:
                    print(f"⚠️ {coin}: not enough price data (have: {', '.join(prices.keys()) or 'none'})\n")
                    continue

                # Show prices
                print(f"{coin} Prices:")
                print(f"  Coinbase: {fmt_price(prices.get('coinbase'))}")
                print(f"  Kraken:   {fmt_price(prices.get('kraken'))}")
                print(f"  Gemini:   {fmt_price(prices.get('gemini'))}")

                # Spread & profit
                sorted_px = sorted(prices.items(), key=lambda kv: kv[1])
                low_name,  low_price  = sorted_px[0]
                high_name, high_price = sorted_px[-1]
                spread_pct = ((high_price - low_price) / low_price) * 100.0
                print(f"  Spread: {spread_pct:.2f}%")

                if spread_pct >= spread_alert_threshold and high_price > low_price:
                    qty = capital / low_price
                    gross_profit = qty * (high_price - low_price)
                    buy_fee  = capital * exchange_fees.get(low_name, 0.0)
                    sell_fee = (qty * high_price) * exchange_fees.get(high_name, 0.0)
                    fees = buy_fee + sell_fee
                    net_profit = gross_profit - fees

                    if net_profit > min_profit_threshold:
                        print(f"🚨 Arbitrage Opportunity: {coin}")
                        print(f"  Buy on:  {low_name} @ ${low_price:.4f}")
                        print(f"  Sell on: {high_name} @ ${high_price:.4f}")
                        print(f"  Gross: ${gross_profit:.2f} | Fees: ${fees:.2f} | Net: ${net_profit:.2f}\n")
                        total_simulated_profit += net_profit

                        # ---- Log the opportunity ----
                        log_row({
                            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                            "coin": coin,
                            "buy_ex": low_name,
                            "sell_ex": high_name,
                            "buy_price": f"{low_price:.6f}",
                            "sell_price": f"{high_price:.6f}",
                            "spread_pct": f"{spread_pct:.4f}",
                            "gross_profit": f"{gross_profit:.2f}",
                            "fees": f"{fees:.2f}",
                            "net_profit": f"{net_profit:.2f}",
                        })

                        # ---- SMS (cooldown) ----
                        if now_ts - last_alert_ts[coin] >= alert_cooldown_seconds:
                            sms = (
                                f"{coin} arb {spread_pct:.2f}% | "
                                f"Buy {low_name} {low_price:.2f} -> "
                                f"Sell {high_name} {high_price:.2f} | "
                                f"Net~${net_profit:.0f}"
                            )
                            send_sms(sms)
                            last_alert_ts[coin] = now_ts
                        else:
                            print(f"⏳ SMS cooldown active for {coin}.")
                    else:
                        print("  Opportunity exists but net profit after fees is too small.\n")
                else:
                    print("  No valid opportunity this round.\n")

            except Exception as e:
                print(f"❌ General error on {coin}: {e}")
                traceback.print_exc()
                print()

        # Optional: log a periodic summary row (makes daily aggregation easy)
        log_row({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "coin": "SUMMARY",
            "buy_ex": "",
            "sell_ex": "",
            "buy_price": "",
            "sell_price": "",
            "spread_pct": "",
            "gross_profit": "",
            "fees": "",
            "net_profit": f"{total_simulated_profit:.2f}",
        })

        print(f"📊 Total Estimated Profit This Round: ${total_simulated_profit:.2f}")
        print(f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run())
