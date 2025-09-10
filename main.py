#!/usr/bin/env python3
import os
import time
import math
import logging
from typing import Dict, Optional

import ccxt
from dotenv import load_dotenv

# ---------------------------
# Settings you can tweak
# ---------------------------

SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "ADA", "XLM", "AAVE", "POL"]  # not all exist on all venues
QUOTE = "USD"                      # use USD markets (e.g., BTC/USD)
REFRESH_SECONDS = 10               # delay between rounds
TRADE_SIZE_USD = 5000              # notional for P&L simulation
DEFAULT_SLIPPAGE = 0.0005          # 0.05% per side

# Realistic retail fee tiers (update if your tier improves)
FEES = {
    "coinbase": {"taker": 0.0060, "maker": 0.0040},  # 0.60% / 0.40%
    "kraken":   {"taker": 0.0026, "maker": 0.0016},  # 0.26% / 0.16%
    "gemini":   {"taker": 0.0035, "maker": 0.0025},  # 0.35% / 0.25%
}
USE_MAKER = False  # set True only if you really post makers and get filled

# Optional: withdrawal fees (flat coin amounts) if you simulate transfers
WITHDRAWAL_FEES = {
    # "BTC": {"coinbase": 0.0003, "kraken": 0.00015, "gemini": 0.0001},
}

INCLUDE_WITHDRAWAL = False  # usually False for fast arb with pre-positioned inventory

# ---------------------------
# Setup
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
load_dotenv()  # loads .env if present

def make_client(name: str):
    """Build a ccxt client; authenticated if keys exist, else public."""
    name = name.lower()
    if name not in ("coinbase", "kraken", "gemini"):
        raise ValueError(name)

    if name == "coinbase":
        api, sec, pwd = (os.getenv("COINBASE_API_KEY"),
                         os.getenv("COINBASE_API_SECRET"),
                         os.getenv("COINBASE_PASSPHRASE"))
        cfg = {"apiKey": api, "secret": sec, "password": pwd} if api and sec and pwd else {}
        client = ccxt.coinbase(cfg)

    elif name == "kraken":
        api, sec = os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")
        cfg = {"apiKey": api, "secret": sec} if api and sec else {}
        client = ccxt.kraken(cfg)

    else:  # gemini
        api, sec = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_SECRET")
        cfg = {"apiKey": api, "secret": sec} if api and sec else {}
        client = ccxt.gemini(cfg)

    # be nice to APIs
    client.enableRateLimit = True
    client.timeout = 15000
    return client

clients = {
    "coinbase": make_client("coinbase"),
    "kraken":   make_client("kraken"),
    "gemini":   make_client("gemini"),
}

def fetch_last_price(ex_name: str, symbol: str) -> Optional[float]:
    """
    Return last trade/mark price for `symbol` (e.g., 'BTC/USD') from exchange ex_name.
    Gracefully returns None if the market isn't available.
    """
    ex = clients[ex_name]
    try:
        market_symbol = symbol
        # Some exchanges use slightly different ids; rely on unified symbol if available
        if market_symbol not in ex.markets:
            # load markets once per process if not loaded
            if not getattr(ex, "_markets_loaded", False):
                ex.load_markets()
                ex._markets_loaded = True
            if market_symbol not in ex.markets:
                return None

        ticker = ex.fetch_ticker(market_symbol)
        # Prefer 'last', fall back to 'close'
        price = ticker.get("last") or ticker.get("close")
        if price and math.isfinite(price):
            return float(price)
        return None
    except Exception:
        return None

def calc_net_arbitrage(
    base: str,
    prices: Dict[str, Optional[float]],
    buyer: str,
    seller: str,
    trade_size_usd: float = TRADE_SIZE_USD,
    use_maker: bool = USE_MAKER,
    slippage: float = DEFAULT_SLIPPAGE,
    include_withdrawal: bool = INCLUDE_WITHDRAWAL
):
    pb, ps = prices.get(buyer), prices.get(seller)
    if pb is None or ps is None:
        return {"ok": False, "reason": "missing price"}

    fee_key = "maker" if use_maker else "taker"
    fb = FEES.get(buyer, {}).get(fee_key, 0.0)
    fs = FEES.get(seller, {}).get(fee_key, 0.0)

    spread = (ps - pb) / pb  # raw %
    eff_pb = pb * (1 + slippage)
    eff_ps = ps * (1 - slippage)
    buy_cost  = eff_pb * (1 + fb)
    sell_take = eff_ps * (1 - fs)

    qty = trade_size_usd / buy_cost
    net_dollars = qty * (sell_take - buy_cost)

    withdraw_cost = 0.0
    if include_withdrawal and base in WITHDRAWAL_FEES and buyer in WITHDRAWAL_FEES[base]:
        withdraw_fee_units = WITHDRAWAL_FEES[base][buyer]
        withdraw_cost = withdraw_fee_units * eff_pb  # approximate in USD using buy side
        net_dollars -= withdraw_cost

    breakeven = fb + fs + 2 * slippage
    return {
        "ok": True,
        "buyer": buyer,
        "seller": seller,
        "price_buy": pb,
        "price_sell": ps,
        "spread_pct": spread * 100,
        "breakeven_pct": breakeven * 100,
        "qty": qty,
        "trade_size_usd": trade_size_usd,
        "net_pnl_usd": net_dollars,
        "withdraw_cost_usd": withdraw_cost,
        "maker_mode": use_maker,
    }

def format_money(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"${x:,.4f}" if x < 10 else f"${x:,.2f}"

def run_once():
    round_total = 0.0
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n===== {ts} =====\n")

    for base in SYMBOLS:
        symbol = f"{base}/{QUOTE}"
        prices = {
            "coinbase": fetch_last_price("coinbase", symbol),
            "kraken":   fetch_last_price("kraken",   symbol),
            "gemini":   fetch_last_price("gemini",   symbol),
        }

        # pretty print prices
        print(f"{base} Prices:")
        for ex in ("coinbase", "kraken", "gemini"):
            print(f"  {ex.capitalize():8}: {format_money(prices[ex])}")

        # choose venues ignoring Nones
        available = {ex: px for ex, px in prices.items() if px is not None}
        if len(available) < 2:
            print("  Not enough venues with prices. Skipping.\n")
            continue

        buyer  = min(available, key=available.get)   # cheapest
        seller = max(available, key=available.get)   # highest
        result = calc_net_arbitrage(base, prices, buyer, seller)

        if result["ok"]:
            print(f"  Spread: {result['spread_pct']:.2f}%  |  "
                  f"Breakeven: {result['breakeven_pct']:.2f}%")
            print(f"  Net P&L for ${result['trade_size_usd']:.0f}: "
                  f"{result['net_pnl_usd']:+.2f}  "
                  f"(BUY {result['buyer']} @ {format_money(result['price_buy'])}, "
                  f"SELL {result['seller']} @ {format_money(result['price_sell'])})")
            if INCLUDE_WITHDRAWAL and result["withdraw_cost_usd"] > 0:
                print(f"  (Includes est. withdrawal cost: {result['withdraw_cost_usd']:.2f})")
            round_total += result["net_pnl_usd"]
            if result["net_pnl_usd"] > 0:
                print("  ✅ Net-positive after fees/slippage.")
            else:
                print("  ❌ Not profitable after fees/slippage.")
        else:
            print("  Could not compute P&L.")

        print()

    print(f"📊 Net P&L this round (after fees/slippage): {round_total:+.2f}\n")

def main():
    # Load markets once for faster symbol existence checks
    for ex in clients.values():
        try:
            ex.load_markets()
            ex._markets_loaded = True
        except Exception:
            pass

    try:
        while True:
            run_once()
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
