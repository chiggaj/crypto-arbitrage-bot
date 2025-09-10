#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async multi-exchange arbitrage scanner (2-leg cross-exchange)
— with curated Tier-1/Tier-2 anchors + dynamic tiering —

Adds on top of the enriched version:
• Static Tier anchors (BTC/ETH/SOL/XRP as Tier-1; ADA/LTC/DOGE/LINK/DOT/AVAX as Tier-2)
• Anchors apply only if the coin is tradable on ≥2 of your venues (USD/USDT/USDC)
• Sector tagging for reporting + CSV
• Logs presence_count (number of venues listing the pair) into CSV
• Keeps dynamic tiering, fill probability, priority scoring, latency, adaptive backoff

Config: edit constants below or via env if you like.
"""

import os
import csv
import math
import time
import asyncio
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

import ccxt.async_support as ccxt  # async

# ===================== CONFIG =====================

EXCHANGE_IDS = ["gemini", "kraken", "coinbase"]

STABLE_QUOTES = ("USD", "USDT", "USDC")
REFRESH_SECONDS = 5
ORDERBOOK_LEVELS = 20
TRADE_SIZE_USD = 5_000

# Realism guards
SLIPPAGE = 0.001            # 0.10% per side
MIN_FILL_RATIO = 0.90       # require ≥95% of TRADE_SIZE fillable on BOTH legs
MIN_24H_VOLUME_USD = 5_000_000
IGNORE_IF_SPREAD_GT = 0.30  # ignore absurd spreads >30%

# Fee mode: "maker-taker" | "taker-maker" | "maker-maker" | "taker-taker"
FEE_MODE = "maker-taker"

TOP_N = 3
MAX_PAIRS = 250

# Retail-ish fees (edit for your tiers)
DEFAULT_FEES = {
    "gemini":   {"maker": 0.0025, "taker": 0.0035},
    "kraken":   {"maker": 0.0016, "taker": 0.0026},
    "coinbase": {"maker": 0.0040, "taker": 0.0060},
}

# CSV logging
WRITE_CSV = True
CSV_FILE = "arb_hits_enriched.csv"

# ====== Static anchors & sectors ======
# Curated anchors (modifiable). Anchors are enforced only if presence_count ≥ 2.
STATIC_TIERS = {
    # Tier 1 anchors (majors / deepest liquidity)
    "BTC": 1, "ETH": 1, "SOL": 1, "XRP": 1,
    # Tier 2 anchors
    "ADA": 2, "LTC": 2, "DOGE": 2, "LINK": 2, "DOT": 2, "AVAX": 2,
    # Others left to dynamic tiering
}

# Optional sector tags (purely for reporting)
SECTORS = {
    "BTC": "Store of Value",
    "ETH": "Smart Contract L1",
    "SOL": "Smart Contract L1",
    "ADA": "Smart Contract L1",
    "AVAX": "Smart Contract L1",
    "XRP": "Payments",
    "LTC": "Payments",
    "DOGE": "Payments / Meme",
    "LINK": "Oracles",
    "DOT": "Interoperability",
}

# ====== Options-style additions ======
# Dynamic tiering (rebuild every TIER_TTL seconds)
TIER_TTL = 60 * 30          # 30 min cache
T1_VOL = 500_000_000        # >= $500M 24h volume -> Tier 1
T2_VOL = 100_000_000        # >= $100M -> Tier 2
T3_VOL = 20_000_000         # >= $20M -> Tier 3 else Tier 4
# Tightness bonus (current book): if best % spread < these, bump score
TIGHT_BPS_T1 = 5            # 0.05%  (very tight)
TIGHT_BPS_T2 = 15           # 0.15%
TIGHT_BPS_T3 = 40           # 0.40%

TIER_SAFETY = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.40}

# Fill probability weights (0..1 blend)
FILL_W = {"tier": 0.40, "edge_vs_spread": 0.40, "latency": 0.20}
# Latency caps (ms): 0ms->1.0 score, cap->0.5 score
LAT_P50_CAP = 250.0
LAT_P90_CAP = 600.0

# Priority scoring weights
PRIORITY_W = {"roi": 0.50, "total_net": 0.30, "tier": 0.20}

# Adaptive backoff for orderbook fetch
BACKOFF_START = 0.15
BACKOFF_MAX = 2.0
BACKOFF_MULT = 1.8
RETRY_MAX = 3

# ===================== DATA STRUCTS =====================

@dataclass
class EffPrice:
    price: float   # VWAP
    qty: float     # base filled
    usd: float     # USD filled

@dataclass
class Route:
    base: str
    quote: str
    buy_ex: str
    sell_ex: str
    buy_px: float
    sell_px: float
    spread: float      # gross spread (fraction)
    be: float          # breakeven (fraction)
    net: float         # net fraction (spread - be)
    net_usd: float
    qty: float
    tier: int
    fill_prob: float
    priority: float
    roi: float         # net / buy_px approximated
    lat_p50: float
    lat_p90: float
    presence_count: int
    sector: str

# ===================== HELPERS =====================

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def fee_pair(buy_ex: str, sell_ex: str) -> Tuple[float, float]:
    """Return (buy_fee, sell_fee) given FEE_MODE config."""
    bf = DEFAULT_FEES.get(buy_ex, {"maker": 0.002, "taker": 0.002})
    sf = DEFAULT_FEES.get(sell_ex, {"maker": 0.002, "taker": 0.002})
    mode = FEE_MODE.lower()
    if mode == "maker-taker":
        return bf["maker"], sf["taker"]
    if mode == "taker-maker":
        return bf["taker"], sf["maker"]
    if mode == "maker-maker":
        return bf["maker"], sf["maker"]
    return bf["taker"], sf["taker"]  # taker-taker

def eligible_symbol(sym: str) -> Optional[Tuple[str, str]]:
    if "/" not in sym:
        return None
    base, quote = sym.split("/")
    if quote not in STABLE_QUOTES:
        return None
    return base, quote

async def fetch_ob_with_backoff(ex: ccxt.Exchange, eid: str, symbol: str):
    """Fetch orderbook with small adaptive backoff; returns (ob, latency_ms) or (None, None)."""
    backoff = BACKOFF_START
    for _ in range(RETRY_MAX):
        t0 = time.perf_counter()
        try:
            ob = await ex.fetch_order_book(symbol, limit=ORDERBOOK_LEVELS)
            dt = (time.perf_counter() - t0) * 1000.0
            return ob, dt
        except Exception:
            await asyncio.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULT, BACKOFF_MAX)
    return None, None

async def effective_price_for_size(orderbook: Dict, side: str, usd_target: float) -> Optional[EffPrice]:
    """
    VWAP for consuming `usd_target` on one side of the book.
    side: 'asks' (BUY) or 'bids' (SELL).
    Require ≥ MIN_FILL_RATIO * target.
    """
    levels = orderbook.get(side, [])
    if not levels:
        return None
    remain = float(usd_target)
    total_base = 0.0
    total_usd = 0.0
    for level in levels[:ORDERBOOK_LEVELS]:
        if not level or len(level) < 2:
            continue
        try:
            price = float(level[0])
            qty   = float(level[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(price) and math.isfinite(qty)) or price <= 0 or qty <= 0:
            continue
        take_usd  = min(remain, price * qty)
        take_base = take_usd / price
        total_usd  += take_usd
        total_base += take_base
        remain     -= take_usd
        if remain <= 1e-8:
            break
    if total_usd < usd_target * MIN_FILL_RATIO or total_base <= 0:
        return None
    vwap = total_usd / total_base
    return EffPrice(price=vwap, qty=total_base, usd=total_usd)

# ===================== VOLUME, PRESENCE & TIERING =====================

async def build_volume_maps(exchanges: Dict[str, ccxt.Exchange]) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Compute approx 24h USD volume per symbol per exchange from tickers."""
    vol_maps: Dict[str, Dict[Tuple[str, str], float]] = {eid: {} for eid in exchanges}
    async def get_one(eid: str, ex: ccxt.Exchange):
        try:
            tickers = await ex.fetch_tickers()
        except Exception:
            tickers = {}
        out: Dict[Tuple[str, str], float] = {}
        for sym, t in tickers.items():
            pair = eligible_symbol(sym)
            if not pair:
                continue
            base, quote = pair
            last = float(t.get("last") or 0) if t.get("last") is not None else 0.0
            qv   = float(t.get("quoteVolume") or 0.0)
            bv   = float(t.get("baseVolume") or 0.0)
            vol_usd = 0.0
            if qv and qv > 0:
                vol_usd = qv
            elif bv and bv > 0 and last > 0:
                vol_usd = bv * last
            out[(base, quote)] = max(vol_usd, 0.0)
        vol_maps[eid] = out
    await asyncio.gather(*[get_one(eid, ex) for eid, ex in exchanges.items()])
    return vol_maps

def pick_candidates(
    exchanges: Dict[str, ccxt.Exchange],
    vol_maps: Dict[str, Dict[Tuple[str, str], float]],
) -> List[Tuple[str, str]]:
    by_ex = {eid: set(ex.markets.keys()) for eid, ex in exchanges.items()}
    presence: Dict[Tuple[str, str], int] = {}
    max_vol: Dict[Tuple[str, str], float] = {}
    for eid, syms in by_ex.items():
        for sym in syms:
            pair = eligible_symbol(sym)
            if not pair:
                continue
            presence[pair] = presence.get(pair, 0) + 1
            v = vol_maps.get(eid, {}).get(pair, 0.0)
            if v > max_vol.get(pair, 0.0):
                max_vol[pair] = v
    candidates = []
    for pair, count in presence.items():
        if count >= 2 and max_vol.get(pair, 0.0) >= MIN_24H_VOLUME_USD:
            candidates.append(pair)
    candidates.sort(key=lambda pq: max_vol.get(pq, 0.0), reverse=True)
    return candidates[:MAX_PAIRS]

# --- tier cache & presence counts ---
_tiers: Dict[str, int] = {}      # base symbol -> tier
_tier_last = 0.0
_presence_counts: Dict[str, int] = {}  # base symbol -> #venues with USD/USDT/USDC

def tier_from_volume(vol_usd: float) -> int:
    if vol_usd >= T1_VOL: return 1
    if vol_usd >= T2_VOL: return 2
    if vol_usd >= T3_VOL: return 3
    return 4

def tier_for_symbol(base: str) -> int:
    return _tiers.get(base, 4)

def maybe_rebuild_tiers(
    vol_maps: Dict[str, Dict[Tuple[str, str], float]],
    sample_spreads_bps: Dict[str, float],
):
    """Compute dynamic tiers by volume + tightness; then apply static anchors where presence≥2."""
    global _tier_last, _tiers, _presence_counts
    now = time.time()
    if (now - _tier_last) < TIER_TTL and _tiers:
        return

    # Presence count (how many venues list USD/USDT/USDC for base)
    presence: Dict[str, int] = {}
    for _, mp in vol_maps.items():
        for (base, quote), _vol in mp.items():
            if quote in STABLE_QUOTES:
                presence[base] = presence.get(base, 0) + 1
    _presence_counts = presence

    # Max 24h volume per base across venues/quotes
    agg_vol: Dict[str, float] = {}
    for _, mp in vol_maps.items():
        for (base, quote), vol in mp.items():
            if quote in STABLE_QUOTES:
                agg_vol[base] = max(agg_vol.get(base, 0.0), vol)

    # Initial tier by volume
    tiers = {base: tier_from_volume(v) for base, v in agg_vol.items()}

    # Tightness bonus (live best top-of-book spread in bps)
    for base, bps in sample_spreads_bps.items():
        if base not in tiers:
            continue
        if bps <= TIGHT_BPS_T1: tiers[base] = min(1, tiers[base])
        elif bps <= TIGHT_BPS_T2: tiers[base] = min(2, tiers[base])
        elif bps <= TIGHT_BPS_T3: tiers[base] = min(3, tiers[base])

    # Static anchors override — ONLY if presence_count ≥ 2
    for base, forced in STATIC_TIERS.items():
        if presence.get(base, 0) >= 2:
            tiers[base] = forced

    _tiers = tiers
    _tier_last = now

# ===================== LATENCY STATS =====================

_lat_samples: Dict[str, List[float]] = {}  # eid -> ms list (cap last 200)

def record_latency(eid: str, ms: Optional[float]):
    if ms is None: return
    arr = _lat_samples.setdefault(eid, [])
    arr.append(ms)
    if len(arr) > 200:
        _lat_samples[eid] = arr[-200:]

def latency_snapshot(eid: str) -> Tuple[float, float]:
    arr = _lat_samples.get(eid, [])
    if not arr:
        return 200.0, 400.0
    p50 = statistics.median(arr)
    p90 = statistics.quantiles(arr, n=10)[-1] if len(arr) >= 10 else max(arr)
    return float(p50), float(p90)

def latency_score(p50: float, p90: float) -> float:
    # 0ms->1.0; at caps -> ~0.5
    p50_sc = max(0.0, min(1.0, 1.0 - (p50 / LAT_P50_CAP) * 0.5))
    p90_sc = max(0.0, min(1.0, 1.0 - (p90 / LAT_P90_CAP) * 0.5))
    return 0.5 * p50_sc + 0.5 * p90_sc

# ===================== CORE SCAN =====================

async def scan_once(
    exchanges: Dict[str, ccxt.Exchange],
    candidates: List[Tuple[str, str]],
):
    # Fetch all orderbooks in parallel (with latency)
    tasks = []
    for eid, ex in exchanges.items():
        for base, quote in candidates:
            tasks.append((eid, base, quote, ex, f"{base}/{quote}"))

    async def do_fetch(eid, base, quote, ex, sym):
        ob, ms = await fetch_ob_with_backoff(ex, eid, sym)
        record_latency(eid, ms)
        return eid, base, quote, ob

    results = await asyncio.gather(*[do_fetch(*t) for t in tasks], return_exceptions=False)

    # Index books
    books: Dict[Tuple[str, str, str], Dict] = {}
    for eid, b, q, ob in results:
        if ob:
            books[(eid, b, q)] = ob

    eids = list(exchanges.keys())
    ts = now_iso()
    routes: List[Route] = []
    closest_desc: Optional[str] = None
    closest_gap: Optional[float] = None

    # For tiering tightness bonus: track best current top-of-book % spread per base
    best_spread_bps: Dict[str, float] = {}

    for base, quote in candidates:
        # capture a quick top-of-book tightness proxy
        tight_bps = math.inf
        for eid in eids:
            ob = books.get((eid, base, quote))
            if not ob: continue
            try:
                best_ask = float(ob["asks"][0][0]) if ob["asks"] else None
                best_bid = float(ob["bids"][0][0]) if ob["bids"] else None
                if best_ask and best_bid and best_ask > 0 and best_bid > 0:
                    bps = (best_ask - best_bid) / ((best_ask + best_bid)/2.0) * 10000.0
                    tight_bps = min(tight_bps, bps)
            except Exception:
                pass
        if tight_bps != math.inf:
            best_spread_bps[base] = tight_bps

    # Rebuild tiers (cached), and compute volume maps for it
    vol_maps = await build_volume_maps(exchanges)
    maybe_rebuild_tiers(vol_maps, best_spread_bps)

    # convenience: presence_count lookup
    def presence_count_for(base: str) -> int:
        return int(_presence_counts.get(base, 0))

    for base, quote in candidates:
        for i in range(len(eids)):
            for j in range(len(eids)):
                if i == j:
                    continue
                buy_e, sell_e = eids[i], eids[j]
                ob_buy  = books.get((buy_e,  base, quote))
                ob_sell = books.get((sell_e, base, quote))
                if not ob_buy or not ob_sell:
                    continue

                eff_buy  = await effective_price_for_size(ob_buy,  "asks", TRADE_SIZE_USD)
                eff_sell = await effective_price_for_size(ob_sell, "bids", TRADE_SIZE_USD)
                if not eff_buy or not eff_sell:
                    continue

                px_buy  = eff_buy.price  * (1 + SLIPPAGE)
                px_sell = eff_sell.price * (1 - SLIPPAGE)
                spread  = (px_sell - px_buy) / px_buy

                if IGNORE_IF_SPREAD_GT is not None and spread > IGNORE_IF_SPREAD_GT:
                    continue

                buy_fee, sell_fee = fee_pair(buy_e, sell_e)
                be = buy_fee + sell_fee + 2 * SLIPPAGE
                net = spread - be

                # === Enrichments ===
                tier = tier_for_symbol(base)
                tier_safety = TIER_SAFETY.get(tier, 0.40)
                edge_vs_spread = (spread - be) / max(spread, 1e-9)
                edge_vs_spread = max(0.0, min(1.0, edge_vs_spread))
                p50_b, p90_b = latency_snapshot(buy_e)
                p50_s, p90_s = latency_snapshot(sell_e)
                lat_sc = 0.5 * latency_score(p50_b, p90_b) + 0.5 * latency_score(p50_s, p90_s)
                fill_prob = (
                    FILL_W["tier"]           * tier_safety +
                    FILL_W["edge_vs_spread"] * edge_vs_spread +
                    FILL_W["latency"]        * lat_sc
                )
                fill_prob = max(0.0, min(1.0, fill_prob))
                roi = max(0.0, net)
                pr_count = presence_count_for(base)
                sector = SECTORS.get(base, "")

                if net > 0:
                    qty = eff_buy.qty
                    routes.append(Route(
                        base=base, quote=quote,
                        buy_ex=buy_e, sell_ex=sell_e,
                        buy_px=px_buy, sell_px=px_sell,
                        spread=spread, be=be, net=net,
                        net_usd=TRADE_SIZE_USD * net,
                        qty=qty,
                        tier=tier,
                        fill_prob=fill_prob,
                        priority=0.0,   # set later
                        roi=roi,
                        lat_p50=(p50_b+p50_s)/2.0,
                        lat_p90=(p90_b+p90_s)/2.0,
                        presence_count=pr_count,
                        sector=sector
                    ))
                else:
                    gap = be - spread
                    desc = f"{base}/{quote} {buy_e}→{sell_e}"
                    if closest_gap is None or gap < closest_gap:
                        closest_gap = gap
                        closest_desc = desc

    # Normalize & rank priority (ROI + total_net + tier)
    if routes:
        max_roi = max((r.roi for r in routes), default=1e-9)
        max_net = max((r.net_usd for r in routes), default=1e-9)
        for r in routes:
            roi_norm = r.roi / max_roi if max_roi > 0 else 0.0
            net_norm = r.net_usd / max_net if max_net > 0 else 0.0
            tier_sc  = TIER_SAFETY.get(r.tier, 0.40)
            r.priority = (PRIORITY_W["roi"] * roi_norm +
                          PRIORITY_W["total_net"] * net_norm +
                          PRIORITY_W["tier"] * tier_sc)

    routes.sort(key=lambda r: (r.priority, r.net_usd), reverse=True)

    if not routes:
        if closest_desc is not None and closest_gap is not None:
            print(f"{ts} — No profitable opps ({FEE_MODE}; depth {ORDERBOOK_LEVELS}). "
                  f"Closest: {closest_desc} | gap {closest_gap*100:.2f}% to breakeven.")
        else:
            print(f"{ts} — No profitable opps ({FEE_MODE}; depth {ORDERBOOK_LEVELS}).")
        return

    print(f"{ts} — Top {min(TOP_N, len(routes))} profitable routes "
          f"({FEE_MODE}; depth {ORDERBOOK_LEVELS}).")
    for idx, r in enumerate(routes[:TOP_N], start=1):
        bf, sf = fee_pair(r.buy_ex, r.sell_ex)
        print(
            f"{idx:>2}. {r.base}/{r.quote}: {r.buy_ex}→{r.sell_ex} | "
            f"Spread {r.spread*100:.2f}% vs BE {r.be*100:.2f}% | "
            f"Net ${r.net_usd:.2f} ({r.net*100:.2f}%) on ${TRADE_SIZE_USD:,.0f} | "
            f"qty {r.qty:.6f} | Tier T{r.tier}"
            f"{f' ({r.sector})' if r.sector else ''} | "
            f"FillProb {r.fill_prob:.2f} | Presence {r.presence_count} | "
            f"Priority {r.priority:.3f} | fees(b={bf*100:.2f}%, s={sf*100:.2f}%) | "
            f"lat p50~{r.lat_p50:.0f}ms p90~{r.lat_p90:.0f}ms"
        )

    if WRITE_CSV:
        newfile = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp","base","quote","sector","presence_count",
                    "buy_ex","sell_ex",
                    "buy_px","sell_px",
                    "spread_pct","breakeven_pct","net_pct","net_usd",
                    "size_usd","qty_base",
                    "fee_mode","slippage","min_fill_ratio","min_24h_vol_usd",
                    "tier","fill_prob","priority","roi",
                    "lat_p50_ms","lat_p90_ms"
                ])
            for r in routes:
                w.writerow([
                    ts, r.base, r.quote, r.sector, r.presence_count,
                    r.buy_ex, r.sell_ex,
                    f"{r.buy_px:.8f}", f"{r.sell_px:.8f}",
                    f"{r.spread*100:.4f}", f"{r.be*100:.4f}",
                    f"{r.net*100:.4f}", f"{r.net_usd:.2f}",
                    f"{TRADE_SIZE_USD:.2f}", f"{r.qty:.8f}",
                    FEE_MODE, f"{SLIPPAGE:.6f}", f"{MIN_FILL_RATIO:.4f}", f"{MIN_24H_VOLUME_USD:.0f}",
                    f"T{r.tier}", f"{r.fill_prob:.3f}", f"{r.priority:.4f}", f"{r.roi:.4f}",
                    f"{r.lat_p50:.1f}", f"{r.lat_p90:.1f}",
                ])

# ===================== MAIN LOOP =====================

async def main():
    # Build exchanges
    ex_list = await asyncio.gather(*[build_exchange(eid) for eid in EXCHANGE_IDS])
    exchanges: Dict[str, ccxt.Exchange] = {eid: ex for eid, ex in zip(EXCHANGE_IDS, ex_list)}

    # Build volume maps & pick candidates
    vol_maps = await build_volume_maps(exchanges)
    candidates = pick_candidates(exchanges, vol_maps)

    print("🚀 Scanner starting | "
          f"FEE_MODE={FEE_MODE} | depth {ORDERBOOK_LEVELS} | refresh {REFRESH_SECONDS}s | "
          f"trade size ${TRADE_SIZE_USD:,}")
    print(f"Liquidity guards: MIN_24H_VOLUME_USD={MIN_24H_VOLUME_USD:,} | "
          f"SLIPPAGE={SLIPPAGE*100:.2f}%/side | MIN_FILL_RATIO={MIN_FILL_RATIO*100:.0f}%")
    if IGNORE_IF_SPREAD_GT is not None:
        print(f"Outlier guard: IGNORE_IF_SPREAD_GT={IGNORE_IF_SPREAD_GT*100:.0f}%")
    print(f"Loaded exchanges: {list(exchanges)} | eligible pairs: {len(candidates)}")
    print(f"Anchors loaded: { {k:v for k,v in STATIC_TIERS.items()} } (enforced only if presence≥2)")

    try:
        while True:
            try:
                await scan_once(exchanges, candidates)
            except Exception as e:
                print("scan_once error:", e)
            await asyncio.sleep(REFRESH_SECONDS)
    finally:
        for ex in exchanges.values():
            try:
                await ex.close()
            except Exception:
                pass

async def build_exchange(eid: str):
    """Create & load async ccxt exchange instance (with creds if present)."""
    cls = getattr(ccxt, eid)
    kwargs = {"enableRateLimit": True}
    key = os.getenv(f"{eid.upper()}_KEY")
    secret = os.getenv(f"{eid.upper()}_SECRET")
    password = os.getenv(f"{eid.upper()}_PASSPHRASE")  # coinbase
    if key and secret:
        kwargs.update({"apiKey": key, "secret": secret})
    if password:
        kwargs.update({"password": password})
    ex = cls(kwargs)
    await ex.load_markets()
    return ex

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bye!")
