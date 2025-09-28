# === ema_crossover_bot_online.py (3m, margin model, risk=2%) ===
# Run: python ema_crossover_bot_online.py

import sys, subprocess, importlib, os, math, json, time, warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

def _pip_install(pkgs):
    env = os.environ.copy()
    env["PIP_USE_PEP517"] = "1"
    for p in pkgs:
        try:
            importlib.import_module(p if p != "yfinance" else "yfinance")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", p], env=env)

_pip_install(["pandas", "numpy", "yfinance", "matplotlib", "requests"])

import numpy as np
import pandas as pd
import yfinance as yf
import requests
warnings.filterwarnings("ignore")

# ---------------- Config ----------------
@dataclass
class BotConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","XAUUSD"
    ])
    yahoo_primary: Dict[str, str] = field(default_factory=lambda: {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X",
        "USDCAD": "USDCAD=X",
        "AUDUSD": "AUDUSD=X",
        "NZDUSD": "NZDUSD=X",
        "XAUUSD": "GC=F",  # reliable intraday gold (futures)
    })
    yahoo_fallbacks: Dict[str, List[str]] = field(default_factory=lambda: {
        "XAUUSD": ["XAUUSD=X", "XAU=X", "GLD"],
    })

    # ===== 3-minute timeframe =====
    timeframe_minutes: int = 3

    # Data retrieval plan:
    #  - Stitch multiple 1m windows (safe 6d chunks) to about 28d, then resample → 3m
    #  - If 1m stitching fails, fallback to 5m/60d and resample → 3m (partial coverage)
    stitch_1m_days: int = 28
    stitch_chunk_days: int = 6          # stay below 7d limit per 1m request
    fallback_attempts: List[Dict] = field(default_factory=lambda: [
        {"interval": "5m", "period": "60d"},
    ])

    # Strategy params
    ema_fast: int = 30
    ema_slow: int = 50
    atr_len: int = 14
    atr_mult_sl: float = 1.0
    tp_rr: float = 2.0

    # Risk & caps
    equity_start: float = 10_000.0
    risk_per_trade: float = 0.02  # 2% per trade (intraday-safe)
    max_new_trades_per_day: int = 3
    max_new_trades_per_day_per_symbol: int = 3

    # Trading frictions
    fee_bps: float = 0.5
    slippage_bps: float = 0.5

    # Limit order behavior
    gtc_valid_bars: int = 5
    allow_same_bar_fill: bool = False

    # News (FinancialModelingPrep)
    use_news_api: bool = True
    fmp_api_key_env: str = "FMP_API_KEY"  # falls back to "demo"
    news_resume_minutes: int = 10
    news_high_impact_labels: List[str] = field(default_factory=lambda: ["High","high","3","High Impact","High Importance"])
    currencies_by_symbol: Dict[str, List[str]] = field(default_factory=lambda: {
        "EURUSD": ["EUR","USD"],
        "GBPUSD": ["GBP","USD"],
        "USDJPY": ["USD","JPY"],
        "USDCHF": ["USD","CHF"],
        "USDCAD": ["USD","CAD"],
        "AUDUSD": ["AUD","USD"],
        "NZDUSD": ["NZD","USD"],
        "XAUUSD": ["USD"],
    })

    # Margin / leverage model
    fx_leverage: int = 30          # adjust if needed
    margin_rate: Optional[float] = None

    # Outputs
    out_dir: str = "."
    trade_log_csv: str = "trades_log.csv"
    equity_curve_csv: str = "equity_curve.csv"
    equity_curve_png: str = "equity_curve.png"
    performance_json: str = "performance_report.json"

    def __post_init__(self):
        if self.margin_rate is None:
            self.margin_rate = 1.0 / float(self.fx_leverage)

# ---------------- Utils & indicators ----------------
def within_bar_reaches(price: float, bar_low: float, bar_high: float) -> bool:
    return (bar_low - 1e-12) <= price <= (bar_high + 1e-12)

def apply_bps(price: float, bps: float, side: str) -> float:
    return price * (1 + bps/10000.0) if side == "buy" else price * (1 - bps/10000.0)

def round_to_lot(qty: float, lot_step: float = 1.0) -> float:
    return math.floor(qty / lot_step) * lot_step

def ensure_dir(path: str):
    if path and path != ".": os.makedirs(path, exist_ok=True)

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    return true_range(df).rolling(window=length, min_periods=length).mean()

# ---------------- News filter ----------------
class NewsFilter:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.events: List[datetime] = []

    def fetch_high_impact_events(self, start_date: datetime, end_date: datetime, tracked_ccys: List[str]):
        key = os.getenv(self.cfg.fmp_api_key_env, "demo")
        url = "https://financialmodelingprep.com/api/v3/economic_calendar"
        params = {"from": start_date.date().isoformat(), "to": end_date.date().isoformat(), "apikey": key}
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            events = []
            for ev in data if isinstance(data, list) else []:
                ccy = str(ev.get("country", ev.get("currency",""))).upper().strip()
                impact = str(ev.get("impact", ev.get("importance",""))).strip()
                when_str = ev.get("date", ev.get("time", ev.get("timestamp","")))
                if "date" in ev and "time" in ev and ev["time"]:
                    when_str = f"{ev['date']} {ev['time']}"
                if not when_str: continue
                try:
                    dt_utc = pd.to_datetime(when_str, utc=True).to_pydatetime()
                except Exception:
                    continue
                if impact in self.cfg.news_high_impact_labels and (ccy in tracked_ccys or ccy == "ALL"):
                    events.append(dt_utc)
            self.events = sorted(list(set(events)))
        except Exception:
            self.events = []

    def is_blackout(self, ts_utc: datetime) -> bool:
        for ev in self.events:
            if ev <= ts_utc <= (ev + timedelta(minutes=self.cfg.news_resume_minutes)):
                return True
        return False

# ---------------- Data fetcher (with 1m stitcher) ----------------
class DataFetcher:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    def _normalize_price_df(self, raw: pd.DataFrame, want_ticker_hint: Optional[str]=None) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = {str(c[0]).lower() for c in df.columns}
            fields = {"open","high","low","close","adj close","volume"}
            if fields & lvl0:
                tickers = sorted({c[1] for c in df.columns})
                t = want_ticker_hint or (tickers[0] if tickers else None)
                if t is None: return pd.DataFrame()
                df = df.xs(t, axis=1, level=1)
            else:
                tickers = sorted({c[0] for c in df.columns})
                t = want_ticker_hint or (tickers[0] if tickers else None)
                if t is None: return pd.DataFrame()
                df = df.xs(t, axis=1, level=0)
        df.columns = [str(c).strip().lower() for c in df.columns]
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        except Exception:
            df.index = pd.to_datetime(df.index, utc=True)
        if "adj close" in df.columns and "close" not in df.columns:
            df = df.rename(columns={"adj close":"close"})
        for c in ["open","high","low","close"]:
            if c not in df.columns:
                return pd.DataFrame()
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["open","high","low","close","volume"]].dropna()

    def _download(self, ticker: str, interval: str, start=None, end=None, period=None) -> pd.DataFrame:
        raw = yf.download(ticker, interval=interval, start=start, end=end, period=period,
                          progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        return self._normalize_price_df(raw, want_ticker_hint=ticker)

    def _fetch_1m_stitched(self, ticker: str, days: int, chunk_days: int) -> pd.DataFrame:
        # Stitch multiple small windows of 1m bars into a single DF
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        dfs = []
        cur_start = start
        while cur_start < end:
            cur_end = min(cur_start + timedelta(days=chunk_days), end)
            try:
                df = self._download(ticker, "1m", start=cur_start, end=cur_end)
                if not df.empty:
                    dfs.append(df)
            except Exception:
                pass
            cur_start = cur_end
            time.sleep(0.2)  # be nice to Yahoo
        if not dfs:
            return pd.DataFrame()
        out = pd.concat(dfs, axis=0)
        out = out[~out.index.duplicated(keep="last")]
        out = out.sort_index()
        return out

    def _fetch_symbol_raw_any(self, sym: str) -> pd.DataFrame:
        tickers = [self.cfg.yahoo_primary.get(sym, sym)] + [t for t in self.cfg.yahoo_fallbacks.get(sym, [])]
        last_err = None
        # First try 1m stitching for each ticker
        for tkr in tickers:
            try:
                df1m = self._fetch_1m_stitched(tkr, self.cfg.stitch_1m_days, self.cfg.stitch_chunk_days)
                if not df1m.empty:
                    df1m.attrs["yf_symbol"] = tkr
                    df1m.attrs["yf_interval"] = "1m(stitched)"
                    df1m.attrs["yf_period"] = f"{self.cfg.stitch_1m_days}d"
                    return df1m
            except Exception as e:
                last_err = e
                continue
        # Fallback attempts (e.g., 5m/60d)
        for tkr in tickers:
            for att in self.cfg.fallback_attempts:
                try:
                    df = self._download(tkr, att["interval"], period=att.get("period"))
                    if not df.empty:
                        df.attrs["yf_symbol"] = tkr
                        df.attrs["yf_interval"] = att["interval"]
                        df.attrs["yf_period"] = att.get("period", "")
                        return df
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            print(f"[WARN] {sym}: fetch failed ({type(last_err).__name__}: {last_err})")
        else:
            print(f"[WARN] {sym}: no data from Yahoo")
        return pd.DataFrame()

    def resample_to_tf(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        rule = f"{minutes}T"
        ohlc = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        ohlc = ohlc.reset_index()
        if ohlc.columns[0].lower() != "timestamp":
            ohlc = ohlc.rename(columns={ohlc.columns[0]: "timestamp"})
        ohlc["timestamp"] = pd.to_datetime(ohlc["timestamp"], utc=True)
        return ohlc[["timestamp","open","high","low","close","volume"]].dropna()

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for sym in self.cfg.symbols:
            df_raw = self._fetch_symbol_raw_any(sym)
            if df_raw.empty:
                print(f"[DATA] {sym}: EMPTY after fetch/normalize")
                data[sym] = pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
            else:
                df_tf = self.resample_to_tf(df_raw, self.cfg.timeframe_minutes)
                src = f"{df_raw.attrs.get('yf_symbol','?')} {df_raw.attrs.get('yf_interval','?')} {df_raw.attrs.get('yf_period','?')}"
                print(f"[DATA] {sym}: {len(df_tf)} rows (src {src})")
                data[sym] = df_tf
            time.sleep(0.2)
        return data

# ---------------- Orders / positions / broker ----------------
@dataclass
class BotOrder:
    symbol: str
    side: str
    limit_price: float
    qty: float
    submitted_at: datetime
    age_bars: int = 0
    valid_for_bars: int = 5
    filled: bool = False

@dataclass
class BotLeg:
    qty: float
    entry: float
    sl: float
    tp: float
    breakeven_moved: bool = False
    half_target_level: float = 0.0

@dataclass
class BotPosition:
    symbol: str
    side: str
    legs: List[BotLeg]
    opened_at: datetime
    def net_qty(self) -> float:
        return sum(leg.qty for leg in self.legs)

class BotBroker:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.equity = cfg.equity_start
        self.cash = cfg.equity_start
        self.positions: Dict[str, BotPosition] = {}
        self.open_orders: List[BotOrder] = []
        self.trades: List[Dict] = []
        self.daily_trade_counts: Dict[datetime, int] = {}
        self.daily_trade_counts_by_symbol: Dict[datetime, Dict[str,int]] = {}

    def day_key(self, ts: datetime) -> datetime:
        dt = ts.astimezone(timezone.utc)
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

    def can_open_new_trade(self, ts: datetime) -> bool:
        k = self.day_key(ts)
        return self.daily_trade_counts.get(k, 0) < self.cfg.max_new_trades_per_day

    def can_open_new_trade_symbol(self, ts: datetime, symbol: str) -> bool:
        k = self.day_key(ts)
        per = self.daily_trade_counts_by_symbol.get(k, {})
        return per.get(symbol, 0) < self.cfg.max_new_trades_per_day_per_symbol

    def register_new_trade(self, ts: datetime, symbol: str):
        k = self.day_key(ts)
        self.daily_trade_counts[k] = self.daily_trade_counts.get(k, 0) + 1
        per = self.daily_trade_counts_by_symbol.setdefault(k, {})
        per[symbol] = per.get(symbol, 0) + 1

    def submit_limit(self, symbol: str, side: str, limit_price: float, qty: float, ts: datetime, valid_bars: int):
        self.open_orders.append(BotOrder(symbol, side, limit_price, qty, ts, age_bars=0, valid_for_bars=valid_bars))

    def _record_trade(self, **kwargs):
        self.trades.append(kwargs)

    def position_for(self, symbol: str) -> Optional[BotPosition]:
        pos = self.positions.get(symbol)
        if pos and pos.net_qty() <= 0: return None
        return pos

    # ---- margin-aware open/close ----
    def open_position(self, symbol: str, side: str, qty: float, entry_px: float, sl: float, tp: float, ts: datetime):
        notional = abs(qty) * entry_px
        margin = notional * float(self.cfg.margin_rate)
        fee = abs(qty) * entry_px * (self.cfg.fee_bps / 10000.0)

        if self.cash < (margin + fee):
            self._record_trade(ts=ts.isoformat(), symbol=symbol, action="reject",
                               side=side, qty=qty, price=entry_px, fee=0.0, note="insufficient_margin")
            return

        self.cash -= (margin + fee)

        leg = BotLeg(qty=qty, entry=entry_px, sl=sl, tp=tp)
        leg.half_target_level = leg.entry + (tp - entry_px) * 0.5 if side == "buy" else leg.entry - (entry_px - tp) * 0.5
        ex_side = "long" if side == "buy" else "short"
        pos = self.positions.get(symbol)
        if pos is None or pos.net_qty() <= 0 or pos.side != ex_side:
            pos = BotPosition(symbol=symbol, side=ex_side, legs=[], opened_at=ts)
            self.positions[symbol] = pos
        pos.legs.append(leg)

        self.register_new_trade(ts, symbol)
        self._record_trade(ts=ts.isoformat(), symbol=symbol, action="open",
                           side=ex_side, qty=qty, price=entry_px, fee=fee, note=f"margin_reserved={margin:.2f}")

    def close_leg(self, symbol: str, leg: BotLeg, exit_px: float, ts: datetime, reason: str):
        pos_side = self.positions[symbol].side
        side = "sell" if pos_side == "long" else "buy"
        fill_px = apply_bps(exit_px, self.cfg.slippage_bps, side)
        fee = abs(leg.qty) * fill_px * (self.cfg.fee_bps / 10000.0)
        pnl = (fill_px - leg.entry) * leg.qty if pos_side == "long" else (leg.entry - fill_px) * leg.qty
        notional_entry = abs(leg.qty) * leg.entry
        margin_release = notional_entry * float(self.cfg.margin_rate)
        self.cash += (margin_release + pnl - fee)
        self._record_trade(ts=ts.isoformat(), symbol=symbol, action="close",
                           side=pos_side, qty=leg.qty, price=fill_px, fee=fee, pnl=pnl, note=reason)

    def close_symbol_if_empty(self, symbol: str):
        pos = self.positions.get(symbol)
        if pos and pos.net_qty() <= 0: del self.positions[symbol]

# ---------------- Risk & strategy ----------------
class BotRiskManager:
    def __init__(self, cfg: BotConfig): self.cfg = cfg
    def units_from_risk(self, equity: float, sl_distance: float) -> float:
        if sl_distance <= 0: return 0.0
        return round_to_lot((equity * self.cfg.risk_per_trade) / sl_distance, 1.0)

class EmaCrossoverStrategy:
    def __init__(self, cfg: BotConfig): self.cfg = cfg
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema_fast'] = ema(df['close'], self.cfg.ema_fast)
        df['ema_slow'] = ema(df['close'], self.cfg.ema_slow)
        df['atr'] = atr(df, self.cfg.atr_len)
        df['ready'] = (~df['ema_fast'].isna()) & (~df['ema_slow'].isna()) & (~df['atr'].isna())
        df['fast_gt_slow'] = df['ema_fast'] > df['ema_slow']
        df['fast_gt_slow_prev'] = df['fast_gt_slow'].shift(1)
        df['xc_long'] = (df['fast_gt_slow'] == True) & (df['fast_gt_slow_prev'] == False)
        df['xc_short'] = (df['fast_gt_slow'] == False) & (df['fast_gt_slow_prev'] == True)
        return df

# ---------------- Backtester ----------------
class Backtester:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.broker = BotBroker(cfg)
        self.sizer = BotRiskManager(cfg)
        self.strategy = EmaCrossoverStrategy(cfg)
        self.news = NewsFilter(cfg)
        self.data: Dict[str, pd.DataFrame] = {}
        self.equity_timeline: List = []

    def load_data_online(self):
        fetcher = DataFetcher(self.cfg)
        self.data = fetcher.fetch_all()
        for sym, df in self.data.items():
            if not df.empty:
                self.data[sym] = self.strategy.prepare(df)

    def integrate_news_if_enabled(self):
        if not self.cfg.use_news_api:
            return
        ts_all = []
        for df in self.data.values():
            if not df.empty:
                ts_all.append(df['timestamp'].iloc[0])
                ts_all.append(df['timestamp'].iloc[-1])
        if not ts_all:
            return
        start_dt, end_dt = min(ts_all), max(ts_all)
        tracked = sorted(list({ccy for s in self.cfg.symbols for ccy in self.cfg.currencies_by_symbol.get(s, [])}))
        self.news.fetch_high_impact_events(start_dt, end_dt, tracked)

    def _try_fill_orders(self, sym: str, row: pd.Series):
        remaining = []
        for order in self.broker.open_orders:
            if order.symbol != sym or order.filled:
                remaining.append(order); continue
            filled = within_bar_reaches(order.limit_price, row['low'], row['high']) or (
                     self.cfg.allow_same_bar_fill and within_bar_reaches(order.limit_price, row['low'], row['high']))
            if filled:
                atr_val = row['atr']
                if pd.isna(atr_val) or atr_val <= 0:
                    continue
                sl_dist = self.cfg.atr_mult_sl * atr_val
                if order.side == "buy":
                    sl = order.limit_price - sl_dist
                    tp = order.limit_price + sl_dist * self.cfg.tp_rr
                else:
                    sl = order.limit_price + sl_dist
                    tp = order.limit_price - sl_dist * self.cfg.tp_rr
                self.broker.open_position(sym, order.side, order.qty, order.limit_price, sl, tp, row['timestamp'])
                order.filled = True
            else:
                order.age_bars += 1
                if order.age_bars < order.valid_for_bars:
                    remaining.append(order)
        self.broker.open_orders = remaining

    def _manage_positions(self, sym: str, row: pd.Series):
        pos = self.broker.position_for(sym)
        if not pos: return
        new_legs: List[BotLeg] = []
        for leg in pos.legs:
            qty_rem = leg.qty
            if not leg.breakeven_moved:
                if pos.side == "long" and row['high'] >= leg.half_target_level:
                    cut = 0.5 * qty_rem
                    if cut > 0:
                        self.broker.close_leg(sym, BotLeg(qty=cut, entry=leg.entry, sl=leg.sl, tp=leg.tp),
                                              leg.half_target_level, row['timestamp'], "partial_50%_and_BE")
                        qty_rem -= cut
                    leg.sl = leg.entry
                    leg.breakeven_moved = True
                elif pos.side == "short" and row['low'] <= leg.half_target_level:
                    cut = 0.5 * qty_rem
                    if cut > 0:
                        self.broker.close_leg(sym, BotLeg(qty=cut, entry=leg.entry, sl=leg.sl, tp=leg.tp),
                                              leg.half_target_level, row['timestamp'], "partial_50%_and_BE")
                        qty_rem -= cut
                    leg.sl = leg.entry
                    leg.breakeven_moved = True

            if qty_rem > 0:
                hit_sl = (pos.side == "long" and row['low'] <= leg.sl) or (pos.side == "short" and row['high'] >= leg.sl)
                hit_tp = (pos.side == "long" and row['high'] >= leg.tp) or (pos.side == "short" and row['low'] <= leg.tp)
                if hit_sl and hit_tp:
                    self.broker.close_leg(sym, BotLeg(qty=qty_rem, entry=leg.entry, sl=leg.sl, tp=leg.tp),
                                          leg.sl, row['timestamp'], "stop")
                    qty_rem = 0.0
                elif hit_sl:
                    self.broker.close_leg(sym, BotLeg(qty=qty_rem, entry=leg.entry, sl=leg.sl, tp=leg.tp),
                                          leg.sl, row['timestamp'], "stop")
                    qty_rem = 0.0
                elif hit_tp:
                    self.broker.close_leg(sym, BotLeg(qty=qty_rem, entry=leg.entry, sl=leg.sl, tp=leg.tp),
                                          leg.tp, row['timestamp'], "target")
                    qty_rem = 0.0

            if qty_rem > 0:
                leg.qty = qty_rem
                new_legs.append(leg)

        pos.legs = new_legs
        if pos.net_qty() <= 0:
            self.broker.close_symbol_if_empty(sym)

    def _entry_logic(self, sym: str, row_prev: pd.Series, row: pd.Series):
        if not row['ready']: return
        ts = row['timestamp']
        if self.cfg.use_news_api and self.news.is_blackout(ts): return
        if not (self.broker.can_open_new_trade(ts) and self.broker.can_open_new_trade_symbol(ts, sym)): return
        if self.broker.position_for(sym): return

        if bool(row['xc_long']):
            sl_dist = self.cfg.atr_mult_sl * row['atr']
            if sl_dist and sl_dist > 0:
                qty = self.sizer.units_from_risk(self.broker.equity, sl_dist)
                # margin fit check
                req_margin = abs(qty) * float(row['close']) * float(self.cfg.margin_rate)
                if req_margin > self.broker.cash:
                    max_qty = (self.broker.cash / max(1e-12, float(self.cfg.margin_rate))) / float(row['close'])
                    qty = round_to_lot(max(0.0, max_qty), 1.0)
                if qty > 0:
                    self.broker.submit_limit(sym, "buy", float(row['close']), qty, ts, self.cfg.gtc_valid_bars)

        elif bool(row['xc_short']):
            sl_dist = self.cfg.atr_mult_sl * row['atr']
            if sl_dist and sl_dist > 0:
                qty = self.sizer.units_from_risk(self.broker.equity, sl_dist)
                req_margin = abs(qty) * float(row['close']) * float(self.cfg.margin_rate)
                if req_margin > self.broker.cash:
                    max_qty = (self.broker.cash / max(1e-12, float(self.cfg.margin_rate))) / float(row['close'])
                    qty = round_to_lot(max(0.0, max_qty), 1.0)
                if qty > 0:
                    self.broker.submit_limit(sym, "sell", float(row['close']), qty, ts, self.cfg.gtc_valid_bars)

    @staticmethod
    def _max_drawdown(equity_series: pd.Series):
        roll_max = equity_series.cummax()
        dd = equity_series / roll_max - 1.0
        mdd = dd.min() if len(dd) else 0.0
        end_idx = dd.idxmin() if len(dd) else None
        start_idx = equity_series[:end_idx].idxmax() if end_idx is not None else None
        peak_eq = float(equity_series.loc[start_idx]) if start_idx is not None else float('nan')
        trough_eq = float(equity_series.loc[end_idx]) if end_idx is not None else float('nan')
        return float(mdd), peak_eq, trough_eq

    def _save_performance(self, first_ts: Optional[datetime], last_ts: Optional[datetime]):
        ensure_dir(self.cfg.out_dir)
        tl = pd.DataFrame(self.broker.trades)
        if not tl.empty:
            tl.to_csv(os.path.join(self.cfg.out_dir, self.cfg.trade_log_csv), index=False)

        equity_df = pd.DataFrame(self.equity_timeline, columns=["timestamp","equity"])
        if not equity_df.empty:
            equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce")
            equity_df = equity_df.dropna(subset=["equity"])
        equity_df.to_csv(os.path.join(self.cfg.out_dir, self.cfg.equity_curve_csv), index=False)

        equity_start = float(self.cfg.equity_start)
        equity_end = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else equity_start

        years = 0.0
        if first_ts and last_ts and last_ts > first_ts:
            years = (last_ts - first_ts).total_seconds() / (365.25*24*3600)

        def safe_cagr(e0, e1, yrs):
            import math
            if yrs <= 0 or e0 <= 0 or e1 <= 0:
                return 0.0
            return math.exp(math.log(e1/e0) / yrs) - 1.0

        cagr = safe_cagr(equity_start, equity_end, years)

        try:
            if equity_df.empty:
                sharpe = 0.0
                mdd, mdd_peak_eq, mdd_trough_eq = 0.0, float("nan"), float("nan")
            else:
                daily = (
                    equity_df.set_index("timestamp")["equity"]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .resample("1D").last()
                    .dropna()
                )
                daily_ret = daily.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0

                roll_max = daily.cummax()
                dd = daily / roll_max - 1.0
                mdd = float(dd.min()) if len(dd) else 0.0
                end_idx = dd.idxmin() if len(dd) else None
                start_idx = daily[:end_idx].idxmax() if end_idx is not None else None
                mdd_peak_eq = float(daily.loc[start_idx]) if start_idx is not None else float("nan")
                mdd_trough_eq = float(daily.loc[end_idx]) if end_idx is not None else float("nan")
        except Exception:
            sharpe, mdd, mdd_peak_eq, mdd_trough_eq = 0.0, 0.0, float("nan"), float("nan")

        eq_png = None
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(equity_df["timestamp"], equity_df["equity"])
            ax.set_title("Equity Curve")
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Equity")
            fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(os.path.join(self.cfg.out_dir, self.cfg.equity_curve_png), dpi=140)
            plt.close(fig)
            eq_png = self.cfg.equity_curve_png
        except Exception:
            pass

        def _safe_num(x, ndigits=None):
            try:
                xv = float(x)
                if not np.isfinite(xv):
                    return None
                return round(xv, ndigits) if ndigits is not None else xv
            except Exception:
                return None

        report = {
            "equity_start": _safe_num(equity_start, 2),
            "equity_end": _safe_num(equity_end, 2),
            "CAGR": _safe_num(cagr, 6),
            "Sharpe_daily": _safe_num(sharpe, 4),
            "max_drawdown": _safe_num(mdd, 6),
            "max_dd_peak_equity": _safe_num(mdd_peak_eq, 2),
            "max_dd_trough_equity": _safe_num(mdd_trough_eq, 2),
            "num_opens": int((tl["action"]=="open").sum() if not tl.empty else 0),
            "num_closes": int((tl["action"]=="close").sum() if not tl.empty else 0),
            "wins_closed": int(((tl["action"]=="close") & (tl["pnl"]>0)).sum() if not tl.empty and "pnl" in tl.columns else 0),
            "win_rate_closed_pct": _safe_num(100.0 * ( ((tl["action"]=="close") & (tl["pnl"]>0)).sum() / max(1,(tl["action"]=="close").sum()) ) if not tl.empty else 0.0, 2),
            "realized_pnl": _safe_num(tl.loc[tl["action"]=="close","pnl"].sum() if not tl.empty and "pnl" in tl.columns else 0.0, 2),
            "trade_log_csv": self.cfg.trade_log_csv,
            "equity_curve_csv": self.cfg.equity_curve_csv,
            "equity_curve_png": eq_png
        }
        with open(os.path.join(self.cfg.out_dir, self.cfg.performance_json), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))

    def run(self):
        # 1) Fetch + prep
        self.load_data_online()

        # DIAGNOSTICS
        print("=== DATA DIAGNOSTICS (3m) ===")
        for sym, df in self.data.items():
            ready = int(df["ready"].sum()) if not df.empty else 0
            xl = int(df["xc_long"].sum()) if not df.empty else 0
            xs = int(df["xc_short"].sum()) if not df.empty else 0
            print(f"{sym:7s} | rows={len(df):5d} | ready={ready:5d} | xc_long={xl:4d} | xc_short={xs:4d}")
        print("===============================")

        # 2) News
        self.integrate_news_if_enabled()
        print(f"[NEWS] High-impact events loaded: {len(self.news.events)} (blocks new entries +{self.cfg.news_resume_minutes}m)")

        # 3) Merge timeline
        all_rows: List = []
        for sym, df in self.data.items():
            for i in range(len(df)):
                all_rows.append((df.at[i,'timestamp'], sym, i))
        all_rows.sort(key=lambda x: x[0])

        last_prices: Dict[str, float] = {sym: np.nan for sym in self.data.keys()}
        first_ts = all_rows[0][0] if all_rows else None
        last_ts = all_rows[-1][0] if all_rows else None

        # 4) Sim
        for ts, sym, i in all_rows:
            df = self.data[sym]
            row = df.iloc[i]
            last_prices[sym] = row['close']

            self._try_fill_orders(sym, row)
            self._manage_positions(sym, row)
            if i > 0:
                self._entry_logic(sym, df.iloc[i-1], row)

            # Mark-to-market equity
            unreal = 0.0
            for s, pos in self.broker.positions.items():
                px = last_prices.get(s, np.nan)
                if np.isnan(px): continue
                for leg in pos.legs:
                    unreal += ((px - leg.entry) * leg.qty) if pos.side == "long" else ((leg.entry - px) * leg.qty)
            self.broker.equity = self.broker.cash + unreal
            self.equity_timeline.append((row['timestamp'], float(self.broker.equity)))

        # 5) Close leftovers at last price
        if last_ts is None: last_ts = datetime.now(timezone.utc)
        for s, pos in list(self.broker.positions.items()):
            px = last_prices.get(s, np.nan)
            if np.isnan(px): continue
            for leg in pos.legs:
                self.broker.close_leg(s, leg, px, last_ts, "eod_close")
            self.broker.close_symbol_if_empty(s)

        # 6) Save report
        self._save_performance(first_ts, last_ts)

# ===== User-selectable symbols support =====
DEFAULT_YAHOO_PRIMARY = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "XAUUSD": "XAUUSD=X",   # spot gold (fallbacks will include futures)
}
DEFAULT_YAHOO_FALLBACKS = {
    "XAUUSD": ["GC=F", "XAU=X", "GLD"],
}

def parse_symbols_from_env(env_var: str = "TRADE_SYMBOLS") -> list[str]:
    """
    Read a comma-separated list like: EURUSD, GBPUSD, XAUUSD
    Returns uppercase symbols, or [] if not set.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return []
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    # Deduplicate, preserve order
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def build_yahoo_maps(symbols: list[str]) -> tuple[dict, dict]:
    """
    For each requested symbol, choose a primary Yahoo ticker and fallbacks.
    Unknown symbols default to SYMBOL=X (Yahoo FX/CFD convention).
    """
    primary = {}
    fallbacks = {}
    for s in symbols:
        primary[s] = DEFAULT_YAHOO_PRIMARY.get(s, f"{s}=X")
        if s in DEFAULT_YAHOO_FALLBACKS:
            fallbacks[s] = DEFAULT_YAHOO_FALLBACKS[s]
    return primary, fallbacks

# ---------------- Run ----------------
if __name__ == "__main__":
    # 1) Read user-chosen symbols from environment (e.g., .env or launch.json)
    user_syms = parse_symbols_from_env("TRADE_SYMBOLS")

    # 2) Fallback to original defaults if not provided
    default_syms = ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","XAUUSD"]
    symbols = user_syms if user_syms else default_syms

    # 3) Build Yahoo mappings for exactly those symbols
    y_primary, y_fallbacks = build_yahoo_maps(symbols)

    cfg = BotConfig(
        symbols=symbols,
        yahoo_primary=y_primary,
        yahoo_fallbacks=y_fallbacks,

        equity_start=10_000.0,
        risk_per_trade=0.02,
        max_new_trades_per_day=3,
        max_new_trades_per_day_per_symbol=3,
        gtc_valid_bars=5,
        allow_same_bar_fill=False,
        use_news_api=True,
        news_resume_minutes=10,
        fx_leverage=30,
        margin_rate=None,   # default to 1/fx_leverage
        out_dir="."
    )
    Backtester(cfg).run()


