# EMA Crossover Trading Bot (Backtester, Margin-Aware)

A Python-based backtester for an **EMA crossover strategy** on FX pairs and Gold (XAUUSD), designed for 3-minute candles.  
Includes risk management, margin sizing, limit order simulation, partial take-profit with breakeven shift, and a **high-impact news blackout** (FinancialModelingPrep API).

---

## ✨ Features

- **EMA crossover strategy** (30 vs 50 EMA by default)  
- **Risk-based position sizing** (2% equity risk/trade, configurable)  
- **Margin/leverage model** (default: 30x leverage)  
- **Limit order simulation** with configurable validity (GTC bars)  
- **Partial take-profit** (50% off at 1R, move stop to breakeven)  
- **News filter** via [FinancialModelingPrep Economic Calendar](https://financialmodelingprep.com/developer/docs/)  
- **Outputs**:  
  - `trades_log.csv` — detailed trades log  
  - `equity_curve.csv` — equity history  
  - `equity_curve.png` — performance chart  
  - `performance_report.json` — summary stats (CAGR, Sharpe, MDD, win-rate, etc.)

---

## 📦 Requirements

- Python 3.9+  
- Packages (installed via `requirements.txt`):
  - `pandas`
  - `numpy`
  - `yfinance`
  - `matplotlib`
  - `requests`

---

## ⚙️ Setup (VS Code Workflow)

1. **Clone repo (or open project folder)** in VS Code.  
2. **Create environment**:  
   - Open Command Palette (Ctrl+Shift+P) → **Python: Create Environment** → choose *venv* and interpreter → select `requirements.txt`.  
3. **Create `.env` file** (at project root) and add your API key + symbols:  

   ```env
   FMP_API_KEY=your_real_api_key_here
   TRADE_SYMBOLS=XAUUSD
   
FMP_API_KEY is required for the news filter (falls back to demo if missing).

TRADE_SYMBOLS is optional. If not set, the bot defaults to all 8 symbols (EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, XAUUSD).
Examples:

Gold only

TRADE_SYMBOLS=XAUUSD


Majors + Gold

TRADE_SYMBOLS=EURUSD,GBPUSD,USDJPY,XAUUSD

  "equity_start": 10000.0,
  "equity_end": 10421.3,
  "CAGR": 0.134562,
  "Sharpe_daily": 1.21,
  "max_drawdown": -0.062,
  "wins_closed": 18,
  "win_rate_closed_pct": 56.25,
  "realized_pnl": 421.3,
  "trade_log_csv": "trades_log.csv",
  "equity_curve_csv": "equity_curve.csv",
  "equity_curve_png": "equity_curve.png"

  Notes

Default symbols: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, XAUUSD

Override with .env → TRADE_SYMBOLS=...

Uses 1m Yahoo Finance data (stitched into 3m bars). Falls back to 5m/60d if needed.

Designed as a backtester / simulator only (not connected to a broker API).

🔮 Future Enhancements

Planned features for upcoming versions:

EMA or SMA choice — users will be able to select whether the strategy runs on Exponential Moving Averages (EMA) or Simple Moving Averages (SMA) directly via .env or config.

Customizable strategy parameters (period lengths, risk %, ATR multiplier) through .env.

More visualization (equity curve with drawdown shading, trade entry/exit markers).

Modular outputs for easier strategy comparison.
