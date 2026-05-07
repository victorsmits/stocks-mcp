from fastmcp import FastMCP
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timedelta

mcp = FastMCP("Stock Information MCP Server")

# ─── Utilitaire ticker ────────────────────────────────────────────────────────

def _normalize_ticker(ticker: str) -> str:
    exchange_map = {
        "EPA": "PA", "AMS": "AS", "EBR": "BR",
        "ETR": "DE", "LON": "L", "NYSE": "", "NASDAQ": "",
    }
    if ":" in ticker:
        symbol, exchange = ticker.split(":", 1)
        suffix = exchange_map.get(exchange.upper())
        if suffix is None:
            return ticker
        return f"{symbol}.{suffix}" if suffix else symbol
    return ticker

def _r(v, n=2):
    """Round float safely."""
    try:
        return round(float(v), n) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
    except:
        return None

# ════════════════════════════════════════════════════════════════════════════════
# BLOC 1 — DONNÉES ACTION
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_stock_info(ticker: str) -> dict:
    """Prix actuel, valorisation, dividende, secteur d'une action.
    Exemples: NVDA, AAPL, AIR.PA, SU.PA, ASML.AS, ABI.BR"""
    try:
        ticker = _normalize_ticker(ticker)
        info = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        dy = info.get("dividendYield")
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "price": price,
            "currency": info.get("currency", ""),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "50d_avg": info.get("fiftyDayAverage"),
            "200d_avg": info.get("twoHundredDayAverage"),
            "dividend_yield": f"{round(dy*100,2)}%" if dy else "N/A",
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),
            "beta": info.get("beta"),
            "short_ratio": info.get("shortRatio"),
            "float_shares": info.get("floatShares"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "description": (info.get("longBusinessSummary") or "")[:500],
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def get_price_history(ticker: str, period: str = "3mo", interval: str = "1d") -> dict:
    """Historique OHLCV. Period: 1d/5d/1mo/3mo/6mo/1y/2y/5y. Interval: 1d/1wk/1mo"""
    try:
        ticker = _normalize_ticker(ticker)
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist.empty:
            return {"ticker": ticker, "error": "No data"}
        data = [
            {"date": d.strftime("%Y-%m-%d"), "open": _r(r.Open), "high": _r(r.High),
             "low": _r(r.Low), "close": _r(r.Close), "volume": int(r.Volume)}
            for d, r in hist.iterrows()
        ]
        return {"ticker": ticker, "period": period, "interval": interval,
                "count": len(data), "history": data}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def get_technical_indicators(ticker: str) -> dict:
    """Indicateurs techniques complets sur 1 an de données.

    Inclut : RSI, MACD, Bollinger, Stochastique, Williams %R, ADX, CCI,
    OBV, MFI, ATR, Ichimoku, Parabolic SAR, moyennes mobiles, performance.
    """
    try:
        ticker = _normalize_ticker(ticker)
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return {"ticker": ticker, "error": "No data"}

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]
        current = close.iloc[-1]

        # ── Moyennes mobiles ──────────────────────────────────────────────────
        ma9   = close.rolling(9).mean().iloc[-1]
        ma20  = close.rolling(20).mean().iloc[-1]
        ma50  = close.rolling(50).mean().iloc[-1]
        ma100 = close.rolling(100).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        ema9  = close.ewm(span=9,  adjust=False).mean().iloc[-1]
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        # ── RSI (14) ──────────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = (100 - 100 / (1 + gain / loss)).iloc[-1]

        # ── Stochastique %K/%D (14,3) ─────────────────────────────────────────
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stoch_k = ((close - low14) / (high14 - low14) * 100)
        stoch_d = stoch_k.rolling(3).mean()
        sk, sd  = stoch_k.iloc[-1], stoch_d.iloc[-1]

        # ── Williams %R (14) ──────────────────────────────────────────────────
        willr = ((high14 - close) / (high14 - low14) * -100).iloc[-1]

        # ── MACD (12,26,9) ────────────────────────────────────────────────────
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        ml, sl, mh = macd.iloc[-1], signal.iloc[-1], (macd - signal).iloc[-1]

        # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = (bb_mid + 2*bb_std).iloc[-1]
        bb_lower = (bb_mid - 2*bb_std).iloc[-1]
        bb_mid_v = bb_mid.iloc[-1]
        bb_pos   = (current - bb_lower) / (bb_upper - bb_lower) * 100

        # ── ATR (14) ──────────────────────────────────────────────────────────
        tr  = pd.concat([high - low,
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr / current * 100

        # ── ADX (14) ──────────────────────────────────────────────────────────
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr14     = tr.rolling(14).sum()
        plus_di  = 100 * plus_dm.rolling(14).sum()  / tr14
        minus_di = 100 * minus_dm.rolling(14).sum() / tr14
        dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di))
        adx      = dx.rolling(14).mean().iloc[-1]
        pdi, mdi = plus_di.iloc[-1], minus_di.iloc[-1]

        # ── CCI (20) ──────────────────────────────────────────────────────────
        tp  = (high + low + close) / 3
        cci = ((tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())).iloc[-1]

        # ── OBV ───────────────────────────────────────────────────────────────
        obv = (volume * close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
        obv_trend = "Rising" if obv.iloc[-1] > obv.iloc[-5] else "Falling"

        # ── MFI (14) ──────────────────────────────────────────────────────────
        mf    = tp * volume
        pos_mf = mf.where(tp > tp.shift(1), 0)
        neg_mf = mf.where(tp < tp.shift(1), 0)
        mfr   = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum()
        mfi   = (100 - 100 / (1 + mfr)).iloc[-1]

        # ── Parabolic SAR (simplifié) ─────────────────────────────────────────
        # Signal directionnel basé sur la position par rapport EMA
        sar_bullish = current > ema20 and ema20 > ema50

        # ── Ichimoku (9,26,52) ────────────────────────────────────────────────
        h9  = high.rolling(9).max();  l9  = low.rolling(9).min()
        h26 = high.rolling(26).max(); l26 = low.rolling(26).min()
        h52 = high.rolling(52).max(); l52 = low.rolling(52).min()
        tenkan  = ((h9 + l9) / 2).iloc[-1]
        kijun   = ((h26 + l26) / 2).iloc[-1]
        span_a  = ((h9 + l9 + h26 + l26) / 4).iloc[-1]
        span_b  = ((h52 + l52) / 2).iloc[-1]
        kumo_bullish = span_a > span_b and current > max(span_a, span_b)

        # ── Performance ───────────────────────────────────────────────────────
        def perf(n): return _r((current / close.iloc[-n] - 1)*100) if len(close) > n else None

        # ── Résumé signal ─────────────────────────────────────────────────────
        bull_signals = sum([
            current > ma50, current > ma200, ma50 > ma200,
            rsi > 50, ml > sl, sk > sd, pdi > mdi,
            sar_bullish, kumo_bullish, obv_trend == "Rising"
        ])
        signal_summary = "Strong Bull" if bull_signals >= 8 else \
                         "Bull"        if bull_signals >= 6 else \
                         "Neutral"     if bull_signals >= 4 else \
                         "Bear"        if bull_signals >= 2 else "Strong Bear"

        return {
            "ticker": ticker,
            "current_price": _r(current),
            "signal_summary": signal_summary,
            "bull_signals_out_of_10": bull_signals,

            "moving_averages": {
                "MA9": _r(ma9),   "EMA9":  _r(ema9),
                "MA20": _r(ma20), "EMA20": _r(ema20),
                "MA50": _r(ma50), "EMA50": _r(ema50),
                "MA100": _r(ma100), "MA200": _r(ma200),
                "above_MA50": bool(current > ma50),
                "above_MA200": bool(current > ma200),
                "golden_cross_50_200": bool(ma50 > ma200),
                "death_cross_50_200":  bool(ma50 < ma200),
            },
            "rsi_14": {
                "value":  _r(rsi),
                "signal": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
            },
            "stochastic_14_3": {
                "K": _r(sk), "D": _r(sd),
                "signal": "Overbought" if sk > 80 else ("Oversold" if sk < 20 else "Neutral"),
                "k_above_d": bool(sk > sd),
            },
            "williams_r_14": {
                "value": _r(willr),
                "signal": "Overbought" if willr > -20 else ("Oversold" if willr < -80 else "Neutral"),
            },
            "macd_12_26_9": {
                "macd":    _r(ml),
                "signal":  _r(sl),
                "hist":    _r(mh),
                "bullish": bool(ml > sl),
            },
            "bollinger_20_2": {
                "upper": _r(bb_upper), "mid": _r(bb_mid_v), "lower": _r(bb_lower),
                "position_%": _r(bb_pos),
                "squeeze": bool(bb_upper - bb_lower < atr * 2),
            },
            "atr_14": {
                "value": _r(atr),
                "pct_of_price_%": _r(atr_pct),
            },
            "adx_14": {
                "adx":   _r(adx),
                "plus_di":  _r(pdi),
                "minus_di": _r(mdi),
                "trend_strength": "Strong" if adx > 25 else ("Weak" if adx < 20 else "Moderate"),
                "bullish_di": bool(pdi > mdi),
            },
            "cci_20": {
                "value": _r(cci),
                "signal": "Overbought" if cci > 100 else ("Oversold" if cci < -100 else "Neutral"),
            },
            "obv": {
                "trend": obv_trend,
                "value": int(obv.iloc[-1]),
            },
            "mfi_14": {
                "value":  _r(mfi),
                "signal": "Overbought" if mfi > 80 else ("Oversold" if mfi < 20 else "Neutral"),
            },
            "ichimoku": {
                "tenkan":  _r(tenkan),
                "kijun":   _r(kijun),
                "span_a":  _r(span_a),
                "span_b":  _r(span_b),
                "bullish_cloud":   bool(kumo_bullish),
                "price_above_cloud": bool(current > max(span_a, span_b)),
            },
            "parabolic_sar": {
                "bullish": bool(sar_bullish),
            },
            "performance_%": {
                "5d":  perf(5),  "1m":  perf(22),
                "3m":  perf(66), "6m":  perf(126),
                "1y":  perf(252),
            },
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def get_financials(ticker: str) -> dict:
    """Compte de résultat, bilan, cash flow des 4 derniers exercices."""
    try:
        ticker = _normalize_ticker(ticker)
        stock = yf.Ticker(ticker)
        def df_to_dict(df):
            if df is None or df.empty:
                return {}
            df.columns = [str(c)[:10] for c in df.columns]
            return {str(k): {c: (None if pd.isna(v) else round(float(v)/1e6, 1))
                             for c, v in row.items()}
                    for k, row in df.iterrows()}
        return {
            "ticker": ticker, "note": "Values in millions",
            "income_statement": df_to_dict(stock.financials),
            "balance_sheet":    df_to_dict(stock.balance_sheet),
            "cash_flow":        df_to_dict(stock.cashflow),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def get_earnings(ticker: str) -> dict:
    """EPS historique, estimations analystes, surprise earnings."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)
        info   = stock.info
        hist   = stock.earnings_history
        rows   = []
        if hist is not None and not hist.empty:
            for _, r in hist.head(8).iterrows():
                rows.append({
                    "date":          str(r.get("quarter", "")),
                    "eps_est":       r.get("epsEstimate"),
                    "eps_act":       r.get("epsActual"),
                    "surprise_pct":  r.get("surprisePercent"),
                })
        return {
            "ticker": ticker,
            "trailing_eps":        info.get("trailingEps"),
            "forward_eps":         info.get("forwardEps"),
            "earnings_growth_yoy": info.get("earningsGrowth"),
            "revenue_growth_yoy":  info.get("revenueGrowth"),
            "next_earnings_date":  str(info.get("earningsTimestamp", "")),
            "earnings_history":    rows,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def get_dividends(ticker: str) -> dict:
    """Historique des dividendes + CAGR 5 ans."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)
        info   = stock.info
        divs   = stock.dividends
        history = []
        if not divs.empty:
            for d, v in divs.tail(20).items():
                history.append({"date": d.strftime("%Y-%m-%d"), "amount": round(float(v), 4)})
        cagr = None
        if not divs.empty:
            annual = divs.resample("YE").sum()
            if len(annual) >= 5 and annual.iloc[-5] > 0:
                cagr = round(((annual.iloc[-1] / annual.iloc[-5]) ** (1/5) - 1) * 100, 2)
        return {
            "ticker": ticker,
            "dividend_rate":    info.get("dividendRate"),
            "dividend_yield":   f"{round((info.get('dividendYield') or 0)*100,2)}%",
            "payout_ratio":     info.get("payoutRatio"),
            "ex_dividend_date": str(info.get("exDividendDate", "")),
            "5y_div_cagr_%":    cagr,
            "history":          history,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def get_analyst_recommendations(ticker: str) -> dict:
    """Recommandations buy/hold/sell + objectifs de cours."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)
        info   = stock.info
        recs   = stock.recommendations
        recent = []
        if recs is not None and not recs.empty:
            for _, r in recs.tail(10).iterrows():
                recent.append({"firm": r.get("Firm",""), "action": r.get("Action",""), "grade": r.get("To Grade","")})
        price = info.get("currentPrice") or info.get("regularMarketPrice", 1)
        target = info.get("targetMeanPrice")
        return {
            "ticker": ticker,
            "current_price":     price,
            "target_mean_price": target,
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price":  info.get("targetLowPrice"),
            "upside_%":          _r((target/price - 1)*100) if target and price else None,
            "analyst_count":     info.get("numberOfAnalystOpinions"),
            "recommendation":    info.get("recommendationKey","").upper(),
            "recent_changes":    recent,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def get_holders(ticker: str) -> dict:
    """Actionnaires institutionnels et transactions insiders."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)
        inst   = stock.institutional_holders
        inst_list = []
        if inst is not None and not inst.empty:
            for _, r in inst.head(10).iterrows():
                inst_list.append({"holder": r.get("Holder",""), "shares": int(r.get("Shares",0)), "pct_held": r.get("% Out")})
        insiders = stock.insider_transactions
        insider_list = []
        if insiders is not None and not insiders.empty:
            for _, r in insiders.head(10).iterrows():
                insider_list.append({
                    "date": str(r.get("Start Date",""))[:10], "insider": r.get("Insider",""),
                    "position": r.get("Position",""), "transaction": r.get("Transaction",""),
                    "shares": r.get("Shares",0), "value": r.get("Value",0),
                })
        info = stock.info
        return {
            "ticker": ticker,
            "insider_ownership_%":       info.get("heldPercentInsiders"),
            "institutional_ownership_%": info.get("heldPercentInstitutions"),
            "top_institutions": inst_list,
            "recent_insider_transactions": insider_list,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


@mcp.tool()
async def compare_stocks(tickers: list[str], metric: str = "all") -> dict:
    """Compare plusieurs actions. metric: all | price | valuation | growth | dividends"""
    try:
        tickers = [_normalize_ticker(t) for t in tickers]
        result  = {}
        for t in tickers:
            info  = yf.Ticker(t).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            dy    = info.get("dividendYield", 0) or 0
            row   = {}
            if metric in ("all","price"):
                row.update({"price": price, "52w_high": info.get("fiftyTwoWeekHigh"),
                            "52w_low": info.get("fiftyTwoWeekLow"),
                            "from_52w_high_%": _r((price/info.get("fiftyTwoWeekHigh",1)-1)*100) if price else None})
            if metric in ("all","valuation"):
                row.update({"market_cap_B": _r((info.get("marketCap") or 0)/1e9,1),
                            "pe_trailing": info.get("trailingPE"), "pe_forward": info.get("forwardPE"),
                            "pb": info.get("priceToBook"), "ps": info.get("priceToSalesTrailing12Months"),
                            "ev_ebitda": info.get("enterpriseToEbitda")})
            if metric in ("all","growth"):
                row.update({"revenue_growth_%": _r((info.get("revenueGrowth") or 0)*100),
                            "earnings_growth_%": _r((info.get("earningsGrowth") or 0)*100),
                            "profit_margin_%": _r((info.get("profitMargins") or 0)*100),
                            "roe_%": _r((info.get("returnOnEquity") or 0)*100)})
            if metric in ("all","dividends"):
                row.update({"div_yield_%": _r(dy*100), "div_rate": info.get("dividendRate"),
                            "payout_ratio": info.get("payoutRatio")})
            result[t] = row
        return {"comparison": result, "metric": metric}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_upcoming_earnings(tickers: list[str]) -> dict:
    """Prochaines dates d'earnings pour une liste de tickers."""
    result = {}
    for t in tickers:
        tn = _normalize_ticker(t)
        try:
            info = yf.Ticker(tn).info
            result[t] = {"next_earnings_date": str(info.get("earningsTimestamp","Unknown")),
                         "fiscal_year_end": info.get("fiscalYearEnd","")}
        except Exception as e:
            result[t] = {"error": str(e)}
    return {"upcoming_earnings": result}


@mcp.tool()
async def analyze_portfolio(holdings: dict) -> dict:
    """Analyse complète d'un portefeuille.
    holdings = {"NVDA": 85, "AMD": 75, "AIR.PA": 48, ...}"""
    portfolio = {}
    total_value = 0.0
    errors = []
    for ticker, shares in holdings.items():
        tn = _normalize_ticker(ticker)
        try:
            info  = yf.Ticker(tn).info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
            value = price * shares
            total_value += value
            portfolio[ticker] = {
                "shares": shares, "price": price, "value": round(value, 2),
                "currency": info.get("currency",""), "sector": info.get("sector","Unknown"),
                "name": info.get("shortName", ticker), "beta": info.get("beta"),
                "pe": info.get("trailingPE"), "dy_%": _r((info.get("dividendYield") or 0)*100),
            }
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
    for t in portfolio:
        portfolio[t]["allocation_%"] = _r(portfolio[t]["value"] / total_value * 100) if total_value else 0
    sectors = {}
    for t, d in portfolio.items():
        s = d.get("sector","Unknown")
        sectors[s] = round(sectors.get(s,0) + (d.get("allocation_%") or 0), 2)
    betas = [d["beta"]*d["allocation_%"] for d in portfolio.values() if d.get("beta")]
    w_beta = sum(betas)/100 if betas else None
    w_dy   = sum(d["dy_%"]*d["allocation_%"] for d in portfolio.values())/100
    return {
        "total_value": round(total_value, 2),
        "positions_count": len(portfolio),
        "sector_allocation_%": dict(sorted(sectors.items(), key=lambda x: -x[1])),
        "weighted_beta": _r(w_beta),
        "weighted_div_yield_%": _r(w_dy),
        "positions": dict(sorted(portfolio.items(), key=lambda x: -x[1]["value"])),
        "errors": errors,
    }


@mcp.tool()
async def screen_stocks(tickers: list[str], min_div_yield: float = 0,
                        max_pe: float = 999, min_roe: float = 0) -> dict:
    """Filtre une liste de tickers selon critères fondamentaux."""
    results = []
    for t in tickers:
        tn = _normalize_ticker(t)
        try:
            info = yf.Ticker(tn).info
            pe  = info.get("trailingPE") or 999
            roe = (info.get("returnOnEquity") or 0)*100
            dy  = (info.get("dividendYield") or 0)*100
            if pe <= max_pe and roe >= min_roe and dy >= min_div_yield:
                results.append({
                    "ticker": t, "name": info.get("shortName",t),
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "pe": _r(pe) if pe < 999 else None, "roe_%": _r(roe),
                    "div_yield_%": _r(dy), "market_cap_B": _r((info.get("marketCap") or 0)/1e9,1),
                })
        except:
            pass
    results.sort(key=lambda x: x.get("roe_%",0), reverse=True)
    return {"screener_results": results, "count": len(results),
            "filters": {"max_pe": max_pe, "min_roe_%": min_roe, "min_div_yield_%": min_div_yield}}


# ════════════════════════════════════════════════════════════════════════════════
# BLOC 2 — MARCHÉ & MACRO
# ════════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_market_overview() -> dict:
    """Indices mondiaux, VIX, matières premières, forex, taux."""
    indices = {
        "S&P 500":        "^GSPC", "Nasdaq 100":   "^NDX",
        "Dow Jones":      "^DJI",  "Russell 2000": "^RUT",
        "VIX":            "^VIX",  "CAC 40":       "^FCHI",
        "DAX":            "^GDAXI","Euro Stoxx 50": "^STOXX50E",
        "Euro Stoxx 600": "^STOXX","AEX":           "^AEX",
        "BEL 20":         "^BFX",  "FTSE 100":      "^FTSE",
        "Nikkei 225":     "^N225", "Hang Seng":     "^HSI",
        "Gold":           "GC=F",  "Silver":        "SI=F",
        "Oil WTI":        "CL=F",  "Oil Brent":     "BZ=F",
        "Natural Gas":    "NG=F",  "Copper":        "HG=F",
        "EUR/USD":        "EURUSD=X","GBP/USD":     "GBPUSD=X",
        "USD/JPY":        "JPY=X", "EUR/CHF":       "EURCHF=X",
        "Bitcoin":        "BTC-USD","Ethereum":     "ETH-USD",
        "10Y US Treasury":"^TNX",  "10Y DE Bund":   "^TNX",
        "2Y US Treasury": "^IRX",
    }
    result = {}
    for name, sym in indices.items():
        try:
            fi    = yf.Ticker(sym).fast_info
            price = getattr(fi, "last_price", None)
            prev  = getattr(fi, "previous_close", None)
            chg   = _r((price/prev-1)*100) if price and prev else None
            result[name] = {"symbol": sym, "price": _r(price), "change_%": chg}
        except:
            result[name] = {"symbol": sym, "price": None, "change_%": None}
    return {"market_overview": result, "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}


@mcp.tool()
async def get_sector_performance(period: str = "1mo") -> dict:
    """Performance des 11 secteurs SPDR. Period: 1d/5d/1mo/3mo/6mo/1y"""
    sectors = {
        "Technology":         "XLK", "Healthcare":        "XLV",
        "Financials":         "XLF", "Consumer Disc.":    "XLY",
        "Consumer Staples":   "XLP", "Industrials":       "XLI",
        "Energy":             "XLE", "Materials":         "XLB",
        "Real Estate":        "XLRE","Utilities":         "XLU",
        "Communication Svcs": "XLC",
    }
    result = {}
    for name, sym in sectors.items():
        try:
            hist = yf.Ticker(sym).history(period=period)
            if not hist.empty:
                perf = _r((hist["Close"].iloc[-1]/hist["Close"].iloc[0]-1)*100)
                result[name] = {"etf": sym, "performance_%": perf}
        except:
            result[name] = {"etf": sym, "performance_%": None}
    return {"sector_performance": dict(sorted(result.items(), key=lambda x: x[1].get("performance_%") or -999, reverse=True)),
            "period": period}


@mcp.tool()
async def get_fear_greed_index() -> dict:
    """CNN Fear & Greed Index (indicateur sentiment de marché, 0=peur extrême, 100=cupidité extrême)."""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://edition.cnn.com/"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()
        fg = data.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating", "")
        prev_close = data.get("fear_and_greed_historical", {}).get("data", [{}])
        history = []
        for point in prev_close[-30:]:
            if "x" in point and "y" in point:
                history.append({
                    "date":   datetime.fromtimestamp(point["x"]/1000).strftime("%Y-%m-%d"),
                    "score":  _r(point["y"]),
                    "rating": point.get("rating",""),
                })
        return {
            "score":   _r(score),
            "rating":  rating,
            "interpretation": (
                "Extreme Fear"  if score and score < 25 else
                "Fear"          if score and score < 45 else
                "Neutral"       if score and score < 55 else
                "Greed"         if score and score < 75 else
                "Extreme Greed"
            ),
            "history_30d": history,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_macro_indicators() -> dict:
    """Indicateurs macro-économiques clés via FRED (Federal Reserve).
    Taux Fed, inflation CPI, chômage, PIB, M2, spread 10Y-2Y, etc."""
    # Données via yfinance pour les taux et ETFs proxy macro
    indicators = {
        "10Y_US_yield":     "^TNX",
        "2Y_US_yield":      "^IRX",
        "30Y_US_yield":     "^TYX",
        "TIP_inflation":    "TIP",     # ETF TIPS proxy inflation
        "DXY_dollar":       "DX-Y.NYB",
        "HYG_high_yield":   "HYG",     # Credit spreads proxy
        "LQD_investment_gr":"LQD",
        "GLD_gold":         "GLD",
        "TLT_long_bond":    "TLT",
        "SHY_short_bond":   "SHY",
        "USO_oil":          "USO",
    }
    result = {}
    for name, sym in indicators.items():
        try:
            hist  = yf.Ticker(sym).history(period="3mo")
            fi    = yf.Ticker(sym).fast_info
            price = getattr(fi, "last_price", None)
            prev  = getattr(fi, "previous_close", None)
            chg_d = _r((price/prev-1)*100) if price and prev else None
            chg_3m = _r((hist["Close"].iloc[-1]/hist["Close"].iloc[0]-1)*100) if not hist.empty else None
            result[name] = {"symbol": sym, "price": _r(price), "change_1d_%": chg_d, "change_3m_%": chg_3m}
        except:
            result[name] = {"symbol": sym, "price": None}

    # Spread 10Y-2Y
    try:
        t10 = result.get("10Y_US_yield",{}).get("price") or 0
        t2  = result.get("2Y_US_yield",{}).get("price") or 0
        result["yield_curve_10Y_2Y_spread"] = _r(t10 - t2)
        result["yield_curve_inverted"] = t10 < t2
    except:
        pass

    return {"macro_indicators": result, "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}


# ════════════════════════════════════════════════════════════════════════════════
# BLOC 3 — ACTUALITÉS (RSS multi-sources)
# ════════════════════════════════════════════════════════════════════════════════

async def _fetch_rss(session, url: str, limit: int = 8) -> list:
    """Fetch et parse un flux RSS, retourne une liste d'articles."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8),
                               headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "fr,en;q=0.9"}) as resp:
            if resp.status != 200:
                return []
            text = await resp.text(errors="replace")
        root = ET.fromstring(text)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)
        articles = []
        for item in items[:limit]:
            def g(tag):
                for prefix in ["", "atom:"]:
                    el = item.find(f"{prefix}{tag}") if ":" not in prefix else item.find(f"{{{ns['atom']}}}{tag}")
                    if el is not None and el.text:
                        return el.text.strip()
                return ""
            title   = g("title")
            pubdate = g("pubDate") or g("updated") or g("published")
            desc    = g("description") or g("summary")
            link    = g("link")
            if not link:
                link_el = item.find("link")
                if link_el is not None:
                    link = link_el.get("href","") or link_el.text or ""
            if title:
                articles.append({
                    "title":   title,
                    "date":    pubdate[:25] if pubdate else "",
                    "summary": desc[:300]   if desc    else "",
                    "url":     link,
                    "source":  url.split("/")[2].replace("www.","").replace("feeds.",""),
                })
        return articles
    except Exception:
        return []


@mcp.tool()
async def get_market_news(category: str = "general", limit: int = 20) -> dict:
    """Actualités financières multi-sources via RSS (EN + FR).

    Categories:
    - general     : Reuters, CNBC, MarketWatch, FT, Bloomberg
    - macro       : Macro/banques centrales, Fed, BCE
    - europe      : Reuters EU, Les Echos, BFM Business, L'Agefi
    - france      : Les Echos, BFM Business, La Tribune, Boursorama
    - commodities : Pétrole, or, matières premières
    - tech        : Tech/IA/semiconducteurs
    - crypto      : Bitcoin, Ethereum, DeFi
    - earnings    : Résultats d'entreprises
    """
    feeds_map = {
        "general": [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://www.ft.com/rss/home",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.investing.com/rss/news.rss",
        ],
        "macro": [
            "https://feeds.reuters.com/reuters/economy",
            "https://www.cnbc.com/id/20910258/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/economy-politics/",
            "https://www.ft.com/rss/stream/economics",
            "https://www.ecb.europa.eu/rss/press.html",
        ],
        "europe": [
            "https://feeds.reuters.com/reuters/europeanBusinessNews",
            "https://www.cnbc.com/id/19794221/device/rss/rss.html",
            "https://www.lesechos.fr/rss/rss_finance.xml",
            "https://bfmbusiness.bfmtv.com/rss/info/flux-rss/flux-toutes-les-actualites/",
            "https://www.lagefi.fr/rss.xml",
        ],
        "france": [
            "https://www.lesechos.fr/rss/rss_finance.xml",
            "https://bfmbusiness.bfmtv.com/rss/info/flux-rss/flux-toutes-les-actualites/",
            "https://www.latribune.fr/rss/rubriques/finance.xml",
            "https://www.boursorama.com/bourse/actualites/rss/une",
            "https://investir.lesechos.fr/rss.xml",
        ],
        "commodities": [
            "https://feeds.reuters.com/reuters/commoditiesNews",
            "https://www.cnbc.com/id/100727362/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/commodities/",
            "https://oilprice.com/rss/main",
        ],
        "tech": [
            "https://feeds.reuters.com/reuters/technologyNews",
            "https://www.cnbc.com/id/19854910/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/tech/",
            "https://www.ft.com/rss/stream/technology",
        ],
        "crypto": [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://feeds.reuters.com/reuters/crypto",
            "https://decrypt.co/feed",
        ],
        "earnings": [
            "https://feeds.reuters.com/reuters/companyNews",
            "https://www.cnbc.com/id/15839135/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/earnings/",
        ],
    }

    selected = feeds_map.get(category, feeds_map["general"])
    all_articles = []
    seen_titles  = set()

    async with aiohttp.ClientSession() as session:
        tasks   = [_fetch_rss(session, url) for url in selected]
        results = []
        for task in tasks:
            results.append(await task)

    for batch in results:
        for art in batch:
            if art["title"] not in seen_titles:
                seen_titles.add(art["title"])
                all_articles.append(art)

    all_articles = all_articles[:limit]
    return {
        "category": category,
        "count":    len(all_articles),
        "sources":  [u.split("/")[2].replace("www.","").replace("feeds.","") for u in selected],
        "articles": all_articles,
    }


@mcp.tool()
async def get_news(ticker: str) -> dict:
    """Dernières actualités pour un ticker spécifique (via yfinance)."""
    try:
        ticker   = _normalize_ticker(ticker)
        news_raw = yf.Ticker(ticker).news or []
        articles = []
        for n in news_raw[:15]:
            articles.append({
                "title":     n.get("title",""),
                "publisher": n.get("publisher",""),
                "date":      datetime.fromtimestamp(n.get("providerPublishTime",0)).strftime("%Y-%m-%d %H:%M"),
                "url":       n.get("link",""),
                "summary":   n.get("summary","")[:300],
            })
        return {"ticker": ticker, "count": len(articles), "articles": articles}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)