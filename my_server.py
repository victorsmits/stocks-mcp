from fastmcp import FastMCP
import aiohttp
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

mcp = FastMCP("Stock Information MCP Server")

# ─── Utilitaire ticker ────────────────────────────────────────────────────────

def _normalize_ticker(ticker: str) -> str:
    """Convertit format Google Finance → yfinance. Ex: AIR:EPA → AIR.PA"""
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

# ─── 1. Info générale ─────────────────────────────────────────────────────────

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
            "description": (info.get("longBusinessSummary") or "")[:500],
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 2. Historique de prix ────────────────────────────────────────────────────

@mcp.tool()
async def get_price_history(ticker: str, period: str = "3mo", interval: str = "1d") -> dict:
    """Historique OHLCV. Period: 1d/5d/1mo/3mo/6mo/1y/2y/5y. Interval: 1d/1wk/1mo"""
    try:
        ticker = _normalize_ticker(ticker)
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist.empty:
            return {"ticker": ticker, "error": "No data"}
        data = [
            {"date": d.strftime("%Y-%m-%d"), "open": round(r.Open,2),
             "high": round(r.High,2), "low": round(r.Low,2),
             "close": round(r.Close,2), "volume": int(r.Volume)}
            for d, r in hist.iterrows()
        ]
        return {"ticker": ticker, "period": period, "interval": interval,
                "count": len(data), "history": data}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 3. Indicateurs techniques ────────────────────────────────────────────────

@mcp.tool()
async def get_technical_indicators(ticker: str) -> dict:
    """RSI, MACD, Bollinger Bands, moyennes mobiles, momentum — calculés sur 1 an."""
    try:
        ticker = _normalize_ticker(ticker)
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return {"ticker": ticker, "error": "No data"}
        close = hist["Close"]

        # Moyennes mobiles
        ma20  = close.rolling(20).mean().iloc[-1]
        ma50  = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        current = close.iloc[-1]

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        rsi   = (100 - 100 / (1 + rs)).iloc[-1]

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line   = (ema12 - ema26).iloc[-1]
        signal_line = (ema12 - ema26).ewm(span=9).mean().iloc[-1]
        macd_hist   = macd_line - signal_line

        # Bollinger Bands (20, 2σ)
        bb_mid = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # ATR (14)
        high, low = hist["High"], hist["Low"]
        tr = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Performance
        perf_1m  = (current / close.iloc[-22]  - 1) * 100 if len(close) > 22  else None
        perf_3m  = (current / close.iloc[-66]  - 1) * 100 if len(close) > 66  else None
        perf_6m  = (current / close.iloc[-126] - 1) * 100 if len(close) > 126 else None
        perf_1y  = (current / close.iloc[0]    - 1) * 100

        def r(v): return round(float(v), 2) if v is not None else None

        return {
            "ticker": ticker,
            "current_price": r(current),
            "moving_averages": {
                "MA20": r(ma20), "MA50": r(ma50), "MA200": r(ma200),
                "above_MA20": current > ma20, "above_MA50": current > ma50,
                "above_MA200": current > ma200,
                "golden_cross": ma50 > ma200,
            },
            "rsi_14": r(rsi),
            "rsi_signal": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
            "macd": {
                "macd_line": r(macd_line), "signal_line": r(signal_line),
                "histogram": r(macd_hist),
                "bullish": macd_line > signal_line,
            },
            "bollinger_bands": {
                "upper": r(bb_upper), "mid": r(bb_mid), "lower": r(bb_lower),
                "position": r((current - bb_lower) / (bb_upper - bb_lower) * 100),
            },
            "atr_14": r(atr),
            "performance": {
                "1m_%": r(perf_1m), "3m_%": r(perf_3m),
                "6m_%": r(perf_6m), "1y_%": r(perf_1y),
            },
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 4. Financials ────────────────────────────────────────────────────────────

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
            "ticker": ticker,
            "note": "Values in millions",
            "income_statement": df_to_dict(stock.financials),
            "balance_sheet":    df_to_dict(stock.balance_sheet),
            "cash_flow":        df_to_dict(stock.cashflow),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 5. Earnings & estimations ────────────────────────────────────────────────

@mcp.tool()
async def get_earnings(ticker: str) -> dict:
    """EPS historique, estimations analystes, surprise earnings."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)
        info   = stock.info

        hist = stock.earnings_history
        rows = []
        if hist is not None and not hist.empty:
            for _, r in hist.head(8).iterrows():
                rows.append({
                    "date":     str(r.get("quarter", "")),
                    "eps_est":  r.get("epsEstimate"),
                    "eps_act":  r.get("epsActual"),
                    "surprise_pct": r.get("surprisePercent"),
                })

        return {
            "ticker": ticker,
            "trailing_eps":       info.get("trailingEps"),
            "forward_eps":        info.get("forwardEps"),
            "earnings_growth_yoy":info.get("earningsGrowth"),
            "revenue_growth_yoy": info.get("revenueGrowth"),
            "next_earnings_date": str(info.get("earningsTimestamp", "")),
            "earnings_history":   rows,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 6. Dividendes ────────────────────────────────────────────────────────────

@mcp.tool()
async def get_dividends(ticker: str) -> dict:
    """Historique des dividendes + métriques de croissance sur 5 ans."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)
        info   = stock.info
        divs   = stock.dividends

        history = []
        if not divs.empty:
            for d, v in divs.tail(20).items():
                history.append({"date": d.strftime("%Y-%m-%d"), "amount": round(float(v), 4)})

        # CAGR dividende 5 ans
        cagr = None
        if not divs.empty:
            annual = divs.resample("YE").sum()
            if len(annual) >= 5:
                cagr = round(((annual.iloc[-1] / annual.iloc[-5]) ** (1/5) - 1) * 100, 2)

        return {
            "ticker": ticker,
            "dividend_rate":    info.get("dividendRate"),
            "dividend_yield":   f"{round(info.get('dividendYield',0)*100,2)}%",
            "payout_ratio":     info.get("payoutRatio"),
            "ex_dividend_date": str(info.get("exDividendDate", "")),
            "5y_div_cagr_%":    cagr,
            "history":          history,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 7. Recommandations analystes ────────────────────────────────────────────

@mcp.tool()
async def get_analyst_recommendations(ticker: str) -> dict:
    """Recommandations buy/hold/sell + objectifs de cours des analystes."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)
        info   = stock.info

        recs = stock.recommendations
        recent = []
        if recs is not None and not recs.empty:
            for _, r in recs.tail(10).iterrows():
                recent.append({
                    "firm":   r.get("Firm", ""),
                    "action": r.get("Action", ""),
                    "grade":  r.get("To Grade", ""),
                })

        return {
            "ticker": ticker,
            "current_price":       info.get("currentPrice") or info.get("regularMarketPrice"),
            "target_mean_price":   info.get("targetMeanPrice"),
            "target_high_price":   info.get("targetHighPrice"),
            "target_low_price":    info.get("targetLowPrice"),
            "upside_%":            round((info.get("targetMeanPrice",0) /
                                         (info.get("currentPrice") or info.get("regularMarketPrice",1)) - 1)*100, 1)
                                   if info.get("targetMeanPrice") else None,
            "analyst_count":       info.get("numberOfAnalystOpinions"),
            "recommendation":      info.get("recommendationKey", "").upper(),
            "recent_changes":      recent,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 8. Actionnariat & insiders ───────────────────────────────────────────────

@mcp.tool()
async def get_holders(ticker: str) -> dict:
    """Actionnaires institutionnels, fonds, et transactions insiders."""
    try:
        ticker = _normalize_ticker(ticker)
        stock  = yf.Ticker(ticker)

        inst = stock.institutional_holders
        inst_list = []
        if inst is not None and not inst.empty:
            for _, r in inst.head(10).iterrows():
                inst_list.append({
                    "holder": r.get("Holder", ""),
                    "shares": int(r.get("Shares", 0)),
                    "pct_held": r.get("% Out"),
                })

        insiders = stock.insider_transactions
        insider_list = []
        if insiders is not None and not insiders.empty:
            for _, r in insiders.head(10).iterrows():
                insider_list.append({
                    "date":       str(r.get("Start Date", ""))[:10],
                    "insider":    r.get("Insider", ""),
                    "position":   r.get("Position", ""),
                    "transaction":r.get("Transaction", ""),
                    "shares":     r.get("Shares", 0),
                    "value":      r.get("Value", 0),
                })

        info = stock.info
        return {
            "ticker": ticker,
            "insider_ownership_%":       info.get("heldPercentInsiders"),
            "institutional_ownership_%": info.get("heldPercentInstitutions"),
            "top_institutions":          inst_list,
            "recent_insider_transactions": insider_list,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 9. Actualités ────────────────────────────────────────────────────────────

@mcp.tool()
async def get_news(ticker: str) -> dict:
    """Dernières actualités pour un ticker (10 articles max)."""
    try:
        ticker = _normalize_ticker(ticker)
        news   = yf.Ticker(ticker).news or []
        articles = []
        for n in news[:10]:
            articles.append({
                "title":     n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "date":      datetime.fromtimestamp(n.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M"),
                "url":       n.get("link", ""),
                "summary":   n.get("summary", "")[:300],
            })
        return {"ticker": ticker, "count": len(articles), "articles": articles}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ─── 10. Comparaison multi-tickers ────────────────────────────────────────────

@mcp.tool()
async def compare_stocks(tickers: list[str], metric: str = "all") -> dict:
    """Compare plusieurs actions côte à côte.
    
    Metric: all | price | valuation | growth | dividends | technical
    Exemple: tickers=["NVDA","AMD","INTC"], metric="valuation"
    """
    try:
        tickers = [_normalize_ticker(t) for t in tickers]
        result  = {}
        for t in tickers:
            info = yf.Ticker(t).info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            dy    = info.get("dividendYield", 0) or 0
            row   = {}
            if metric in ("all", "price"):
                row.update({
                    "price": price,
                    "52w_high": info.get("fiftyTwoWeekHigh"),
                    "52w_low":  info.get("fiftyTwoWeekLow"),
                    "from_52w_high_%": round((price / info.get("fiftyTwoWeekHigh",1) - 1)*100, 1) if price else None,
                })
            if metric in ("all", "valuation"):
                row.update({
                    "market_cap_B": round(info.get("marketCap",0)/1e9, 1),
                    "pe_trailing":  info.get("trailingPE"),
                    "pe_forward":   info.get("forwardPE"),
                    "pb":           info.get("priceToBook"),
                    "ps":           info.get("priceToSalesTrailing12Months"),
                    "ev_ebitda":    info.get("enterpriseToEbitda"),
                })
            if metric in ("all", "growth"):
                row.update({
                    "revenue_growth_%": round(info.get("revenueGrowth",0)*100, 1),
                    "earnings_growth_%":round(info.get("earningsGrowth",0)*100, 1),
                    "profit_margin_%":  round((info.get("profitMargins",0) or 0)*100, 1),
                    "roe_%":            round((info.get("returnOnEquity",0) or 0)*100, 1),
                })
            if metric in ("all", "dividends"):
                row.update({
                    "div_yield_%":  round(dy*100, 2),
                    "div_rate":     info.get("dividendRate"),
                    "payout_ratio": info.get("payoutRatio"),
                })
            result[t] = row
        return {"comparison": result, "metric": metric}
    except Exception as e:
        return {"error": str(e)}

# ─── 11. Marché global & indices ──────────────────────────────────────────────

@mcp.tool()
async def get_market_overview() -> dict:
    """Vue d'ensemble des principaux indices mondiaux et indicateurs macro."""
    indices = {
        "S&P 500":    "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones":  "^DJI",
        "VIX":        "^VIX",
        "CAC 40":     "^FCHI",
        "DAX":        "^GDAXI",
        "Euro Stoxx 600": "^STOXX",
        "AEX":        "^AEX",
        "Gold":       "GC=F",
        "Oil (WTI)":  "CL=F",
        "EUR/USD":    "EURUSD=X",
        "10Y US":     "^TNX",
        "10Y DE":     "^TNX",
    }
    result = {}
    for name, sym in indices.items():
        try:
            info  = yf.Ticker(sym).fast_info
            price = getattr(info, "last_price", None)
            prev  = getattr(info, "previous_close", None)
            chg   = round((price/prev - 1)*100, 2) if price and prev else None
            result[name] = {"price": round(float(price),2) if price else None, "change_%": chg}
        except:
            result[name] = {"price": None, "change_%": None}
    return {"market_overview": result, "as_of": datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

# ─── 12. Performance secteurs ─────────────────────────────────────────────────

@mcp.tool()
async def get_sector_performance(period: str = "1mo") -> dict:
    """Performance des 11 secteurs SPDR (ETFs US) sur la période choisie.
    Period: 1d / 5d / 1mo / 3mo / 6mo / 1y"""
    sectors = {
        "Technology":          "XLK",
        "Healthcare":          "XLV",
        "Financials":          "XLF",
        "Consumer Disc.":      "XLY",
        "Consumer Staples":    "XLP",
        "Industrials":         "XLI",
        "Energy":              "XLE",
        "Materials":           "XLB",
        "Real Estate":         "XLRE",
        "Utilities":           "XLU",
        "Communication Svcs":  "XLC",
    }
    result = {}
    for name, sym in sectors.items():
        try:
            hist = yf.Ticker(sym).history(period=period)
            if not hist.empty:
                perf = round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)*100, 2)
                result[name] = {"etf": sym, "performance_%": perf}
        except:
            result[name] = {"etf": sym, "performance_%": None}
    sorted_result = dict(sorted(result.items(), key=lambda x: x[1].get("performance_%") or -999, reverse=True))
    return {"sector_performance": sorted_result, "period": period}

# ─── 13. Calendrier earnings à venir ─────────────────────────────────────────

@mcp.tool()
async def get_upcoming_earnings(tickers: list[str]) -> dict:
    """Prochaines dates d'earnings pour une liste de tickers."""
    result = {}
    for t in tickers:
        tn = _normalize_ticker(t)
        try:
            info = yf.Ticker(tn).info
            result[t] = {
                "next_earnings_date":   str(info.get("earningsTimestamp", "Unknown")),
                "earnings_quarter":     info.get("earningsTimestampEnd", ""),
                "fiscal_year_end":      info.get("fiscalYearEnd", ""),
            }
        except Exception as e:
            result[t] = {"error": str(e)}
    return {"upcoming_earnings": result}

# ─── 14. Analyse du portefeuille ─────────────────────────────────────────────

@mcp.tool()
async def analyze_portfolio(holdings: dict) -> dict:
    """Analyse complète d'un portefeuille.
    
    holdings = {"NVDA": 85, "AMD": 75, "AIR.PA": 48, ...}
    Retourne : valeur totale, allocation, performance, diversification.
    """
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
                "shares":   shares,
                "price":    price,
                "value":    round(value, 2),
                "currency": info.get("currency", ""),
                "sector":   info.get("sector", "Unknown"),
                "name":     info.get("shortName", ticker),
                "beta":     info.get("beta"),
                "pe":       info.get("trailingPE"),
                "dy_%":     round((info.get("dividendYield") or 0)*100, 2),
            }
        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")

    # Allocation %
    for t in portfolio:
        portfolio[t]["allocation_%"] = round(portfolio[t]["value"] / total_value * 100, 2) if total_value else 0

    # Répartition par secteur
    sectors = {}
    for t, d in portfolio.items():
        s = d.get("sector", "Unknown")
        sectors[s] = round(sectors.get(s, 0) + d.get("allocation_%", 0), 2)

    # Métriques agrégées (pondérées)
    w_beta = sum((d["beta"] or 1) * d["allocation_%"] for d in portfolio.values() if d.get("beta")) / 100
    w_dy   = sum(d["dy_%"] * d["allocation_%"] for d in portfolio.values()) / 100

    return {
        "total_value":         round(total_value, 2),
        "positions_count":     len(portfolio),
        "sector_allocation_%": dict(sorted(sectors.items(), key=lambda x: -x[1])),
        "weighted_beta":       round(w_beta, 2),
        "weighted_div_yield_%":round(w_dy, 2),
        "positions":           dict(sorted(portfolio.items(), key=lambda x: -x[1]["value"])),
        "errors":              errors,
    }

# ─── 15. Screener simple ──────────────────────────────────────────────────────

@mcp.tool()
async def screen_stocks(tickers: list[str], min_div_yield: float = 0,
                        max_pe: float = 999, min_roe: float = 0) -> dict:
    """Filtre une liste de tickers selon critères fondamentaux.
    
    Ex: screen_stocks(["NVDA","MSFT","INTC","AMD"], max_pe=30, min_roe=15)
    """
    results = []
    for t in tickers:
        tn = _normalize_ticker(t)
        try:
            info = yf.Ticker(tn).info
            pe   = info.get("trailingPE") or 999
            roe  = (info.get("returnOnEquity") or 0) * 100
            dy   = (info.get("dividendYield") or 0) * 100
            if pe <= max_pe and roe >= min_roe and dy >= min_div_yield:
                results.append({
                    "ticker": t,
                    "name":   info.get("shortName", t),
                    "price":  info.get("currentPrice") or info.get("regularMarketPrice"),
                    "pe":     round(pe, 1) if pe < 999 else None,
                    "roe_%":  round(roe, 1),
                    "div_yield_%": round(dy, 2),
                    "market_cap_B": round((info.get("marketCap") or 0)/1e9, 1),
                })
        except:
            pass
    results.sort(key=lambda x: x.get("roe_%", 0), reverse=True)
    return {"screener_results": results, "count": len(results),
            "filters": {"max_pe": max_pe, "min_roe_%": min_roe, "min_div_yield_%": min_div_yield}}


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)

# ─── 16. Actualités marché global (RSS) ──────────────────────────────────────

@mcp.tool()
async def get_market_news(category: str = "general", limit: int = 15) -> dict:
    """Dernières actualités financières via flux RSS gratuits.
    
    Categories:
    - general     : Reuters Finance, MarketWatch, CNBC
    - macro       : Reuters Economy, Fed/BCE news
    - europe      : Reuters Europe, Les Echos
    - commodities : Reuters Commodities, Oil/Gold news
    - tech        : Reuters Tech, Seeking Alpha Tech
    - crypto      : CoinDesk, Reuters Crypto
    """
    import xml.etree.ElementTree as ET

    feeds = {
        "general": [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ],
        "macro": [
            "https://feeds.reuters.com/reuters/economy",
            "https://www.cnbc.com/id/20910258/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/economy-politics/",
        ],
        "europe": [
            "https://feeds.reuters.com/reuters/europeanBusinessNews",
            "https://www.cnbc.com/id/19794221/device/rss/rss.html",
        ],
        "commodities": [
            "https://feeds.reuters.com/reuters/commoditiesNews",
            "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        ],
        "tech": [
            "https://feeds.reuters.com/reuters/technologyNews",
            "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        ],
        "crypto": [
            "https://feeds.reuters.com/reuters/crypto",
            "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        ],
    }

    selected = feeds.get(category, feeds["general"])
    articles = []

    async with aiohttp.ClientSession() as session:
        for url in selected:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=8),
                                       headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    root = ET.fromstring(text)

                    # Support RSS 2.0 et Atom
                    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
                    for item in items[:8]:
                        def g(tag):
                            el = item.find(tag) or item.find(f"{{http://www.w3.org/2005/Atom}}{tag}")
                            return el.text.strip() if el is not None and el.text else ""

                        title   = g("title")
                        pubdate = g("pubDate") or g("updated") or g("published")
                        desc    = g("description") or g("summary") or g("content")
                        link    = g("link")

                        if title and title not in [a["title"] for a in articles]:
                            articles.append({
                                "title":   title,
                                "date":    pubdate[:25] if pubdate else "",
                                "summary": desc[:300] if desc else "",
                                "url":     link,
                                "source":  url.split("/")[2],
                            })
            except Exception:
                continue

    articles = articles[:limit]
    return {
        "category": category,
        "count":    len(articles),
        "articles": articles,
        "sources":  [u.split("/")[2] for u in selected],
    }