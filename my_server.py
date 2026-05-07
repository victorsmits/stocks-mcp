from fastmcp import FastMCP
import yfinance as yf

mcp = FastMCP("Stock Information MCP Server")

@mcp.tool()
async def get_stock_info(ticker: str) -> dict:
    """Get stock price and description for a given ticker using yfinance.
    
    Ticker examples: NVDA, AAPL, MSFT, AIR.PA (Air Liquide), SU.PA (Schneider), ASML.AS
    """
    try:
        # Convertir format Google Finance (AIR:EPA) → yfinance (AIR.PA)
        ticker = _normalize_ticker(ticker)

        stock = yf.Ticker(ticker)
        info = stock.info

        # Prix : essayer plusieurs champs selon dispo
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        currency = info.get("currency", "")
        name = info.get("longName") or info.get("shortName", ticker)
        description = info.get("longBusinessSummary", "No description available")
        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        week_52_high = info.get("fiftyTwoWeekHigh")
        week_52_low = info.get("fiftyTwoWeekLow")
        dividend_yield = info.get("dividendYield")
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        return {
            "ticker": ticker,
            "name": name,
            "price": f"{price} {currency}" if price else "Price not available",
            "sector": sector,
            "industry": industry,
            "market_cap": f"{market_cap:,}" if market_cap else "N/A",
            "pe_ratio": round(pe_ratio, 2) if pe_ratio else "N/A",
            "52w_high": week_52_high,
            "52w_low": week_52_low,
            "dividend_yield": f"{round(dividend_yield * 100, 2)}%" if dividend_yield else "N/A",
            "description": description[:600],
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "price": "Error",
            "description": f"Error retrieving data: {str(e)}"
        }


@mcp.tool()
async def get_stock_history(ticker: str, period: str = "1mo") -> dict:
    """Get historical price data for a ticker.
    
    Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y
    Ticker examples: NVDA, AAPL, AIR.PA, ASML.AS
    """
    try:
        ticker = _normalize_ticker(ticker)
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return {"ticker": ticker, "error": "No historical data found"}

        data = []
        for date, row in hist.tail(30).iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })

        return {
            "ticker": ticker,
            "period": period,
            "data_points": len(data),
            "history": data,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def _normalize_ticker(ticker: str) -> str:
    """Convertit les tickers format Google Finance vers yfinance.
    
    AIR:EPA  → AIR.PA
    SU:EPA   → SU.PA
    ASML:AMS → ASML.AS
    NVDA     → NVDA (inchangé)
    """
    exchange_map = {
        "EPA": "PA",   # Euronext Paris
        "AMS": "AS",   # Euronext Amsterdam
        "EBR": "BR",   # Euronext Bruxelles
        "ETR": "DE",   # Xetra (Allemagne)
        "LON": "L",    # London Stock Exchange
        "NYSE": "",    # NYSE — pas de suffixe
        "NASDAQ": "",  # Nasdaq — pas de suffixe
    }
    if ":" in ticker:
        symbol, exchange = ticker.split(":", 1)
        suffix = exchange_map.get(exchange.upper())
        if suffix is None:
            return ticker  # exchange inconnu, on laisse tel quel
        return f"{symbol}.{suffix}" if suffix else symbol
    return ticker


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)