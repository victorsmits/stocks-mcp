from fastmcp import FastMCP
import aiohttp
from bs4 import BeautifulSoup

mcp = FastMCP("Stock Information MCP Server")

@mcp.tool()
async def get_stock_info(ticker: str) -> dict:
    """Get stock price and description for a given ticker from Google Finance."""
    try:
        url = f"https://www.google.com/finance/quote/{ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')

        # Get current price
        price_element = soup.select_one('div.YMlKec.fxKbKc')
        price = price_element.text if price_element else "Price not found"

        # Get description - try different selectors
        description = None
        for selector in ['.bLLb2d', '.P6K39c']:
            description = soup.select_one(selector)
            if description:
                break

        description_text = description.get_text().strip() if description else "Description not found"

        result = {
            "ticker": ticker,
            "price": price,
            "description": description_text[:500]
        }
        return result

    except Exception as e:
        return {
            "ticker": ticker,
            "price": "Error",
            "description": f"Error retrieving data: {str(e)}"
        }

if __name__ == "__main__":
    # Transport SSE natif — plus besoin de supergateway
    mcp.run(transport="sse", host="0.0.0.0", port=8000)