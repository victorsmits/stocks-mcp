from my_server import mcp
from portfolio_tools import register_portfolio_tools

register_portfolio_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
