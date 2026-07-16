from my_server import mcp
from portfolio_memory_tools import register_portfolio_memory_tools

register_portfolio_memory_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
