from __future__ import annotations

from typing import Any

from portfolio_store_v2 import PortfolioStore


def register_portfolio_tools(mcp) -> None:
    def store() -> PortfolioStore:
        return PortfolioStore()

    @mcp.tool()
    async def save_portfolio_snapshot(snapshot: dict[str, Any], source: str = "user_confirmed", note: str | None = None) -> dict:
        """Persist a confirmed full portfolio snapshot."""
        return store().save_snapshot(snapshot, source, note)

    @mcp.tool()
    async def apply_confirmed_portfolio_transaction(transaction: dict[str, Any], updated_snapshot: dict[str, Any], note: str | None = None) -> dict:
        """Atomically save a confirmed transaction and the resulting full portfolio snapshot. transaction.confirmed must be true."""
        return store().apply_confirmed_transaction(transaction, updated_snapshot, note)

    @mcp.tool()
    async def get_portfolio_snapshot() -> dict:
        """Return the latest persisted portfolio snapshot."""
        return {"snapshot": store().latest_snapshot()}

    @mcp.tool()
    async def get_portfolio_history(limit: int = 20) -> dict:
        """Return versioned portfolio snapshots, newest first."""
        return {"snapshots": store().snapshot_history(limit)}

    @mcp.tool()
    async def get_portfolio_transactions(limit: int = 100, ticker: str | None = None) -> dict:
        """Return persisted confirmed transactions, optionally filtered by ticker."""
        return {"transactions": store().list_transactions(limit, ticker)}

    @mcp.tool()
    async def update_investment_thesis(ticker: str, thesis: dict[str, Any]) -> dict:
        """Create or replace the persistent investment thesis for a ticker."""
        return store().upsert_document("investment_theses", ticker, thesis)

    @mcp.tool()
    async def get_investment_thesis(ticker: str) -> dict:
        """Return the persistent investment thesis for a ticker."""
        return {"thesis": store().get_document("investment_theses", ticker)}

    @mcp.tool()
    async def update_watchlist_entry(ticker: str, entry: dict[str, Any]) -> dict:
        """Create or replace a persistent watchlist entry."""
        return store().upsert_document("portfolio_watchlist", ticker, entry)

    @mcp.tool()
    async def get_watchlist_entry(ticker: str) -> dict:
        """Return a persistent watchlist entry."""
        return {"entry": store().get_document("portfolio_watchlist", ticker)}

    @mcp.tool()
    async def append_decision_journal(action: str, entry: dict[str, Any], ticker: str | None = None) -> dict:
        """Append a persistent decision-journal entry."""
        return store().append_journal(action, entry, ticker)
