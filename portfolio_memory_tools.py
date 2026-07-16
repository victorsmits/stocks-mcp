from __future__ import annotations

from typing import Any

from portfolio_memory import PortfolioMemory


def register_portfolio_memory_tools(mcp) -> None:
    def memory() -> PortfolioMemory:
        return PortfolioMemory()

    @mcp.tool()
    async def get_portfolio_state() -> dict:
        """Return latest confirmed snapshot plus all open/pending orders."""
        return memory().current_state()

    @mcp.tool()
    async def save_portfolio_snapshot(snapshot: dict[str, Any], source: str = "user_confirmed",
                                      reason: str | None = None, as_of: str | None = None) -> dict:
        """Persist a confirmed full snapshot. Never call for a simulation."""
        return memory().save_snapshot(snapshot, source, reason, as_of)

    @mcp.tool()
    async def get_portfolio_snapshot() -> dict:
        """Return the latest confirmed portfolio snapshot."""
        return {"snapshot": memory().latest_snapshot()}

    @mcp.tool()
    async def get_portfolio_history(limit: int = 50) -> dict:
        """Return versioned portfolio snapshots, newest first."""
        return {"snapshots": memory().snapshot_history(limit)}

    @mcp.tool()
    async def apply_confirmed_portfolio_transaction(transaction: dict[str, Any],
                                                    updated_snapshot: dict[str, Any],
                                                    reason: str | None = None) -> dict:
        """Atomically save a confirmed transaction, new snapshot, audit event and journal entry.

        transaction.confirmed must be true. Do not call for simulated or planned trades.
        """
        return memory().apply_confirmed_transaction(transaction, updated_snapshot, reason)

    @mcp.tool()
    async def upsert_portfolio_order(order: dict[str, Any]) -> dict:
        """Create or update a confirmed order, including pending and partially-filled orders.

        Supported statuses: draft, pending_confirmation, submitted, partially_filled,
        pending_cancel, filled, cancelled, expired, rejected.
        """
        return memory().upsert_order(order)

    @mcp.tool()
    async def get_open_portfolio_orders(limit: int = 100) -> dict:
        """Return all orders that can still affect future portfolio exposure."""
        return {"orders": memory().list_orders(limit=limit)}

    @mcp.tool()
    async def get_portfolio_orders(statuses: list[str], limit: int = 100) -> dict:
        """Return orders filtered by one or more lifecycle statuses."""
        return {"orders": memory().list_orders(statuses, limit)}

    @mcp.tool()
    async def get_portfolio_events(limit: int = 200, event_type: str | None = None,
                                   entity_type: str | None = None,
                                   entity_id: str | None = None) -> dict:
        """Return immutable audit events for portfolio, orders, transactions and documents."""
        return {"events": memory().list_events(limit, event_type, entity_type, entity_id)}

    @mcp.tool()
    async def reconcile_portfolio(observed_state: dict[str, Any], source: str = "broker_import") -> dict:
        """Compare a broker/import state with persisted memory and record the reconciliation result."""
        return memory().reconcile(observed_state, source)

    @mcp.tool()
    async def update_investment_thesis(ticker: str, thesis: dict[str, Any]) -> dict:
        """Create or version the persistent investment thesis for a ticker."""
        return memory().upsert_document("investment_theses", ticker, thesis)

    @mcp.tool()
    async def get_investment_thesis(ticker: str) -> dict:
        """Return the latest version of a persistent investment thesis."""
        return {"thesis": memory().get_document("investment_theses", ticker)}

    @mcp.tool()
    async def update_watchlist_entry(ticker: str, entry: dict[str, Any]) -> dict:
        """Create or version a persistent watchlist entry."""
        return memory().upsert_document("portfolio_watchlist", ticker, entry)

    @mcp.tool()
    async def get_watchlist_entry(ticker: str) -> dict:
        """Return the latest version of a persistent watchlist entry."""
        return {"entry": memory().get_document("portfolio_watchlist", ticker)}
