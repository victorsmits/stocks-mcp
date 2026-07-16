from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row


class PortfolioStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required")
        self.initialize()

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    source TEXT NOT NULL,
                    note TEXT,
                    payload JSONB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    trade_date DATE NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('buy','sell','dividend','fee','cash')),
                    quantity NUMERIC NOT NULL DEFAULT 0,
                    price NUMERIC,
                    currency TEXT NOT NULL DEFAULT 'EUR',
                    fees NUMERIC NOT NULL DEFAULT 0,
                    note TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE TABLE IF NOT EXISTS investment_theses (
                    ticker TEXT PRIMARY KEY,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS portfolio_watchlist (
                    ticker TEXT PRIMARY KEY,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decision_journal (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    ticker TEXT,
                    action TEXT NOT NULL,
                    payload JSONB NOT NULL
                );
            """)

    @staticmethod
    def _validated_transaction(tx: dict[str, Any]) -> tuple[str, str]:
        if tx.get("confirmed") is not True:
            raise ValueError("The transaction must contain confirmed=true")
        for field in ("trade_date", "ticker", "side"):
            if not tx.get(field):
                raise ValueError(f"Missing transaction field: {field}")
        side = str(tx["side"]).lower()
        if side not in {"buy", "sell", "dividend", "fee", "cash"}:
            raise ValueError("Invalid side")
        return str(tx["ticker"]).upper(), side

    @staticmethod
    def _transaction_values(tx: dict[str, Any], ticker: str, side: str) -> tuple[Any, ...]:
        metadata = dict(tx.get("metadata", {}))
        metadata["confirmed"] = True
        return (
            tx["trade_date"], ticker, side, tx.get("quantity", 0), tx.get("price"),
            str(tx.get("currency", "EUR")).upper(), tx.get("fees", 0),
            tx.get("note"), json.dumps(metadata),
        )

    def save_snapshot(self, snapshot: dict[str, Any], source: str = "user_confirmed", note: str | None = None) -> dict[str, Any]:
        if not snapshot:
            raise ValueError("snapshot cannot be empty")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO portfolio_snapshots(source,note,payload) VALUES (%s,%s,%s) RETURNING id,created_at",
                (source, note, json.dumps(snapshot)),
            )
            row = cur.fetchone()
        return {"id": row["id"], "created_at": row["created_at"].isoformat(), "source": source, "note": note, "snapshot": snapshot}

    def apply_confirmed_transaction(self, tx: dict[str, Any], updated_snapshot: dict[str, Any], note: str | None = None) -> dict[str, Any]:
        """Atomically persist a confirmed transaction and the resulting full snapshot."""
        ticker, side = self._validated_transaction(tx)
        if not updated_snapshot:
            raise ValueError("updated_snapshot cannot be empty")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO portfolio_transactions(trade_date,ticker,side,quantity,price,currency,fees,note,metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id,created_at
            """, self._transaction_values(tx, ticker, side))
            tx_row = cur.fetchone()
            cur.execute(
                "INSERT INTO portfolio_snapshots(source,note,payload) VALUES (%s,%s,%s) RETURNING id,created_at",
                ("confirmed_transaction", note or tx.get("note"), json.dumps(updated_snapshot)),
            )
            snapshot_row = cur.fetchone()
        return {
            "transaction": {**tx, "ticker": ticker, "side": side, "id": tx_row["id"], "created_at": tx_row["created_at"].isoformat()},
            "snapshot": {"id": snapshot_row["id"], "created_at": snapshot_row["created_at"].isoformat(), "snapshot": updated_snapshot},
        }

    def latest_snapshot(self) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,created_at,source,note,payload FROM portfolio_snapshots ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
        if not row:
            return None
        return {"id": row["id"], "created_at": row["created_at"].isoformat(), "source": row["source"], "note": row["note"], "snapshot": row["payload"]}

    def snapshot_history(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,created_at,source,note,payload FROM portfolio_snapshots ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [{"id": row["id"], "created_at": row["created_at"].isoformat(), "source": row["source"], "note": row["note"], "snapshot": row["payload"]} for row in rows]

    def list_transactions(self, limit: int = 100, ticker: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self._connect() as conn, conn.cursor() as cur:
            if ticker:
                cur.execute("SELECT * FROM portfolio_transactions WHERE ticker=%s ORDER BY trade_date DESC,id DESC LIMIT %s", (ticker.upper(), limit))
            else:
                cur.execute("SELECT * FROM portfolio_transactions ORDER BY trade_date DESC,id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["created_at"] = item["created_at"].isoformat()
            item["trade_date"] = item["trade_date"].isoformat()
            for key in ("quantity", "price", "fees"):
                item[key] = float(item[key]) if item[key] is not None else None
            result.append(item)
        return result

    def upsert_document(self, table: str, ticker: str, payload: dict[str, Any]) -> dict[str, Any]:
        if table not in {"investment_theses", "portfolio_watchlist"}:
            raise ValueError("Invalid table")
        normalized = ticker.upper()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table}(ticker,payload) VALUES (%s,%s) ON CONFLICT(ticker) DO UPDATE SET payload=EXCLUDED.payload,updated_at=now() RETURNING updated_at",
                (normalized, json.dumps(payload)),
            )
            updated_at = cur.fetchone()["updated_at"]
        return {"ticker": normalized, "updated_at": updated_at.isoformat(), "payload": payload}

    def get_document(self, table: str, ticker: str) -> dict[str, Any] | None:
        if table not in {"investment_theses", "portfolio_watchlist"}:
            raise ValueError("Invalid table")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT ticker,updated_at,payload FROM {table} WHERE ticker=%s", (ticker.upper(),))
            row = cur.fetchone()
        if not row:
            return None
        return {"ticker": row["ticker"], "updated_at": row["updated_at"].isoformat(), "payload": row["payload"]}

    def append_journal(self, action: str, payload: dict[str, Any], ticker: str | None = None) -> dict[str, Any]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO decision_journal(ticker,action,payload) VALUES (%s,%s,%s) RETURNING id,created_at",
                (ticker.upper() if ticker else None, action, json.dumps(payload)),
            )
            row = cur.fetchone()
        return {"id": row["id"], "created_at": row["created_at"].isoformat(), "ticker": ticker.upper() if ticker else None, "action": action, "payload": payload}
