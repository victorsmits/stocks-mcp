from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortfolioStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or DATABASE_URL
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

    def save_snapshot(self, snapshot: dict[str, Any], source: str, note: str | None = None) -> dict[str, Any]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO portfolio_snapshots(source,note,payload) VALUES (%s,%s,%s) RETURNING id,created_at",
                (source, note, json.dumps(snapshot)),
            )
            row = cur.fetchone()
        return {"id": row["id"], "created_at": row["created_at"].isoformat(), "source": source, "note": note, "snapshot": snapshot}

    def latest_snapshot(self) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,created_at,source,note,payload FROM portfolio_snapshots ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
        if not row:
            return None
        row["created_at"] = row["created_at"].isoformat()
        row["snapshot"] = row.pop("payload")
        return row

    def snapshot_history(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,created_at,source,note,payload FROM portfolio_snapshots ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [{**row, "created_at": row["created_at"].isoformat(), "snapshot": row.pop("payload")} for row in rows]

    def add_transaction(self, tx: dict[str, Any]) -> dict[str, Any]:
        for field in ("trade_date", "ticker", "side"):
            if field not in tx:
                raise ValueError(f"Missing transaction field: {field}")
        side = str(tx["side"]).lower()
        if side not in {"buy", "sell", "dividend", "fee", "cash"}:
            raise ValueError("Invalid side")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO portfolio_transactions(trade_date,ticker,side,quantity,price,currency,fees,note,metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id,created_at
            """, (
                tx["trade_date"], str(tx["ticker"]).upper(), side, tx.get("quantity", 0), tx.get("price"),
                str(tx.get("currency", "EUR")).upper(), tx.get("fees", 0), tx.get("note"), json.dumps(tx.get("metadata", {})),
            ))
            row = cur.fetchone()
        return {**tx, "id": row["id"], "created_at": row["created_at"].isoformat(), "side": side}

    def list_transactions(self, limit: int = 100, ticker: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self._connect() as conn, conn.cursor() as cur:
            if ticker:
                cur.execute("SELECT * FROM portfolio_transactions WHERE ticker=%s ORDER BY trade_date DESC,id DESC LIMIT %s", (ticker.upper(), limit))
            else:
                cur.execute("SELECT * FROM portfolio_transactions ORDER BY trade_date DESC,id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        for row in rows:
            row["created_at"] = row["created_at"].isoformat()
            row["trade_date"] = row["trade_date"].isoformat()
            for key in ("quantity", "price", "fees"):
                row[key] = float(row[key]) if row[key] is not None else None
        return rows

    def upsert_document(self, table: str, ticker: str, payload: dict[str, Any]) -> dict[str, Any]:
        if table not in {"investment_theses", "portfolio_watchlist"}:
            raise ValueError("Invalid table")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table}(ticker,payload) VALUES (%s,%s) ON CONFLICT(ticker) DO UPDATE SET payload=EXCLUDED.payload,updated_at=now() RETURNING updated_at",
                (ticker.upper(), json.dumps(payload)),
            )
            updated_at = cur.fetchone()["updated_at"]
        return {"ticker": ticker.upper(), "updated_at": updated_at.isoformat(), "payload": payload}

    def get_document(self, table: str, ticker: str) -> dict[str, Any] | None:
        if table not in {"investment_theses", "portfolio_watchlist"}:
            raise ValueError("Invalid table")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT ticker,updated_at,payload FROM {table} WHERE ticker=%s", (ticker.upper(),))
            row = cur.fetchone()
        if row:
            row["updated_at"] = row["updated_at"].isoformat()
        return row

    def append_journal(self, action: str, payload: dict[str, Any], ticker: str | None = None) -> dict[str, Any]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO decision_journal(ticker,action,payload) VALUES (%s,%s,%s) RETURNING id,created_at", (ticker.upper() if ticker else None, action, json.dumps(payload)))
            row = cur.fetchone()
        return {"id": row["id"], "created_at": row["created_at"].isoformat(), "ticker": ticker, "action": action, "payload": payload}
