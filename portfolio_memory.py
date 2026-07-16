from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL")
OPEN_ORDER_STATUSES = {"draft", "pending_confirmation", "submitted", "partially_filled", "pending_cancel"}
ORDER_STATUSES = OPEN_ORDER_STATUSES | {"filled", "cancelled", "expired", "rejected"}
ORDER_TYPES = {"market", "limit", "stop", "stop_limit", "trailing_stop"}
SIDES = {"buy", "sell"}
TRANSACTION_TYPES = {"buy", "sell", "dividend", "fee", "tax", "cash", "interest", "split", "transfer"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _idempotency_key(payload: dict[str, Any]) -> str:
    canonical = _json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _serialize(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key, value in list(result.items()):
        if isinstance(value, (datetime,)):
            result[key] = value.isoformat()
        elif isinstance(value, Decimal):
            result[key] = float(value)
    return result


class PortfolioMemory:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or DATABASE_URL
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required")
        self.initialize()

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_events (
                    id UUID PRIMARY KEY,
                    sequence BIGSERIAL UNIQUE NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    event_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT,
                    source TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    payload JSONB NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id UUID PRIMARY KEY,
                    version BIGSERIAL UNIQUE NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    as_of TIMESTAMPTZ NOT NULL,
                    source TEXT NOT NULL,
                    reason TEXT,
                    payload JSONB NOT NULL,
                    event_sequence BIGINT REFERENCES portfolio_events(sequence)
                );

                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    id UUID PRIMARY KEY,
                    external_id TEXT,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    trade_date DATE NOT NULL,
                    settlement_date DATE,
                    ticker TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    quantity NUMERIC NOT NULL DEFAULT 0,
                    price NUMERIC,
                    gross_amount NUMERIC,
                    net_amount NUMERIC,
                    currency TEXT NOT NULL,
                    fees NUMERIC NOT NULL DEFAULT 0,
                    taxes NUMERIC NOT NULL DEFAULT 0,
                    broker TEXT,
                    account_id TEXT,
                    confirmed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_transactions_external
                ON portfolio_transactions(broker, account_id, external_id)
                WHERE external_id IS NOT NULL;

                CREATE TABLE IF NOT EXISTS portfolio_orders (
                    id UUID PRIMARY KEY,
                    external_id TEXT,
                    client_order_id TEXT,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    quantity NUMERIC NOT NULL,
                    filled_quantity NUMERIC NOT NULL DEFAULT 0,
                    limit_price NUMERIC,
                    stop_price NUMERIC,
                    trailing_percent NUMERIC,
                    average_fill_price NUMERIC,
                    currency TEXT NOT NULL,
                    time_in_force TEXT,
                    submitted_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ,
                    cancelled_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    broker TEXT,
                    account_id TEXT,
                    confirmed BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_orders_external
                ON portfolio_orders(broker, account_id, external_id)
                WHERE external_id IS NOT NULL;

                CREATE TABLE IF NOT EXISTS investment_theses (
                    ticker TEXT PRIMARY KEY,
                    version BIGINT NOT NULL DEFAULT 1,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_watchlist (
                    ticker TEXT PRIMARY KEY,
                    version BIGINT NOT NULL DEFAULT 1,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    payload JSONB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decision_journal (
                    id UUID PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    ticker TEXT,
                    action TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    event_sequence BIGINT REFERENCES portfolio_events(sequence)
                );

                CREATE TABLE IF NOT EXISTS reconciliation_runs (
                    id UUID PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expected JSONB NOT NULL,
                    observed JSONB NOT NULL,
                    differences JSONB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS ix_events_type_sequence ON portfolio_events(event_type, sequence DESC);
                CREATE INDEX IF NOT EXISTS ix_events_entity ON portfolio_events(entity_type, entity_id, sequence DESC);
                CREATE INDEX IF NOT EXISTS ix_orders_status ON portfolio_orders(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS ix_transactions_ticker_date ON portfolio_transactions(ticker, trade_date DESC);
            """)

    def append_event(self, event_type: str, entity_type: str, payload: dict[str, Any], *,
                     entity_id: str | None = None, source: str = "user_confirmed",
                     actor: str = "InvestmentOS", metadata: dict[str, Any] | None = None,
                     occurred_at: str | None = None, idempotency_key: str | None = None,
                     cursor=None) -> dict[str, Any]:
        eid = str(uuid.uuid4())
        event_payload = {
            "event_type": event_type, "entity_type": entity_type, "entity_id": entity_id,
            "payload": payload, "source": source, "occurred_at": occurred_at,
        }
        key = idempotency_key or _idempotency_key(event_payload)
        owns = cursor is None
        conn = self.connect() if owns else None
        cur = conn.cursor() if owns else cursor
        try:
            cur.execute("""
                INSERT INTO portfolio_events(id,occurred_at,event_type,entity_type,entity_id,source,actor,idempotency_key,payload,metadata)
                VALUES (%s,COALESCE(%s::timestamptz,now()),%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(idempotency_key) DO UPDATE SET idempotency_key=EXCLUDED.idempotency_key
                RETURNING *
            """, (eid, occurred_at, event_type, entity_type, entity_id, source, actor, key, _json(payload), _json(metadata or {})))
            row = cur.fetchone()
            if owns:
                conn.commit()
            return _serialize(row) or {}
        finally:
            if owns:
                cur.close(); conn.close()

    def save_snapshot(self, snapshot: dict[str, Any], source: str = "user_confirmed", reason: str | None = None,
                      as_of: str | None = None, cursor=None) -> dict[str, Any]:
        owns = cursor is None
        conn = self.connect() if owns else None
        cur = conn.cursor() if owns else cursor
        try:
            event = self.append_event("SNAPSHOT_CREATED", "portfolio", snapshot, source=source,
                                      actor="InvestmentOS", cursor=cur)
            sid = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO portfolio_snapshots(id,as_of,source,reason,payload,event_sequence)
                VALUES (%s,COALESCE(%s::timestamptz,now()),%s,%s,%s,%s) RETURNING *
            """, (sid, as_of, source, reason, _json(snapshot), event["sequence"]))
            row = cur.fetchone()
            if owns:
                conn.commit()
            return _serialize(row) or {}
        finally:
            if owns:
                cur.close(); conn.close()

    def latest_snapshot(self) -> dict[str, Any] | None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM portfolio_snapshots ORDER BY version DESC LIMIT 1")
            return _serialize(cur.fetchone())

    def snapshot_history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM portfolio_snapshots ORDER BY version DESC LIMIT %s", (max(1, min(limit, 500)),))
            return [_serialize(r) for r in cur.fetchall()]

    def apply_confirmed_transaction(self, transaction: dict[str, Any], updated_snapshot: dict[str, Any],
                                    reason: str | None = None) -> dict[str, Any]:
        if transaction.get("confirmed") is not True:
            raise ValueError("confirmed must be true")
        tx_type = str(transaction.get("transaction_type") or transaction.get("side", "")).lower()
        if tx_type not in TRANSACTION_TYPES:
            raise ValueError(f"Unsupported transaction_type: {tx_type}")
        required = ("trade_date", "ticker", "currency")
        missing = [f for f in required if not transaction.get(f)]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
        key = transaction.get("idempotency_key") or _idempotency_key(transaction)
        txid = str(uuid.uuid4())
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO portfolio_transactions(
                    id,external_id,idempotency_key,trade_date,settlement_date,ticker,transaction_type,
                    quantity,price,gross_amount,net_amount,currency,fees,taxes,broker,account_id,confirmed,metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s)
                ON CONFLICT(idempotency_key) DO UPDATE SET idempotency_key=EXCLUDED.idempotency_key
                RETURNING *
            """, (txid, transaction.get("external_id"), key, transaction["trade_date"], transaction.get("settlement_date"),
                  str(transaction["ticker"]).upper(), tx_type, transaction.get("quantity", 0), transaction.get("price"),
                  transaction.get("gross_amount"), transaction.get("net_amount"), str(transaction["currency"]).upper(),
                  transaction.get("fees", 0), transaction.get("taxes", 0), transaction.get("broker"),
                  transaction.get("account_id"), _json(transaction.get("metadata", {}))))
            tx = _serialize(cur.fetchone())
            event = self.append_event("TRANSACTION_RECORDED", "transaction", tx or {}, entity_id=(tx or {}).get("id"),
                                      source="user_confirmed", cursor=cur, idempotency_key=f"event:{key}")
            snapshot = self.save_snapshot(updated_snapshot, "transaction_reconciliation", reason,
                                          updated_snapshot.get("as_of"), cursor=cur)
            cur.execute("INSERT INTO decision_journal(id,ticker,action,payload,event_sequence) VALUES (%s,%s,%s,%s,%s) RETURNING *",
                        (str(uuid.uuid4()), str(transaction["ticker"]).upper(), tx_type, _json({"transaction": tx, "reason": reason}), event["sequence"]))
            journal = _serialize(cur.fetchone())
            conn.commit()
        return {"transaction": tx, "snapshot": snapshot, "journal": journal, "event": event}

    def upsert_order(self, order: dict[str, Any]) -> dict[str, Any]:
        if order.get("confirmed") is not True:
            raise ValueError("confirmed must be true")
        status = str(order.get("status", "draft")).lower()
        side = str(order.get("side", "")).lower()
        order_type = str(order.get("order_type", "")).lower()
        if status not in ORDER_STATUSES or side not in SIDES or order_type not in ORDER_TYPES:
            raise ValueError("Invalid order status, side or order_type")
        for field in ("ticker", "quantity", "currency"):
            if order.get(field) is None:
                raise ValueError(f"Missing order field: {field}")
        key = order.get("idempotency_key") or _idempotency_key(order)
        oid = order.get("id") or str(uuid.uuid4())
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO portfolio_orders(
                    id,external_id,client_order_id,idempotency_key,ticker,side,order_type,status,quantity,
                    filled_quantity,limit_price,stop_price,trailing_percent,average_fill_price,currency,time_in_force,
                    submitted_at,expires_at,cancelled_at,completed_at,broker,account_id,confirmed,metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,%s)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    status=EXCLUDED.status, filled_quantity=EXCLUDED.filled_quantity,
                    average_fill_price=EXCLUDED.average_fill_price, cancelled_at=EXCLUDED.cancelled_at,
                    completed_at=EXCLUDED.completed_at, updated_at=now(), metadata=EXCLUDED.metadata
                RETURNING *
            """, (oid, order.get("external_id"), order.get("client_order_id"), key, str(order["ticker"]).upper(),
                  side, order_type, status, order["quantity"], order.get("filled_quantity", 0), order.get("limit_price"),
                  order.get("stop_price"), order.get("trailing_percent"), order.get("average_fill_price"),
                  str(order["currency"]).upper(), order.get("time_in_force"), order.get("submitted_at"),
                  order.get("expires_at"), order.get("cancelled_at"), order.get("completed_at"), order.get("broker"),
                  order.get("account_id"), _json(order.get("metadata", {}))))
            saved = _serialize(cur.fetchone())
            event_type = "ORDER_CREATED" if status in {"draft", "pending_confirmation", "submitted"} else "ORDER_UPDATED"
            event = self.append_event(event_type, "order", saved or {}, entity_id=(saved or {}).get("id"),
                                      source="user_confirmed", cursor=cur, idempotency_key=f"event:{key}:{status}:{order.get('filled_quantity', 0)}")
            conn.commit()
        return {"order": saved, "event": event}

    def list_orders(self, statuses: list[str] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        statuses = statuses or list(OPEN_ORDER_STATUSES)
        invalid = set(statuses) - ORDER_STATUSES
        if invalid:
            raise ValueError(f"Invalid statuses: {sorted(invalid)}")
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM portfolio_orders WHERE status = ANY(%s) ORDER BY updated_at DESC LIMIT %s",
                        (statuses, max(1, min(limit, 1000))))
            return [_serialize(r) for r in cur.fetchall()]

    def current_state(self) -> dict[str, Any]:
        return {
            "snapshot": self.latest_snapshot(),
            "open_orders": self.list_orders(),
            "generated_at": _utcnow().isoformat(),
        }

    def list_events(self, limit: int = 200, event_type: str | None = None,
                    entity_type: str | None = None, entity_id: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if event_type: clauses.append("event_type=%s"); params.append(event_type)
        if entity_type: clauses.append("entity_type=%s"); params.append(entity_type)
        if entity_id: clauses.append("entity_id=%s"); params.append(entity_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(max(1, min(limit, 2000)))
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM portfolio_events{where} ORDER BY sequence DESC LIMIT %s", params)
            return [_serialize(r) for r in cur.fetchall()]

    def upsert_document(self, table: str, ticker: str, payload: dict[str, Any]) -> dict[str, Any]:
        if table not in {"investment_theses", "portfolio_watchlist"}:
            raise ValueError("Invalid document table")
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table}(ticker,payload) VALUES (%s,%s)
                ON CONFLICT(ticker) DO UPDATE SET payload=EXCLUDED.payload,version={table}.version+1,updated_at=now()
                RETURNING *
            """, (ticker.upper(), _json(payload)))
            row = _serialize(cur.fetchone())
            self.append_event("THESIS_UPDATED" if table == "investment_theses" else "WATCHLIST_UPDATED",
                              "thesis" if table == "investment_theses" else "watchlist", row or {},
                              entity_id=ticker.upper(), source="user_confirmed", cursor=cur)
            conn.commit()
            return row or {}

    def get_document(self, table: str, ticker: str) -> dict[str, Any] | None:
        if table not in {"investment_theses", "portfolio_watchlist"}:
            raise ValueError("Invalid document table")
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table} WHERE ticker=%s", (ticker.upper(),))
            return _serialize(cur.fetchone())

    def reconcile(self, observed: dict[str, Any], source: str) -> dict[str, Any]:
        expected = self.current_state()
        differences = {"matches": expected.get("snapshot", {}).get("payload") == observed,
                       "expected_snapshot": expected.get("snapshot"), "observed": observed}
        status = "matched" if differences["matches"] else "difference_found"
        rid = str(uuid.uuid4())
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO reconciliation_runs(id,source,status,expected,observed,differences) VALUES (%s,%s,%s,%s,%s,%s) RETURNING *",
                        (rid, source, status, _json(expected), _json(observed), _json(differences)))
            row = _serialize(cur.fetchone())
            self.append_event("RECONCILIATION_COMPLETED", "portfolio", row or {}, entity_id=rid,
                              source=source, cursor=cur)
            conn.commit()
            return row or {}
