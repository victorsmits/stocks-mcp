# Portfolio memory model

The portfolio memory uses PostgreSQL as a hybrid relational/document store.
Financial identifiers, lifecycle states, quantities, prices and dates are typed
columns. Extensible agent-generated context lives in JSONB payloads and metadata.

## State layers

### Confirmed current state

`portfolio_snapshots` stores immutable, versioned full snapshots. A snapshot may
contain holdings, cash balances, accrued income, liabilities, FX assumptions,
account identifiers and any broker-specific details.

### Future exposure

`portfolio_orders` stores orders independently from executed positions. Open
statuses are:

- `draft`
- `pending_confirmation`
- `submitted`
- `partially_filled`
- `pending_cancel`

Terminal statuses are `filled`, `cancelled`, `expired` and `rejected`.

`get_portfolio_state()` returns the latest snapshot and all open orders, allowing
InvestmentOS to distinguish actual exposure from potential future exposure.

### Immutable history

`portfolio_events` is an append-only audit log. Every meaningful mutation emits
an event with an idempotency key, actor, source, entity and JSONB payload.

Typical events:

- `SNAPSHOT_CREATED`
- `TRANSACTION_RECORDED`
- `ORDER_CREATED`
- `ORDER_UPDATED`
- `THESIS_UPDATED`
- `WATCHLIST_UPDATED`
- `RECONCILIATION_COMPLETED`

## Safety rules

- Real transactions and orders require `confirmed: true`.
- Simulations must never call persistence tools.
- Transaction + snapshot + journal are committed atomically.
- Idempotency keys prevent duplicate imports and retries.
- External broker IDs can be unique per broker and account.
- PostgreSQL and the MCP server are private Docker services.
- Google OAuth and the explicit email allowlist protect the public endpoint.

## Reconciliation

`reconcile_portfolio()` compares the latest persisted state with an observed
broker/import state and records both the differences and an immutable audit event.
It never silently overwrites the persisted state.

## Why PostgreSQL rather than MongoDB

The source of truth requires constraints, transactions, indexes, uniqueness and
atomic updates. JSONB provides document flexibility for theses, metadata and
broker-specific structures without sacrificing relational integrity.
