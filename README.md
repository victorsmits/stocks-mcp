# Stock Information MCP Server

MCP financier basé sur FastMCP et yfinance. Il fournit des données de marché,
des analyses de portefeuille et une mémoire persistante PostgreSQL.

## Fonctionnalités

- cours, historiques, fondamentaux, dividendes et actualités ;
- indicateurs techniques, comparaisons et screener ;
- analyse de portefeuille ;
- snapshots versionnés du portefeuille dans PostgreSQL ;
- transactions, thèses d'investissement, watchlist et journal de décisions ;
- accès protégé par Google OAuth via oauth2-proxy ;
- liste blanche stricte des adresses Google autorisées.

## Démarrage sécurisé avec Docker Compose

### 1. Préparer l'environnement

```bash
cp .env.example .env
```

Renseignez les mots de passe et identifiants OAuth dans `.env`.

### 2. Créer les identifiants Google OAuth

Dans Google Cloud Console :

1. créez ou sélectionnez un projet ;
2. configurez l'écran de consentement OAuth ;
3. créez un client OAuth de type **Web application** ;
4. ajoutez comme URI de redirection autorisée :
   `https://votre-domaine.example/oauth2/callback` ;
5. copiez le Client ID et le Client Secret dans `.env`.

Pour un déploiement local HTTP, utilisez par exemple :
`http://localhost:8400/oauth2/callback` et définissez
`OAUTH2_PROXY_COOKIE_SECURE=false`.

### 3. Restreindre les comptes autorisés

Éditez `config/allowed-emails.txt` et placez une adresse Google autorisée par
ligne. Supprimez impérativement l'adresse d'exemple.

```text
victor@example.com
```

Même si le domaine OAuth est `*`, oauth2-proxy refusera les comptes absents de
ce fichier.

### 4. Lancer

```bash
docker compose up -d --build
```

Le conteneur MCP et PostgreSQL ne sont pas publiés directement. Seul
oauth2-proxy écoute sur `PUBLIC_PORT` et impose l'authentification Google avant
de transmettre les requêtes au MCP.

## Variables principales

| Variable | Description |
|---|---|
| `POSTGRES_PASSWORD` | mot de passe PostgreSQL |
| `GOOGLE_CLIENT_ID` | identifiant du client OAuth Google |
| `GOOGLE_CLIENT_SECRET` | secret OAuth Google |
| `OAUTH2_REDIRECT_URL` | URL publique `/oauth2/callback` |
| `OAUTH2_PROXY_COOKIE_SECRET` | secret de chiffrement des cookies |
| `PUBLIC_PORT` | port public d'oauth2-proxy, 8400 par défaut |

Générez le secret cookie avec :

```bash
openssl rand -base64 32 | tr -- '+/' '-_'
```

Ne commitez jamais le fichier `.env` ni une sauvegarde PostgreSQL.

## Outils de mémoire du portefeuille

### Portefeuille

- `save_portfolio_snapshot(snapshot, source, note)`
- `get_portfolio_snapshot()`
- `get_portfolio_history(limit)`

### Transactions

- `add_portfolio_transaction(transaction)`
- `get_portfolio_transactions(limit, ticker)`

Une transaction doit contenir au minimum :

```json
{
  "trade_date": "2026-07-16",
  "ticker": "NVDA",
  "side": "buy",
  "quantity": 5,
  "price": 170.0,
  "currency": "USD",
  "fees": 7.5,
  "confirmed": true
}
```

Les écritures doivent uniquement concerner des transactions confirmées. Une
simulation ne doit jamais être enregistrée comme une opération réelle.

### Thèses, watchlist et décisions

- `update_investment_thesis(ticker, thesis)`
- `get_investment_thesis(ticker)`
- `update_watchlist_entry(ticker, entry)`
- `get_watchlist_entry(ticker)`
- `append_decision_journal(action, entry, ticker)`

## Source complémentaire

Les données yfinance et les calculs du MCP sont des sources complémentaires.
Pour une décision importante, croisez-les avec les documents officiels de
l'émetteur, les données du courtier et les sources réglementaires.

## Développement sans Docker

Python 3.13 est requis.

```bash
uv sync
export DATABASE_URL='postgresql://stocks:password@localhost:5432/stocks'
uv run my_server.py
```

## Format des tickers

- `AAPL:NASDAQ`
- `MSFT:NASDAQ`
- `IBM:NYSE`
- `AIR.PA`
- `ASML.AS`
- `ABI.BR`

## Sécurité

- N'exposez jamais directement le port 8000 du serveur MCP.
- Utilisez HTTPS en production.
- Limitez `config/allowed-emails.txt` à vos comptes personnels.
- Utilisez des secrets longs et distincts.
- Sauvegardez PostgreSQL de manière chiffrée.
- Les outils d'écriture ne doivent être appelés qu'après confirmation explicite.

## Licence

GNU General Public License v3.0 — voir `LICENSE`.
