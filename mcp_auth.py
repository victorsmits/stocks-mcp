from __future__ import annotations

import os
from collections.abc import Iterable

from fastmcp.server.auth import OAuthProxy
from fastmcp.server.auth.providers.google import GoogleTokenVerifier


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _emails(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


class AllowlistedGoogleTokenVerifier(GoogleTokenVerifier):
    """Validate Google tokens and restrict access to explicit emails."""

    def __init__(self, allowed_emails: Iterable[str]) -> None:
        super().__init__(required_scopes=["openid", "email"])
        self.allowed_emails = {email.lower() for email in allowed_emails}
        if not self.allowed_emails:
            raise RuntimeError("ALLOWED_GOOGLE_EMAILS must contain at least one email")

    async def verify_token(self, token: str):
        access_token = await super().verify_token(token)
        if access_token is None:
            return None
        email = str(access_token.claims.get("email") or "").lower()
        verified = access_token.claims.get("email_verified")
        if email not in self.allowed_emails or verified in {False, "false", "False", 0}:
            return None
        return access_token


def build_auth() -> OAuthProxy:
    public_url = _required("MCP_PUBLIC_URL").rstrip("/")
    return OAuthProxy(
        upstream_authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        upstream_token_endpoint="https://oauth2.googleapis.com/token",
        upstream_client_id=_required("GOOGLE_CLIENT_ID"),
        upstream_client_secret=_required("GOOGLE_CLIENT_SECRET"),
        token_verifier=AllowlistedGoogleTokenVerifier(_emails(os.getenv("ALLOWED_GOOGLE_EMAILS"))),
        base_url=public_url,
        resource_base_url=public_url,
        issuer_url=public_url,
        redirect_path="/auth/callback",
        valid_scopes=["openid", "email"],
        jwt_signing_key=_required("MCP_JWT_SIGNING_KEY"),
        forward_pkce=True,
        forward_resource=False,
        enable_cimd=True,
        require_authorization_consent="external",
        extra_authorize_params={"access_type": "offline", "prompt": "select_account consent"},
    )
