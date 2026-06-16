# ADR 0007: JWT (HS256) authentication (v1, locked)

- Status: Accepted (v1, frozen)
- Date: 2026-06-15
- Deciders: ML Platform, Security

## Context
The browser app and any non-browser callers need to authenticate to
`POST /v1/predict`. v1 should be simple, secure by default, and not
introduce a third-party auth provider or a managed secret store.

## Decision
- **HS256 JWTs** signed by the FastAPI app itself with a secret read
  from an environment variable (`JWT_SECRET`). In production, the
  variable is configured in the Render service settings. In local dev
  it lives in `.env`.
- `POST /v1/auth/token` issues a token given a device key (anonymous
  first-load registration) or a username/password (admin). TTL: 15
  min.
- Tokens carry `{sub, exp, iat, scope}`. Scope defaults to `predict`.
- A FastAPI dependency validates the token on every protected route.
- Rate limiting: token bucket per `sub` (default 60 req/min).
- The browser app reads the token from localStorage and attaches it as
  `Authorization: Bearer <token>`.
- **Removed from earlier draft**: AWS Secrets Manager. The Render
  service's env-var configuration is the v1 secret store.

## Consequences
- No external IdP to operate in v1.
- Symmetric key: anyone with the secret can mint tokens. Mitigation:
  the secret is set in Render's dashboard, never in the repo or `.env`
  that gets committed.
- If we add a real user system later, swap to RS256 + a proper IdP
  (Cognito, Auth0, etc.) without changing the client surface.

## Alternatives considered
- mTLS: heavy for a browser client, dropped.
- API keys: no expiry, no per-user rate limiting.
- AWS Secrets Manager: removed with the rest of the AWS footprint.
- OAuth2 / OIDC: overkill for v1.