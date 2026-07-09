# Authentication

Acme Cloud uses bearer-token authentication. Every request to the API must include
an `Authorization: Bearer <API_KEY>` header. Requests without a valid key are
rejected with `401 Unauthorized` and the error code `E_AUTH_MISSING`.

## Getting started

1. Sign in to the Acme Cloud dashboard.
2. Open **Settings → API Keys**.
3. Create a key scoped to the environment you need (test or live).

Test keys are prefixed with `ak_test_` and live keys with `ak_live_`. Test keys can
only reach sandbox resources and never move real money or data.

## Scopes

Keys can be scoped to limit what they can do. A read-only key carries the `read`
scope; a key that can create or modify resources needs the `write` scope. Requests
that exceed a key's scope fail with `403 Forbidden` and code `E_SCOPE`.

## Session tokens

For browser apps, exchange your API key for a short-lived session token via
`POST /v1/sessions`. Session tokens expire after 60 minutes and should never be
embedded in client-side code.
