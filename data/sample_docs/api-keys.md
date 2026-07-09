# API keys

API keys authenticate your application to Acme Cloud. Treat them like passwords:
never commit them to source control and never expose them in client-side code.

## Creating a key

Go to **Settings → API Keys** and click **Create key**. Choose an environment
(test or live) and the scopes the key needs. The secret is shown only once at
creation time — store it in a secrets manager.

## Rotating a key

To rotate an API key, go to **Settings → API Keys**, find the key, and click
**Rotate**. A new secret is generated immediately. The old secret stays valid for
**24 hours** so you can roll it out without downtime, then it is automatically
revoked. To revoke immediately instead, click **Revoke** rather than **Rotate**.

## Revoking a key

Click **Revoke** to disable a key at once. Any request using a revoked key fails
with `401 Unauthorized` and code `E_AUTH_INVALID`. Revocation cannot be undone —
create a new key if you revoked one by mistake.

## Best practices

- Use a separate key per service so you can rotate one without touching the others.
- Prefer narrowly scoped keys (`read` only where possible).
- Rotate keys on a schedule and whenever a team member with access leaves.
