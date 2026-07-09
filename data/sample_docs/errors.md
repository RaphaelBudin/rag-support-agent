# Errors

The Acme Cloud API uses conventional HTTP status codes and returns a machine-readable
`code` plus a human-readable `message` in the response body.

## Error format

```json
{ "error": { "code": "E_RATE_LIMIT", "message": "Too many requests." } }
```

## Common error codes

### E_AUTH_MISSING (401)
No `Authorization` header was sent. Add `Authorization: Bearer <API_KEY>`.

### E_AUTH_INVALID (401)
The key is wrong, revoked, or expired. Create or rotate a key.

### E_SCOPE (403)
The key lacks the scope for this operation. Issue a key with the `write` scope.

### E_RATE_LIMIT (429)
You exceeded the request quota for your plan. Back off and retry with **exponential
backoff**, respecting the `Retry-After` header. If you hit this consistently,
upgrade the plan for a higher quota.

### E_VALIDATION (422)
The request body failed validation. The `message` field names the offending field.

### E_INTERNAL (500)
Something went wrong on our side. Retry with backoff; if it persists, contact support.

## Retry guidance

Retry only idempotent requests (GET, PUT, DELETE) automatically. For POST, use an
`Idempotency-Key` header so a retry cannot create a duplicate resource.
