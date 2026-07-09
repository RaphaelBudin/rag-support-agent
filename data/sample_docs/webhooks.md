# Webhooks

Webhooks let Acme Cloud notify your application when events happen, instead of you
polling the API. Configure them under **Settings → Webhooks**.

## Creating an endpoint

Register an HTTPS URL to receive events. We send a `POST` with a JSON body for each
event. Your endpoint must respond with a `2xx` status within 5 seconds, or the
delivery is considered failed.

## Verifying signatures

Every webhook request includes an `Acme-Signature` header: an HMAC-SHA256 of the raw
request body using your webhook signing secret. Recompute the HMAC and compare it in
constant time before trusting the payload. Reject the request if it does not match.

## Retries

Failed deliveries are retried with exponential backoff for up to 24 hours. After
that the event is dropped. Use the `event.id` field to make your handler idempotent —
the same event may be delivered more than once.

## Event types

Common events include `resource.created`, `resource.updated`, `resource.deleted`,
and `invoice.paid`. Subscribe only to the events you need to reduce traffic.
