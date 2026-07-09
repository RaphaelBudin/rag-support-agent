# Sample docs

Drop the neutral corpus to ingest here. Two clean-room options:

1. **Public open-source docs** — clone the docs of a permissively-licensed project
   (Markdown), keep the LICENSE, and note the source in the README.
2. **Synthetic SaaS docs** — generate docs for a fictional product ("Acme Cloud":
   API keys, billing, errors, webhooks, auth). Fully owned, zero provenance risk.

The eval set in `evaluation/datasets/support_qa.jsonl` references files like
`api-keys.md` and `errors.md` — create those (or repoint the dataset) so the
harness has gold sources to score against.

> Reminder: this repo is clean-room. No employer code or data — ever.
