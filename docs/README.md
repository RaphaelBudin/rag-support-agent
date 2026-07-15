# docs/ — manual assets

The main README references **`docs/ui.png`** (in the *Thin chat UI* write-up). That image is a
manual asset — drop it here to make it render.

## How to capture it

1. Bring the stack up and ingest, then start the server (all keyless — no API key needed):

   ```bash
   docker compose up -d
   uv sync
   python -m rag_support_agent.ingestion.run --source data/sample_docs
   python -m rag_support_agent.api.server        # http://localhost:8000
   ```

2. Open <http://localhost:8000>, click the **"How do I rotate an API key?"** example chip (it
   answers with three citations and a confidence badge), and screenshot the answer card.
   Save it as **`docs/ui.png`**.

   - For an animated version, record a short GIF that also clicks **"How do I bake sourdough
     bread?"** to show the agent *abstaining* — that contrast (answers vs. knows-when-not-to) is
     the whole pitch. Save it as `docs/ui.gif` and, if you use it, point the README image at it.

## Demo video (the last portfolio-ready item)

Record a 2–3 min Loom walking through: ask → streamed answer with citations + confidence →
an abstention → the eval table (`eval.run`) → the gap report (`observability.gap_report`).
Link it at the top of the main README.
