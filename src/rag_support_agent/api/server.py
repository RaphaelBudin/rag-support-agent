"""API + thin chat UI:  python -m rag_support_agent.api.server  ->  http://localhost:8000

Exposes /ask and serves the minimal chat UI. Responses carry the trust signals
(confidence badge, citations, stale flag) so the UI can show them.

TODO(M3/M8): wire answer_question + serve ui/.
"""

from __future__ import annotations


def create_app():
    """Build the FastAPI app. Imported lazily so the module stays import-cheap."""
    from fastapi import FastAPI

    app = FastAPI(title="RAG Support Agent")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # @app.post("/ask")  -> answer_question(query)  (TODO)
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
