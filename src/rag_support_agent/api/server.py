"""API + thin chat UI:  python -m rag_support_agent.api.server  ->  http://localhost:8000

Serves a thin, vanilla chat UI and two answer endpoints over the M8 pipeline:

  - ``POST /ask``         -> the full grounded ``Answer`` as JSON (the tested data contract).
  - ``GET  /ask/stream``  -> the *same* answer over Server-Sent Events: real token streaming on
                             the gemini/openai path, a single whole-answer event on the keyless
                             extractive path (which has no incremental output). The UI renders
                             incrementally either way.

Both pass ``record_event=True``, so real UI traffic feeds the M7 blind-spot log — the
knowledge-gap report then closes the loop over *live* questions, not just a replayed eval set.

Responses carry every trust signal (confidence, citations, stale flag, cost/latency) so the UI
is pure render; the numbers are all computed upstream in ``generation.answer``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

_UI_INDEX = Path(__file__).parent / "ui" / "index.html"


class AskRequest(BaseModel):
    """POST /ask body. Defined at module level (not inside ``create_app``) so FastAPI's
    ``get_type_hints`` can resolve the annotation — with ``from __future__ import annotations``
    a locally-scoped model would be an unresolvable forward-ref and silently fall back to
    query params."""

    query: str
    top_k: int | None = None


def create_app():
    """Build the FastAPI app. Heavy imports are local so the module stays import-cheap."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse

    from rag_support_agent.api.serialize import serialize_answer, sse_event
    from rag_support_agent.config import get_settings
    # Import the module (not the names) so tests can monkeypatch answer_question /
    # answer_question_stream on it and the handlers pick the patched versions up at call time.
    from rag_support_agent.generation import answer as answer_mod

    app = FastAPI(title="RAG Support Agent")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _UI_INDEX.read_text(encoding="utf-8")

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        query = req.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        s = get_settings()
        answer = answer_mod.answer_question(
            query, top_k=req.top_k, settings=s, record_event=True
        )
        return serialize_answer(answer, query, s.llm_provider)

    @app.get("/ask/stream")
    def ask_stream(q: str, top_k: int | None = None):
        query = q.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        s = get_settings()

        def events():
            # A stream error must not leave the browser's EventSource hanging (and silently
            # reconnecting): surface it as a terminal `error` event the UI can render.
            try:
                for kind, payload in answer_mod.answer_question_stream(
                    query, top_k=top_k, settings=s, record_event=True
                ):
                    if kind == "token":
                        yield sse_event("token", {"text": payload})
                    else:  # ("answer", Answer) — the final frame with all trust signals
                        yield sse_event("done", serialize_answer(payload, query, s.llm_provider))
            except Exception as exc:  # noqa: BLE001 — report any failure to the client, don't hang
                yield sse_event("error", {"message": str(exc)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
