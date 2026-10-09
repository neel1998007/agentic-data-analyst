"""
src/api/main.py
---------------
FastAPI application entry point for the agentic-data-analyst backend.

Run the development server from the project root:
    uvicorn src.api.main:app --reload --port 8000

API docs are auto-generated at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.claude_agent import ask_agent

logger = logging.getLogger(__name__)

# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Data Analyst",
    description="AI agent that answers natural-language questions about NHTSA vehicle safety complaints.",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the React / Vite dev servers to call this API during local development.
# Tighten these origins before deploying to production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Create React App default
        "http://localhost:5173",   # Vite default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health_check() -> dict:
    """
    Liveness probe — returns 200 OK when the server is running.

    Useful for deployment health checks and smoke-testing the API is up
    before sending real requests.
    """
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
def ask(request: AskRequest) -> AskResponse:
    """
    Send a natural-language question to the Claude agent and get an answer.

    The agent may call internal tools (e.g. querying the complaints database)
    before composing its final response.  All of that happens transparently —
    the caller just sends a question and receives a plain-English answer.

    Returns HTTP 500 with a JSON error body if the agent raises any exception,
    so the client always receives a structured response rather than a crash.
    """
    logger.info("Received question: %s", request.question)

    try:
        result = ask_agent(request.question)
    except Exception as exc:
        logger.exception("ask_agent() raised an unexpected error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"The agent encountered an error: {exc}",
        )

    return AskResponse(answer=result["answer"])
