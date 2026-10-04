# agentic-data-analyst

An AI agent that answers natural-language questions about automotive complaint data, powered by FastAPI, LangGraph, Anthropic Claude, and RAG with pgvector.

---

## Project Structure

```
agentic-data-analyst/
├── src/
│   ├── api/        # FastAPI route definitions and request/response schemas
│   ├── agents/     # LangGraph agent graphs, nodes, and tool definitions
│   ├── db/         # SQLAlchemy engine, session management, and ORM models
│   ├── rag/        # Embedding generation and pgvector similarity retrieval
│   └── evals/      # Offline evaluation harness scripts (not part of the API)
├── tests/          # Pytest test suite
├── venv/           # Python virtual environment (not committed)
├── .env            # Local secrets — never commit this (see .env.example)
├── .env.example    # Template showing required environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and fill in your values
cp .env.example .env
```
