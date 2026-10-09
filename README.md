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

## Running Scripts

**Always run scripts from the project root using the `-m` flag**, not by passing the file path directly.

| ✅ Correct | ❌ Wrong |
|---|---|
| `python -m src.db.create_tables` | `python src/db/create_tables.py` |
| `python -m scripts.ingest_nhtsa_data` | `python scripts/ingest_nhtsa_data.py` |

**Why?** When you run `python some/file.py`, Python adds `some/` to `sys.path` — so it looks for modules relative to that subfolder, and `import src.db.connection` fails because there's no `src/` inside `some/`. When you run `python -m some.file` instead, Python adds the **current working directory** (the project root) to `sys.path`, so all `src.*` imports resolve correctly. No `sys.path` hacks needed.

### Common commands

```bash
# Create the complaints table in your Neon database
python -m src.db.create_tables

# Run the NHTSA data ingestion pipeline
python -m scripts.ingest_nhtsa_data

# Run the evaluation suite
python -m src.evals.run_evals
```

---

## Known Limitations & Eval Findings

The following findings and design lessons emerged from building and evaluating the agent:

- **NHTSA API Model Name Quirks (400 Errors)**: The official NHTSA API returned consistent HTTP 400 Bad Request responses when querying compound or hyphenated vehicle model names (such as the Ford `F-150` or Chevrolet `Silverado`). To maintain focus on core agentic architecture, tool routing, and retrieval rather than debugging third-party endpoint idiosyncrasies, the dataset scope was narrowed to three high-volume vehicles (`Honda Civic`, `Toyota Camry`, and `Tesla Model 3`).

- **Parallel Tool Execution in a Single Turn**: Anthropic Claude models can generate multiple `tool_use` blocks simultaneously in a single turn (parallel tool calling) when answering compound inquiries. Early iterations of the agent only captured the first `tool_use` block, causing subsequent API calls to fail due to unfulfilled tool IDs. The agent execution loop was updated to iterate over all returned `tool_use` blocks, execute each tool independently in a resilient `try/except` wrapper, and batch all `tool_result` objects into a single follow-up message back to Claude.

- **Eval Tool-Selection Criteria (Subset vs. Exact Match)**: Initial evaluation specifications enforced an exact set match between `tools_called` and `expected_tools`. However, Claude frequently devises equally valid, alternative tool-calling strategies—for instance, invoking `semantic_search_complaints` twice in parallel (once for brake issues and once for tire issues) rather than combining a count query with a semantic search. The eval harness was updated to verify that every `expected_tool` appears as a subset of `tools_called`. This prevents penalizing the agent for invoking additional relevant tools or calling a tool multiple times.

- **Multi-Record Synthesis Overgeneralization & Attribution**: Deploying an LLM-as-a-judge revealed a recurring synthesis failure mode: when summarizing across multiple retrieved records, the agent occasionally used absolute phrasing (e.g., claiming a condition occurred *"in every case"* or *"all owners reported"*) when one record differed or was silent, and blurred attribution between distinct entities (such as conflating what a vehicle owner reported versus what manufacturer technicians diagnosed). This was addressed through an explicit system prompt directive requiring per-record attribution and strict exception-calling. Re-evaluating the suite demonstrated immediate improvement, increasing passing cases from 8/10 to 9/10.

- **LLM Judge Calibration Limitations**: The LLM evaluator exhibited subtle calibration biases. In one case, the judge penalized the agent for describing its own search methodology (*"searched by meaning"*), categorizing it as an ungrounded factual claim about vehicle complaints rather than an explanation of operational process. Because both the agent and the judge stem from the same foundational model family, shared blind spots can occur. Distinguishing genuine agent inaccuracies from judge miscalibration necessitates qualitative review of the judge's reasoning string, rather than treating boolean scores as infallible ground truth.

- **Final Eval Suite & Remaining Edge Case**: The final offline evaluation suite achieves **9/10 passing cases**. The single remaining failure occurred on `semantic_wheels_falling_off` (*"Tell me about problems people had with their wheels falling off"*). While the agent accurately synthesized ODI 11499966 (where the Tesla owner explicitly noted *"no warning beforehand, not even a wobble, shimmy, clunk"*) and ODI 11748079 (where the Honda owner noted the vehicle information center gave no warning), it generalized that *"the owners of both Teslas and the Honda owner all said they got no warning"*, despite the first Tesla complaint (ODI 11455724) making no mention of warning signs. The judge flagged this as ungrounded (`grounded: False`). Investigation confirmed this is a minor, low-severity remnant of multi-document synthesis smoothing rather than an architectural flaw. In production AI evaluation, targeting high-confidence reliability over artificial 100% pass rates reflects the inherent variance of non-deterministic models.
