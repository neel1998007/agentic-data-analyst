# Agentic Data Analyst

An AI agent that answers natural-language questions about NHTSA vehicle
safety complaint data — built as a portfolio project demonstrating
production-style AI engineering practices end-to-end.

---

## Project Overview

This project is a full-stack agentic AI system. A user types a plain-English
question about vehicle safety complaints ("How many fire complaints were filed
for the Tesla Model 3?", "Find complaints about brakes engaging on their own")
and a Claude-powered agent decides which tools to call, queries a live Postgres
database, and returns a synthesized, grounded answer with a full reasoning trace.

It demonstrates the complete engineering stack required for a production AI
feature: tool-calling agent design, RAG with vector similarity search,
observability via Langfuse tracing, an automated eval harness with LLM-as-judge
grading, and a React/TypeScript frontend that surfaces both the answer and the
tools used to produce it. Every layer was built deliberately rather than
scaffolded — the bugs, fixes, and design lessons below reflect what building
this in practice actually looks like.

---

## Architecture

```
Browser
        │  GET /  (static HTML/JS/CSS)
        ▼
Nginx  (frontend container, port 80)
  ├── Serves React SPA static files from /usr/share/nginx/html
  └── Reverse-proxies /ask and /health → http://backend:8000
        │  Docker internal network
        ▼
FastAPI backend  (src/api/main.py, backend container, port 8000)
        │  ask_agent(question)
        ▼
Claude Agent  (src/agents/claude_agent.py)
  ├── model: claude-sonnet-5-5
  ├── Tool 1: count_complaints     — structured SQL filter (make / model / year / fire / crash)
  └── Tool 2: semantic_search_complaints — pgvector cosine similarity search via sentence-transformers
        │
        │  SQLAlchemy + psycopg2
        ▼
Neon Postgres  (pgvector extension)
  └── complaints table  (~1,165 rows, 384-dim embeddings per complaint)

Deployment: two containers orchestrated by Docker Compose. The React app
is built to static files at image build time (multi-stage Dockerfile);
Nginx serves them and proxies API routes to the FastAPI container over
Docker's internal network so no CORS configuration is needed.

Observability: every ask_agent() call is wrapped in a @observe() decorator
and traced end-to-end in Langfuse, recording the question, tool calls,
tool results, and final answer in one unified trace.

Evals: an offline harness (src/evals/run_evals.py) runs 10 labelled test
cases through the live agent, grades each with either exact-count matching
(deterministic) or an LLM-as-judge (claude-sonnet-5-5, independent client),
and writes a timestamped JSON report to src/evals/results/.
```

---

## Tech Stack

- **Python 3.11** — runtime (required for sentence-transformers / torch compatibility)
- **FastAPI** — REST API backend
- **SQLAlchemy** — ORM and query builder
- **Anthropic Python SDK** — `claude-sonnet-5-5` for agent reasoning and LLM-as-judge eval
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embedding generation at ingest time
- **pgvector** — Postgres extension for cosine-similarity vector search
- **Neon** — serverless Postgres hosting
- **Langfuse** — LLM observability and tracing
- **React 18 + TypeScript** — frontend chat interface
- **Vite** — frontend build tooling
- **react-markdown** — renders Claude's markdown-formatted answers in the browser
- **Docker + Docker Compose** — containerises both services for reproducible deployment
- **Nginx** — serves the compiled React SPA and reverse-proxies API routes to FastAPI

---

## Setup & Running Locally

### Option A: Run with Docker Compose (recommended)

Requires Docker Desktop (or Docker Engine + Compose plugin) installed.

```bash
git clone https://github.com/your-username/agentic-data-analyst.git
cd agentic-data-analyst

# Copy the env template and fill in your API keys
cp .env.example .env
# Open .env and fill in:
#   ANTHROPIC_API_KEY   — from console.anthropic.com
#   DATABASE_URL        — Neon Postgres connection string (postgresql+psycopg2://...)
#   LANGFUSE_SECRET_KEY — from cloud.langfuse.com
#   LANGFUSE_PUBLIC_KEY — from cloud.langfuse.com
#   LANGFUSE_HOST       — https://cloud.langfuse.com

# Build images and start both services in the background
docker compose up -d --build
```

The app is then available at **http://localhost** (port 80, served by Nginx).
The FastAPI backend is also directly reachable at **http://localhost:8000/docs**.

To stop: `docker compose down`

---

### Option B: Run manually (two terminals)

> **Python 3.11 is required.** The `sentence-transformers` library pulls in
> `torch`, which has strict Python version constraints. Using 3.12+ will cause
> dependency resolution failures.

#### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-username/agentic-data-analyst.git
cd agentic-data-analyst

python3.11 -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

#### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in the same keys listed in Option A above.
```

#### 4. Run the backend

```bash
uvicorn src.api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

#### 5. Run the frontend (separate terminal)

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
# Vite proxies /ask and /health to localhost:8000 automatically.
```

### Running data pipeline scripts

Always run Python scripts from the project root using `-m`, not by file path:

```bash
# Create the database schema
python -m src.db.create_tables

# Ingest NHTSA complaint data (fetches + embeds + stores)
python -m scripts.ingest_nhtsa_data

# Run the full evaluation suite
python -m src.evals.run_evals
```

---

## Data Source

Complaint data is fetched from the [NHTSA Complaints API](https://api.nhtsa.gov/complaints/complaintsByVehicle).
Three vehicles were ingested, totalling approximately **1,165 complaints**:

| Vehicle | Model Year |
|---|---|
| Honda Civic | 2020 |
| Toyota Camry | 2020 |
| Tesla Model 3 | 2021 |

Each complaint record is stored with its ODI number, make, model, year, fire/crash
flags, and complaint summary text. At ingest time, the summary is embedded using
`sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and stored in a
`pgvector` column, enabling cosine-similarity retrieval at query time.

---

## Known Limitations & Engineering Findings

The following are real findings and bugs encountered during this build, documented
here because the debugging process is as instructive as the final result.

### NHTSA API — 400 Errors on Compound Model Names
The NHTSA Complaints API returns consistent HTTP 400 Bad Request errors when
querying compound or hyphenated model names (e.g. Ford `F-150`, Chevrolet
`Silverado`). Rather than investing time debugging a third-party API quirk
unrelated to the project's core goals, the dataset was scoped to three
high-volume vehicles with clean, single-token model names.

### Parallel Tool-Calling Bug
Claude models can emit multiple `tool_use` blocks in a single response turn when
answering compound questions. The original agent loop used `next()` to grab only
the first block, silently dropping the rest. The Anthropic API then rejected
the follow-up message because it referenced `tool_use_id` values that had no
matching `tool_result`. Fixed by iterating over all `tool_use` blocks, executing
each tool independently with a per-tool `try/except`, and batching all
`tool_result` objects into a single follow-up message.

### Eval Design — Subset Match vs. Exact Match for Tool Selection
The initial eval harness used an exact set match between `expected_tools` and
`tools_called`. This caused false failures: Claude would call
`semantic_search_complaints` twice in parallel (once per sub-topic) when the
spec only listed it once, or would add a second reasonable tool the spec didn't
anticipate. Updated to a subset check — `set(expected_tools).issubset(set(tools_called))`
— which verifies every expected tool was called without penalizing additional
valid tool use.

### Extended Thinking Token Exhaustion (Judge)
The LLM-as-judge function (using `claude-sonnet-5-5`) would occasionally return
an empty response because an internal `ThinkingBlock` consumed the full `max_tokens`
budget before any `TextBlock` was generated. Bare `next()` on the content list
raised `StopIteration` and crashed the harness. Fixed by: raising `max_tokens`
from 300 → 1024, looping through `response.content` to find the first block
where `block.type == "text"` (skipping ThinkingBlocks), and wrapping extraction
with `next(..., None)` plus a logged fallback.

### Extended Thinking Token Exhaustion (Main Agent)
The same `next(generator)` / `StopIteration` pattern existed in the main agent's
final-answer extraction at line 415. Additionally, `max_tokens=1024` on the
synthesis call was too small: after any thinking budget was consumed, insufficient
tokens remained for a full answer. Fixed by: raising `max_tokens` to 4096 on
the second (synthesis) API call, and switching to `next(..., None)` with a
graceful user-facing fallback message.

### ThinkingBlock / Tools Parameter Consistency (400 Error)
Anthropic's API enforces a hard constraint: if a `ThinkingBlock` appears in
conversation history passed to a follow-up API call, the `tools` parameter on
that call must be **identical** to the call that generated the thinking block.
The second agent call (synthesis) was omitting `tools=` entirely, causing a 400.
Fixed with two layers:
1. Strip `ThinkingBlock`s from the Turn 2 assistant history before building the
   messages list for the second call — pre-empts the constraint entirely
   (explicitly allowed by Anthropic's API docs).
2. Restore `tools=TOOLS` to the second call (same object reference as the first)
   as belt-and-suspenders, so even if a thinking block slips through the filter
   the constraint is still satisfied.

### Multi-Document Synthesis Overgeneralization
LLM-judge evaluation revealed a recurring failure mode: when synthesizing across
multiple complaint records, the agent used absolute language ("every case",
"all owners reported") even when one record was silent or differed. It also
blurred attribution between vehicle owners and manufacturers. Addressed with an
explicit system prompt directive requiring per-record precision, exception
call-outs, and party attribution. Immediately improved the eval suite from
**8/10 → 9/10 passing**.

### LLM-as-Judge Calibration Bias
The judge occasionally penalized the agent for describing its own retrieval
process (e.g. "searched by meaning") — categorizing process-description language
as an "ungrounded factual claim about vehicle complaints." Because the agent and
judge share the same model family, shared blind spots can occur. Judge scores
should be treated as a signal, not ground truth; qualitative review of the
`reasoning` field is necessary to distinguish genuine agent inaccuracies from
judge miscalibration.

### Top-K Retrieval Without a Similarity Threshold
The `semantic_search_complaints` tool returns the top-N most similar complaints
by cosine distance regardless of how similar they actually are. There is no
minimum relevance threshold, so marginally related or entirely unrelated records
can appear in results when the query is unusual. This is the root cause of the
one remaining eval failure (see below).

### Final Eval Status — 9/10 Passing
The eval suite achieves **9/10 passing**. The remaining failure is
`semantic_wheels_falling_off` ("Tell me about problems people had with their
wheels falling off"). The agent accurately summarized ODI 11499966 (Tesla:
explicit "no warning, not even a wobble, shimmy, clunk") and ODI 11748079
(Honda: vehicle information center gave no warning), but generalized the
no-warning finding to a third complaint (ODI 11455724) that made no mention of
warning signs. The judge correctly flagged this as ungrounded (`grounded: false`).
This is a low-severity residual of multi-document smoothing, not an architectural
flaw. Chasing 100% pass rate by overfitting the system prompt to this one case
was deliberately rejected in favour of documenting it as an understood edge case.

### CUDA Torch Bloat During Docker Build
Pip's default index includes the full CUDA-enabled GPU build of `torch` (a
dependency of `sentence-transformers`), which is several gigabytes. On the
deployment VM (Google Cloud e2-micro, 1 vCPU, 1 GB RAM, 10 GB disk), the
default install ran for ~40 minutes and ultimately corrupted the Docker image
export step with `failed call to UtimesNanoAt` — a containerd/overlayfs error
consistent with disk I/O exhaustion from writing a multi-GB image. Fixed by
adding `--extra-index-url https://download.pytorch.org/whl/cpu` to
`requirements.txt`, which forces pip to resolve the CPU-only torch wheel
(~200 MB instead of ~2 GB). The fix is committed to the repo so any
subsequent `docker compose up --build` completes in a few minutes.

### 1 GB RAM Deployment Constraint & Swap
The deployment target (Google Cloud e2-micro free tier) has 1 GB RAM shared
across Nginx, FastAPI, and the in-memory `sentence-transformers` model. The
Linux OOM killer terminated the backend process under memory pressure on first
deploy. Mitigated by provisioning a 2 GB swap file on the VM and making it
persistent via `/etc/fstab`. This is an explicit, documented tradeoff:
performance is suboptimal (swap is slow) but acceptable for a low-traffic
portfolio demo that doesn't require sub-second inference latency.

---

## Future Improvements

- **Conversation memory** — multi-turn context so follow-up questions can
  reference prior answers in the same session
- **Similarity threshold for RAG** — reject or down-weight retrieved complaints
  below a minimum cosine similarity score, reducing irrelevant context
- **Re-ranking** — apply a cross-encoder re-ranker after initial retrieval to
  improve relevance ordering before passing context to the agent
- **Expanded vehicle coverage** — ingest more makes/models to make the dataset
  more generally useful (currently limited by NHTSA API model-name constraints)
- **Human-reviewed eval sampling** — periodic manual review of LLM-judge
  decisions to detect and correct systematic calibration drift
- **Text-to-SQL** — replace fixed tool functions with a natural-language-to-SQL
  generator to handle arbitrary structured queries beyond the current
  `count_complaints` filter schema
- **HTTPS + custom domain** — add a domain name and provision a TLS certificate
  via Let's Encrypt / Certbot; currently the deployment runs plain HTTP because
  no custom domain is configured for this project
- **Docker buildx plugin** — the current build uses Docker's deprecated legacy
  builder; migrating to `buildx` enables multi-platform builds and BuildKit
  cache mounts for faster CI/CD pipelines

---

## Demo

🔗 **Live deployment: http://34.61.231.215/**

The app runs over plain HTTP (no custom domain or TLS configured). Nginx serves
the React frontend on port 80 and proxies `/ask` requests to the FastAPI
backend container on the same VM.

To run locally, follow the Setup instructions above.
