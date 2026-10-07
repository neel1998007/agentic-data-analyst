"""
src/agents/tools.py
-------------------
Queryable functions that the LangGraph agent can call as tools.

Each function accepts optional filter arguments, builds a SQLAlchemy
query against the ``complaints`` table, and returns a Python-native
result (int or list[dict]) that is easy to serialise into a tool-call
response.

All string comparisons use ``ilike`` (case-insensitive LIKE) so that
the agent doesn't need to worry about casing differences between how
NHTSA stores data and how the user phrases a question.
"""

from __future__ import annotations

from typing import Optional

from sentence_transformers import SentenceTransformer
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.db.connection import get_engine, get_session_factory
from src.db.models import Complaint

# ── Embedding model (loaded once at import time) ───────────────────────────────
# Loading a SentenceTransformer downloads weights and warms the tokeniser —
# this takes a few seconds and must NOT be done inside a per-call function.
# By placing it here at module level it runs exactly once, no matter how many
# times semantic_search_complaints() is called.

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_filters(query, make, model, model_year, fire, crash):
    """
    Attach WHERE clauses to *query* for every argument that is not None.

    Parameters
    ----------
    query       : A SQLAlchemy Select or Query object.
    make        : Optional vehicle make filter (case-insensitive).
    model       : Optional vehicle model filter (case-insensitive).
    model_year  : Optional exact model year filter.
    fire        : Optional boolean filter — True = fire involved.
    crash       : Optional boolean filter — True = crash involved.

    Returns
    -------
    The query with all applicable filters chained on.
    """
    if make is not None:
        query = query.where(Complaint.make.ilike(f"%{make}%"))
    if model is not None:
        query = query.where(Complaint.model.ilike(f"%{model}%"))
    if model_year is not None:
        query = query.where(Complaint.model_year == model_year)
    if fire is not None:
        query = query.where(Complaint.fire == fire)
    if crash is not None:
        query = query.where(Complaint.crash == crash)
    return query


def _get_session() -> Session:
    """Return a new SQLAlchemy Session using the shared engine."""
    engine = get_engine()
    SessionLocal = get_session_factory(engine)
    return SessionLocal()


# ── Public tool functions ─────────────────────────────────────────────────────

def count_complaints(
    make: Optional[str] = None,
    model: Optional[str] = None,
    model_year: Optional[int] = None,
    fire: Optional[bool] = None,
    crash: Optional[bool] = None,
) -> int:
    """
    Count vehicle safety complaints that match the supplied filters.

    All parameters are optional. Any parameter left as ``None`` is
    simply omitted from the WHERE clause, so calling the function with
    no arguments returns the total number of complaints in the table.

    String filters (``make``, ``model``) use case-insensitive partial
    matching (SQL ``ILIKE '%value%'``), so "tesla", "Tesla", and
    "TESLA" all match the same rows.

    Parameters
    ----------
    make : str, optional
        Vehicle manufacturer/make (e.g. "Toyota", "Tesla").
    model : str, optional
        Vehicle model name (e.g. "Camry", "Model 3").
    model_year : int, optional
        Four-digit model year (e.g. 2021).
    fire : bool, optional
        If True, only include complaints where a fire was involved.
        If False, only include complaints where no fire was reported.
    crash : bool, optional
        If True, only include complaints where a crash was involved.
        If False, only include complaints where no crash was reported.

    Returns
    -------
    int
        The number of complaint rows matching all supplied filters.

    Examples
    --------
    >>> count_complaints()                          # total in table
    >>> count_complaints(make="Tesla")              # all Tesla complaints
    >>> count_complaints(make="Honda", fire=True)   # Honda complaints with fire
    """
    session = _get_session()
    try:
        stmt = select(func.count()).select_from(Complaint)
        stmt = _apply_filters(stmt, make, model, model_year, fire, crash)
        result: int = session.execute(stmt).scalar_one()
        return result
    finally:
        session.close()


def get_sample_complaints(
    make: Optional[str] = None,
    model: Optional[str] = None,
    model_year: Optional[int] = None,
    fire: Optional[bool] = None,
    crash: Optional[bool] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Return up to ``limit`` full complaint records matching the supplied filters.

    Applies the same optional, case-insensitive filtering logic as
    :func:`count_complaints`, but returns the actual complaint data
    (including the full ``summary`` text) rather than just a count.

    Records are returned as plain dictionaries so they are easy to
    serialise into a tool-call response or pass to an LLM as context.

    Parameters
    ----------
    make : str, optional
        Vehicle manufacturer/make (e.g. "Toyota", "Tesla").
    model : str, optional
        Vehicle model name (e.g. "Camry", "Model 3").
    model_year : int, optional
        Four-digit model year (e.g. 2021).
    fire : bool, optional
        If True, only include complaints where a fire was involved.
    crash : bool, optional
        If True, only include complaints where a crash was involved.
    limit : int, optional
        Maximum number of records to return (default: 5).

    Returns
    -------
    list[dict]
        Each dict contains all scalar columns from the ``complaints``
        table.  Date fields are converted to ISO-8601 strings so the
        result is JSON-serialisable without extra processing.

    Examples
    --------
    >>> get_sample_complaints(make="Tesla", limit=3)
    >>> get_sample_complaints(model_year=2021, crash=True, limit=10)
    """
    session = _get_session()
    try:
        stmt = select(Complaint)
        stmt = _apply_filters(stmt, make, model, model_year, fire, crash)
        stmt = stmt.limit(limit)
        rows = session.execute(stmt).scalars().all()

        return [
            {
                "id":                   row.id,
                "odi_number":           row.odi_number,
                "manufacturer":         row.manufacturer,
                "make":                 row.make,
                "model":                row.model,
                "model_year":           row.model_year,
                "crash":                row.crash,
                "fire":                 row.fire,
                "num_injuries":         row.num_injuries,
                "num_deaths":           row.num_deaths,
                # Convert date objects → ISO strings for JSON-friendliness
                "date_of_incident":     row.date_of_incident.isoformat()
                                        if row.date_of_incident else None,
                "date_complaint_filed": row.date_complaint_filed.isoformat()
                                        if row.date_complaint_filed else None,
                "components":           row.components,
                "summary":              row.summary,
            }
            for row in rows
        ]
    finally:
        session.close()


def semantic_search_complaints(query: str, limit: int = 5) -> list[dict]:
    """
    Find complaints whose summary text is semantically similar to ``query``.

    Unlike :func:`count_complaints` and :func:`get_sample_complaints`, which
    do exact structured filtering (make == "Tesla", fire == True, etc.), this
    function understands *meaning*.  A query like "engine suddenly stalled on
    the highway" will match complaints that describe that experience even if
    none of the exact words appear — because both the query and the complaints
    are projected into the same 384-dimensional embedding space, and proximity
    in that space corresponds to semantic similarity.

    Internally this uses pgvector's cosine distance operator (<=>), which
    measures the angle between two vectors.  Complaints are ranked from most
    similar (smallest angle / distance) to least similar.

    Requires the ``embedding`` column to be populated for rows to surface in
    results.  Run ``python -m scripts.generate_embeddings`` first if needed.

    Parameters
    ----------
    query : str
        A natural-language description of the kind of complaint to find,
        e.g. "car suddenly lost power while driving".
    limit : int, optional
        Maximum number of results to return (default: 5).

    Returns
    -------
    list[dict]
        Each dict contains: odi_number, make, model, model_year, components,
        and summary — the fields most useful for presenting results to a user
        or passing as context to an LLM.

    Examples
    --------
    >>> semantic_search_complaints("sudden unintended acceleration")
    >>> semantic_search_complaints("battery fire while parked", limit=10)
    """
    # ── Step 1: Embed the query using the pre-loaded model ────────────────
    # We reuse the module-level _embedding_model so we never pay the
    # model-loading cost more than once per process.
    query_vector = _embedding_model.encode(query, convert_to_numpy=True).tolist()

    # ── Step 2: Build the cosine-distance query ───────────────────────────
    # pgvector exposes <=> as .cosine_distance() on the column comparator.
    # ORDER BY distance ASC → most similar row first.
    # We also filter for rows where embedding IS NOT NULL so that any rows
    # without embeddings are silently excluded rather than causing errors.
    session = _get_session()
    try:
        stmt = (
            select(Complaint)
            .where(Complaint.embedding.is_not(None))
            .order_by(Complaint.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        rows = session.execute(stmt).scalars().all()

        # ── Step 3: Serialise to plain dicts ──────────────────────────────
        # Return only the fields useful for RAG context / display.
        # The embedding vector itself is intentionally excluded — it's large
        # and meaningless to a human or an LLM reading the result.
        return [
            {
                "odi_number": row.odi_number,
                "make":       row.make,
                "model":      row.model,
                "model_year": row.model_year,
                "components": row.components,
                "summary":    row.summary,
            }
            for row in rows
        ]
    finally:
        session.close()


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running smoke test: count_complaints(make='Tesla')")
    total = count_complaints(make="Tesla")
    print(f"  → {total} Tesla complaint(s) in the database.")

    if total > 0:
        print("\nRunning smoke test: get_sample_complaints(make='Tesla', limit=2)")
        samples = get_sample_complaints(make="Tesla", limit=2)
        for i, c in enumerate(samples, 1):
            print(f"\n  [{i}] ODI {c['odi_number']} | {c['model_year']} {c['make']} {c['model']}")
            print(f"       Fire: {c['fire']}  Crash: {c['crash']}")
            summary_preview = (c["summary"] or "")[:120]
            print(f"       Summary: {summary_preview}{'...' if len(c['summary'] or '') > 120 else ''}")

    print("\n" + "─" * 60)
    print("Running smoke test: semantic_search_complaints('car suddenly lost power while driving')")
    results = semantic_search_complaints("car suddenly lost power while driving", limit=3)
    if not results:
        print("  No results — have you run generate_embeddings.py yet?")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] ODI {r['odi_number']} | {r['model_year']} {r['make']} {r['model']}")
        print(f"       Components: {r['components']}")
        preview = (r["summary"] or "")[:150]
        print(f"       Summary: {preview}{'...' if len(r['summary'] or '') > 150 else ''}")
