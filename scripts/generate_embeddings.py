"""
scripts/generate_embeddings.py
------------------------------
Generates sentence embeddings for all complaints that don't have one yet
and writes them back to the `embedding` column in the database.

The model is loaded ONCE before the loop — sentence-transformers models
take several seconds to initialise, so never put the load call inside a
per-row loop.

Run from the project root:
    python -m scripts.generate_embeddings

Safe to re-run: the query filters for `WHERE embedding IS NULL`, so
rows that already have an embedding are completely untouched.
"""

import re

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from src.db.connection import get_engine, get_session_factory
from src.db.models import Complaint

# ── Constants ──────────────────────────────────────────────────────────────────

# The embedding model.  all-MiniLM-L6-v2 produces 384-dimensional vectors,
# is fast on CPU, and works well for semantic similarity on short paragraphs.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Commit to the database after this many rows to keep memory usage low and
# give visible progress checkpoints.  Tune up/down to taste.
BATCH_SIZE = 50

# Boilerplate injected by NHTSA's FOIA redaction process.  It adds length
# without adding meaning, so we strip it before encoding.
FOIA_BOILERPLATE = (
    "INFORMATION REDACTED PURSUANT TO THE FREEDOM OF INFORMATION ACT "
    "(FOIA), 5 U.S.C. 552(B)(6)"
)

# ── 1. Load the model once ────────────────────────────────────────────────────
# This happens at import time (module level) so the model is ready before
# we open a DB connection or start iterating rows.

print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' …")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model loaded.\n")


# ── 2. Text cleaning ──────────────────────────────────────────────────────────

def clean_summary(text: str | None) -> str:
    """
    Return a cleaned version of a complaint summary, ready for embedding.

    Steps
    -----
    1. Replace None / empty strings with an empty string so .encode()
       doesn't crash.
    2. Remove the FOIA boilerplate phrase (case-insensitive, anywhere in
       the text).
    3. Collapse runs of whitespace left behind by the removal.

    Parameters
    ----------
    text : str or None
        Raw summary text from the ``complaints`` table.

    Returns
    -------
    str
        Cleaned text, possibly empty if the entire summary was boilerplate.
    """
    if not text:
        return ""

    # Remove the FOIA phrase wherever it appears (case-insensitive).
    cleaned = re.sub(re.escape(FOIA_BOILERPLATE), "", text, flags=re.IGNORECASE)

    # Normalise leftover whitespace (double spaces, leading/trailing gaps).
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


# ── 3. Main embedding pipeline ────────────────────────────────────────────────

def generate_embeddings() -> None:
    """
    Query all complaints with a NULL embedding, encode their summaries,
    and write the vectors back to the database in batches.

    Progress is printed every BATCH_SIZE rows so you can see it's working.
    """
    engine = get_engine()
    SessionLocal = get_session_factory(engine)
    session = SessionLocal()

    try:
        # ── Fetch only rows that need embeddings ──────────────────────────
        # Using .all() loads all IDs into memory up front so we can report
        # the total count.  Summaries are fetched lazily below, but since
        # the table is small this is fine.  For very large tables, consider
        # using .yield_per(BATCH_SIZE) instead.
        stmt = (
            select(Complaint)
            .where(Complaint.embedding.is_(None))
            .order_by(Complaint.id)
        )
        complaints = session.execute(stmt).scalars().all()
        total = len(complaints)

        if total == 0:
            print("No complaints with missing embeddings found. Nothing to do.")
            return

        print(f"Found {total} complaint(s) with no embedding. Starting…\n")

        processed = 0

        for complaint in complaints:
            # ── a. Clean the summary text ─────────────────────────────────
            text = clean_summary(complaint.summary)

            # ── b. Encode → numpy array of shape (384,) ───────────────────
            # convert_to_numpy=True (the default) gives us a plain ndarray
            # which pgvector's SQLAlchemy type accepts directly.
            vector = model.encode(text, convert_to_numpy=True)

            # ── c. Write the vector back to the ORM object ────────────────
            complaint.embedding = vector.tolist()   # list[float] is safest
            processed += 1

            # ── d. Batch commit + progress report ─────────────────────────
            if processed % BATCH_SIZE == 0:
                session.commit()
                print(f"  Processed {processed}/{total} …")

        # Commit any remaining rows that didn't fill a full batch.
        session.commit()

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print("EMBEDDING GENERATION COMPLETE")
    print(f"  Embeddings generated : {processed}")
    print(f"  Skipped (had embedding): {0}")   # script only touches NULL rows
    print(f"{'═' * 50}")


if __name__ == "__main__":
    generate_embeddings()
