"""
src/db/create_tables.py
-----------------------
One-shot script that creates all SQLAlchemy-defined tables in the
target database (if they do not already exist).

Run directly from the project root after activating the venv:

    python -m src.db.create_tables

Or from the src/db directory:

    python create_tables.py

Idempotent: safe to run multiple times; existing tables are untouched.
"""

from src.db.connection import get_engine
from src.db.models import Base


def create_all_tables() -> None:
    """
    Use SQLAlchemy's metadata to CREATE all tables registered on Base.

    Tables that already exist in the database are silently skipped
    (checkfirst=True is the default for create_all).
    """
    engine = get_engine()
    print(f"Connecting to database: {engine.url!r}")

    Base.metadata.create_all(bind=engine)

    table_names = list(Base.metadata.tables.keys())
    print(f"✅ Tables ensured: {table_names}")


if __name__ == "__main__":
    create_all_tables()
