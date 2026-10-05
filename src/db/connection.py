"""
src/db/connection.py
--------------------
Database connection utilities for the agentic-data-analyst project.

Usage
-----
    from src.db.connection import get_engine, get_session

    engine = get_engine()
    with get_session() as session:
        session.add(some_model_instance)
        session.commit()

Environment
-----------
Requires DATABASE_URL to be set in your .env file (or already in the
shell environment).  See .env.example for the expected format:
    DATABASE_URL=postgresql://user:password@host:5432/dbname
"""

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

# Load variables from .env into os.environ (no-op if already loaded)
load_dotenv()


def get_engine() -> Engine:
    """
    Create and return a SQLAlchemy Engine using DATABASE_URL from .env.

    The engine is created fresh on each call; callers that need a
    long-lived engine should cache the return value themselves.

    Raises
    ------
    ValueError
        If DATABASE_URL is not set in the environment.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is not set. "
            "Copy .env.example to .env and fill in your Neon connection string."
        )

    # pool_pre_ping=True keeps connections healthy across idle periods,
    # which is important for managed/serverless databases like Neon.
    engine = create_engine(database_url, pool_pre_ping=True)
    return engine


def get_session_factory(engine: Engine) -> sessionmaker:
    """
    Return a sessionmaker bound to the given engine.

    Parameters
    ----------
    engine : Engine
        A SQLAlchemy Engine (e.g. from get_engine()).

    Returns
    -------
    sessionmaker
        A factory that produces new Session objects when called.
    """
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a database Session and handles
    commit / rollback / close automatically.

    Example
    -------
        with get_session() as session:
            session.add(record)
            session.commit()
    """
    engine = get_engine()
    SessionLocal = get_session_factory(engine)
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
