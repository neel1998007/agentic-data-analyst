"""
src/db/models.py
----------------
SQLAlchemy ORM models for the agentic-data-analyst project.

Each class maps to one database table via the DeclarativeBase pattern.
Add new models here as the schema grows.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base – every ORM model inherits from this."""
    pass


class Complaint(Base):
    """
    Represents a single vehicle safety complaint sourced from the NHTSA API.

    Table name: complaints
    Primary key: id (auto-incremented surrogate key)
    Natural key: odi_number (NHTSA's own unique complaint identifier)
    """

    __tablename__ = "complaints"
    __table_args__ = (
        UniqueConstraint("odi_number", name="uq_complaints_odi_number"),
    )

    # ── Surrogate primary key ──────────────────────────────────────────────
    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── NHTSA identifiers ─────────────────────────────────────────────────
    odi_number = Column(BigInteger, nullable=False, unique=True, index=True,
                        comment="NHTSA ODI complaint number – globally unique")

    # ── Vehicle identifiers ───────────────────────────────────────────────
    manufacturer = Column(String(255), nullable=True)
    make = Column(String(100), nullable=True)
    model = Column(String(100), nullable=True)
    model_year = Column(Integer, nullable=True)

    # ── Incident flags ────────────────────────────────────────────────────
    crash = Column(Boolean, nullable=True, comment="Was a crash involved?")
    fire = Column(Boolean, nullable=True, comment="Was a fire involved?")

    # ── Casualties ────────────────────────────────────────────────────────
    num_injuries = Column(Integer, nullable=True, default=0)
    num_deaths = Column(Integer, nullable=True, default=0)

    # ── Dates ─────────────────────────────────────────────────────────────
    date_of_incident = Column(Date, nullable=True)
    date_complaint_filed = Column(Date, nullable=True)

    # ── Descriptive fields ────────────────────────────────────────────────
    components = Column(Text, nullable=True,
                        comment="Comma-separated list of affected components")
    summary = Column(Text, nullable=True,
                     comment="Full narrative text of the complaint")

    def __repr__(self) -> str:
        return (
            f"<Complaint odi_number={self.odi_number} "
            f"{self.model_year} {self.make} {self.model}>"
        )
