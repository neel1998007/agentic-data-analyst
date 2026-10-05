"""
scripts/ingest_nhtsa_data.py
----------------------------
Fetches vehicle safety complaints from the public NHTSA Complaints API
and ingests them into the local `complaints` table.

API endpoint used
-----------------
    GET https://api.nhtsa.gov/complaints/complaintsByVehicle
        ?make={make}&model={model}&modelYear={year}

Usage (from the project root, with the venv activated)
------------------------------------------------------
    python -m scripts.ingest_nhtsa_data

The script is idempotent: complaints already present in the database
(identified by their unique odi_number) are silently skipped.
"""

from datetime import datetime, date
from typing import Optional

import requests
from sqlalchemy.exc import IntegrityError

from src.db.connection import get_session
from src.db.models import Complaint

# ── NHTSA API ─────────────────────────────────────────────────────────────────

NHTSA_BASE_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"

# Request timeout in seconds (connect timeout, read timeout)
REQUEST_TIMEOUT = (5, 30)

# ── Vehicles to ingest ────────────────────────────────────────────────────────

# Each tuple is (make, model, model_year).
# Extend this list to pull data for additional vehicles.
VEHICLES = [
    ("Honda",      "Civic",      2020),
    ("Toyota",     "Camry",      2020),
    ("Tesla",      "Model 3",    2021),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(value: Optional[str]) -> Optional[date]:
    """
    Convert an NHTSA date string to a Python date object.

    NHTSA dates are millisecond-epoch timestamps (e.g. 1609459200000)
    or ISO-8601 strings.  Returns None for missing / unparseable values.

    Parameters
    ----------
    value : str or None
        Raw date value from the NHTSA JSON response.

    Returns
    -------
    date or None
    """
    if not value:
        return None
    s = str(value).strip()
    try:
        # NHTSA sometimes returns epoch-milliseconds as an integer or string
        if s.lstrip("-").isdigit():
            return datetime.utcfromtimestamp(int(s) / 1000).date()
        # NHTSA complaint/incident dates arrive as MM/DD/YYYY
        if len(s) == 10 and s[2] == "/" and s[5] == "/":
            return datetime.strptime(s, "%m/%d/%Y").date()
        # ISO-8601 fallback (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        return datetime.fromisoformat(s[:10]).date()
    except (ValueError, OSError, OverflowError):
        return None


def _parse_bool(value) -> Optional[bool]:
    """
    Normalise the varied boolean representations used by NHTSA.

    The API may return True/False (bool), "Y"/"N", or "Yes"/"No".

    Parameters
    ----------
    value : any
        Raw value from the NHTSA JSON response.

    Returns
    -------
    bool or None
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() in ("Y", "YES", "TRUE", "1")
    return None


def fetch_complaints(make: str, model: str, year: int) -> list[dict]:
    """
    Call the NHTSA Complaints API for a single (make, model, year) and
    return the raw list of complaint dicts from the "results" key.

    Parameters
    ----------
    make : str
        Vehicle manufacturer (e.g. "Honda").
    model : str
        Vehicle model name (e.g. "Civic").
    year : int
        Four-digit model year (e.g. 2020).

    Returns
    -------
    list[dict]
        Raw complaint records as returned by the NHTSA API.

    Raises
    ------
    requests.HTTPError
        If the API responds with a 4xx or 5xx status code.
    requests.RequestException
        For network-level failures (timeout, DNS, etc.).
    """
    params = {"make": make, "model": model, "modelYear": year}
    response = requests.get(NHTSA_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("results", [])


def map_to_complaint(raw: dict) -> Optional[Complaint]:
    """
    Map a raw NHTSA complaint dict to a Complaint ORM instance.

    Vehicle identity (make, model, model_year) is nested inside the
    ``products`` array on each complaint.  A single complaint may have
    multiple product entries of different types (e.g. "Vehicle" and
    "Tire"); we always take the first one whose ``type`` == "Vehicle".

    Parameters
    ----------
    raw : dict
        A single element from the NHTSA API "results" array.

    Returns
    -------
    Complaint or None
        An unsaved SQLAlchemy ORM object ready to be added to a session,
        or ``None`` if no Vehicle product was found (caller should skip
        this record and log a warning).
    """
    odi = raw.get("odiNumber")

    # ── Locate the Vehicle product entry ─────────────────────────────
    products: list[dict] = raw.get("products") or []
    vehicle_product = next(
        (p for p in products if str(p.get("type", "")).strip().lower() == "vehicle"),
        None,
    )

    if vehicle_product is None:
        # Warn the caller; returning None signals that this row should be skipped.
        print(
            f"  ⚠️  No 'Vehicle' product found for ODI {odi} — skipping record."
        )
        return None

    # ── Extract vehicle identity from the matched product ─────────────
    raw_year = vehicle_product.get("productYear")
    try:
        model_year = int(raw_year) if raw_year is not None else None
    except (ValueError, TypeError):
        model_year = None

    return Complaint(
        odi_number=odi,
        manufacturer=raw.get("manufacturer"),
        make=vehicle_product.get("productMake"),
        model=vehicle_product.get("productModel"),
        model_year=model_year,
        crash=_parse_bool(raw.get("crash")),
        fire=_parse_bool(raw.get("fire")),
        num_injuries=raw.get("numberOfInjuries") or 0,
        num_deaths=raw.get("numberOfDeaths") or 0,
        date_of_incident=_parse_date(raw.get("dateOfIncident")),
        date_complaint_filed=_parse_date(raw.get("dateComplaintFiled")),
        components=raw.get("components"),
        summary=raw.get("summary"),
    )


# ── Main ingestion logic ──────────────────────────────────────────────────────

def ingest_all() -> None:
    """
    Iterate over VEHICLES, fetch complaints from NHTSA, and persist
    new records to the database.

    Duplicates (same odi_number) are skipped without raising an error.
    If any single vehicle API call fails, a warning is printed and the
    loop continues with the next vehicle.
    """
    total_inserted = 0
    total_skipped = 0

    with get_session() as session:
        for make, model, year in VEHICLES:
            vehicle_label = f"{year} {make} {model}"
            print(f"\n─── Fetching: {vehicle_label} ───")

            # ── Fetch from NHTSA ──────────────────────────────────────
            try:
                raw_complaints = fetch_complaints(make, model, year)
            except requests.RequestException as exc:
                print(f"  ⚠️  WARNING: API call failed for {vehicle_label}: {exc}")
                print("       Skipping this vehicle and continuing…")
                continue

            print(f"  → {len(raw_complaints)} complaint(s) returned by API")

            # ── Insert each complaint ─────────────────────────────────
            inserted = 0
            skipped = 0
            for raw in raw_complaints:
                complaint = map_to_complaint(raw)

                # Skip records where no Vehicle product was found (map_to_complaint
                # already printed a warning) or where ODI number is missing.
                if complaint is None:
                    skipped += 1
                    continue
                if complaint.odi_number is None:
                    print(f"  ⚠️  Missing odiNumber on record — skipping.")
                    skipped += 1
                    continue

                try:
                    session.add(complaint)
                    session.flush()   # detect IntegrityError early
                    inserted += 1
                except IntegrityError:
                    # Duplicate odi_number – roll back just this record
                    session.rollback()
                    skipped += 1

            # Commit after each vehicle so a failure later doesn't
            # roll back already-processed vehicles.
            session.commit()

            print(f"  ✅ Inserted: {inserted}  |  Skipped (duplicates): {skipped}")
            total_inserted += inserted
            total_skipped += skipped

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("INGESTION COMPLETE")
    print(f"  Total inserted : {total_inserted}")
    print(f"  Total skipped  : {total_skipped} (duplicates or missing ODI)")
    print("═" * 50)


if __name__ == "__main__":
    ingest_all()
