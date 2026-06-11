"""
SQLite Loader — ingestion/sqlite_loader.py

Creates and populates the MedDocAI SQLite database.

Tables:
  patient_summaries  — parsed + enriched patient records (from FHIR + batch pipeline)
  nadac_pricing      — drug prices (most recent per NDC, loaded from nadac_2026.csv)
  eligibility        — Medicaid income eligibility by state (wide format)
  chat_history       — per-session chat messages with source citations

Usage:
    from ingestion.sqlite_loader import MedDocDB

    db = MedDocDB("data/meddocai.db")
    db.create_tables()
    db.load_nadac("data/cms_medicaid/nadac_2026.csv")
    db.load_eligibility("data/cms_medicaid/eligibility_levels.csv")

    # Save a parsed patient (pre-summary)
    db.save_patient(parsed_patient)

    # Retrieve for UI dropdown / agent context
    patients = db.list_patients()
    patient = db.get_patient("some-uuid")

    # Agent tools
    price = db.get_drug_price("00069420030")
    eligibility = db.get_state_eligibility("Texas")
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from models.schemas import ParsedPatient

logger = logging.getLogger(__name__)

# Default DB path — can be overridden
DEFAULT_DB_PATH = "data/meddocai.db"


# ---------------------------------------------------------------------------
# DDL — table definitions
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS patient_summaries (
    patient_id        TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    age               INTEGER,
    gender            TEXT,
    conditions_json   TEXT,     -- JSON: list of Condition dicts
    medications_json  TEXT,     -- JSON: list of Medication dicts (all statuses)
    labs_json         TEXT,     -- JSON: list of LabResult dicts
    encounters_json   TEXT,     -- JSON: list of Encounter dicts
    summary_md        TEXT,     -- Generated markdown summary (NULL until batch pipeline)
    generated_at      TEXT,     -- ISO datetime when summary was generated
    fhir_path         TEXT      -- Absolute path to source FHIR file
);

CREATE TABLE IF NOT EXISTS nadac_pricing (
    ndc               TEXT PRIMARY KEY,
    drug_name         TEXT,
    price_per_unit    REAL,
    pricing_unit      TEXT,     -- EA, ML, GM
    otc               TEXT,     -- 'Y' or 'N'
    effective_date    TEXT,     -- ISO date YYYY-MM-DD (most recent)
    as_of_date        TEXT      -- Report as-of date
);

CREATE TABLE IF NOT EXISTS eligibility (
    state             TEXT PRIMARY KEY,
    notes             TEXT,
    medicaid_0_1      TEXT,     -- Income limit as '141%' string
    medicaid_1_5      TEXT,
    medicaid_6_18     TEXT,
    chip_separate     TEXT,
    pregnant_medicaid TEXT,
    pregnant_chip     TEXT,
    parent_caretaker  TEXT,
    expansion_adults  TEXT,     -- '133%' | 'No' | '' | '200%' etc.
    chip_ages         TEXT,
    parent_income_std TEXT      -- 'FPL' or 'Dollars'
);

CREATE TABLE IF NOT EXISTS chat_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT    NOT NULL,
    user_id        TEXT,         -- NULL for anonymous
    patient_id     TEXT,         -- NULL if no patient context was active
    persona        TEXT,         -- 'patient' | 'care_manager' (who was asking)
    role           TEXT    NOT NULL,  -- 'user' | 'assistant'
    content        TEXT    NOT NULL,
    sources_json   TEXT,         -- JSON: assistant render data (evidence/disclaimers/trace)
    timestamp      TEXT    NOT NULL   -- ISO datetime
);

CREATE INDEX IF NOT EXISTS idx_nadac_drug_name  ON nadac_pricing(drug_name);
CREATE INDEX IF NOT EXISTS idx_chat_session     ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_patient     ON chat_history(patient_id);
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MedDocDB:
    """Thin wrapper around a SQLite connection for the MedDocAI project.

    All methods acquire and release connections via a context manager —
    safe for single-threaded Streamlit use.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    # ── Connection management ──────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection, commit on success, rollback on error."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row          # rows behave like dicts
        conn.execute("PRAGMA journal_mode=WAL") # safer concurrent reads
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema ─────────────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """Create all tables and indexes (idempotent — safe to call repeatedly)."""
        with self._conn() as conn:
            conn.executescript(_DDL)
            # Lightweight migration: add chat_history.persona to pre-existing DBs
            cols = {row["name"] for row in
                    conn.execute("PRAGMA table_info(chat_history)").fetchall()}
            if "persona" not in cols:
                conn.execute("ALTER TABLE chat_history ADD COLUMN persona TEXT")
                logger.info("Migrated chat_history: added persona column")
        logger.info("Tables created/verified in %s", self.db_path)

    # ── NADAC pricing ──────────────────────────────────────────────────────

    def load_nadac(self, csv_path: str, batch_size: int = 5000) -> int:
        """Load NADAC pricing CSV, keeping only the most recent row per NDC.

        Strategy (Option B — decided by user):
          - Group all rows by NDC
          - Keep the row with the latest Effective Date
          - Insert via INSERT OR REPLACE (upsert on primary key)

        Returns number of rows inserted.
        """
        logger.info("Loading NADAC from %s ...", csv_path)

        # Group rows by NDC, keep most recent effective date
        ndc_best: dict[str, dict] = {}
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ndc = row["NDC"].strip()
                if not ndc:
                    continue
                date_str = row["Effective Date"].strip()
                try:
                    row_date = datetime.strptime(date_str, "%m/%d/%Y")
                except ValueError:
                    continue

                existing = ndc_best.get(ndc)
                if existing is None:
                    ndc_best[ndc] = {**row, "_date_obj": row_date}
                else:
                    if row_date > existing["_date_obj"]:
                        ndc_best[ndc] = {**row, "_date_obj": row_date}

        # Convert dates to ISO format and insert in batches
        rows_to_insert = []
        for row in ndc_best.values():
            try:
                iso_effective = row["_date_obj"].strftime("%Y-%m-%d")
                as_of_raw = row.get("As of Date", "").strip()
                try:
                    iso_as_of = datetime.strptime(as_of_raw, "%m/%d/%Y").strftime("%Y-%m-%d")
                except ValueError:
                    iso_as_of = as_of_raw

                price_str = row.get("NADAC Per Unit", "").strip()
                price = float(price_str) if price_str else None

                rows_to_insert.append((
                    row["NDC"].strip(),
                    row.get("NDC Description", "").strip(),
                    price,
                    row.get("Pricing Unit", "").strip(),
                    row.get("OTC", "").strip(),
                    iso_effective,
                    iso_as_of,
                ))
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping NADAC row: %s", exc)

        with self._conn() as conn:
            for i in range(0, len(rows_to_insert), batch_size):
                batch = rows_to_insert[i : i + batch_size]
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO nadac_pricing
                        (ndc, drug_name, price_per_unit, pricing_unit, otc,
                         effective_date, as_of_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )

        logger.info("Loaded %d NDC pricing records", len(rows_to_insert))
        return len(rows_to_insert)

    def get_drug_price(self, ndc: str) -> Optional[dict]:
        """Look up pricing for a single NDC code.

        Returns a dict with keys: ndc, drug_name, price_per_unit, pricing_unit,
        otc, effective_date — or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM nadac_pricing WHERE ndc = ?", (ndc.strip(),)
            ).fetchone()
        return dict(row) if row else None

    def search_drug_price_by_name(self, drug_name: str, limit: int = 10) -> list[dict]:
        """Search NADAC by drug name (case-insensitive partial match).

        Useful for agents that have a drug name but not an NDC code.
        Returns up to `limit` matching rows sorted by drug name.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM nadac_pricing
                WHERE LOWER(drug_name) LIKE LOWER(?)
                ORDER BY drug_name
                LIMIT ?
                """,
                (f"%{drug_name}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Eligibility ────────────────────────────────────────────────────────

    def load_eligibility(self, csv_path: str) -> int:
        """Load Medicaid eligibility CSV (wide format, one row per state).

        Returns number of rows inserted.
        """
        logger.info("Loading eligibility from %s ...", csv_path)

        rows_to_insert = []
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = row.get("State", "").strip()
                if not state:
                    continue
                rows_to_insert.append((
                    state,
                    row.get("Notes", "").strip(),
                    row.get("Medicaid Ages 0-1", "").strip(),
                    row.get("Medicaid Ages 1-5", "").strip(),
                    row.get("Medicaid Ages 6-18", "").strip(),
                    row.get("Separate CHIP", "").strip(),
                    row.get("Pregnant Women Medicaid", "").strip(),
                    row.get("Pregnant Women CHIP", "").strip(),
                    row.get("Parent/Caretaker", "").strip(),
                    row.get("Expansion to Adults", "").strip(),
                    row.get("Separate CHIP Ages", "").strip(),
                    row.get("Parent/Caretaker Income Standard", "").strip(),
                ))

        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO eligibility
                    (state, notes, medicaid_0_1, medicaid_1_5, medicaid_6_18,
                     chip_separate, pregnant_medicaid, pregnant_chip,
                     parent_caretaker, expansion_adults, chip_ages,
                     parent_income_std)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )

        logger.info("Loaded %d state eligibility records", len(rows_to_insert))
        return len(rows_to_insert)

    def get_state_eligibility(self, state: str) -> Optional[dict]:
        """Return Medicaid eligibility info for a state (case-insensitive).

        Returns a dict with all eligibility columns, or None if state not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM eligibility WHERE LOWER(state) = LOWER(?)",
                (state.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def list_states(self) -> list[str]:
        """Return all state names in the eligibility table."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT state FROM eligibility ORDER BY state"
            ).fetchall()
        return [r["state"] for r in rows]

    # ── Patient summaries ──────────────────────────────────────────────────

    def save_patient(
        self,
        patient: ParsedPatient,
        summary_md: Optional[str] = None,
    ) -> None:
        """Insert or update a patient record.

        Can be called twice:
          1. After FHIR parsing — summary_md is None
          2. After batch pipeline — summary_md is the generated markdown

        Uses INSERT OR REPLACE so both calls are idempotent.
        """
        generated_at = datetime.utcnow().isoformat() if summary_md else None

        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO patient_summaries
                    (patient_id, name, age, gender,
                     conditions_json, medications_json,
                     labs_json, encounters_json,
                     summary_md, generated_at, fhir_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient.patient_id,
                    patient.name,
                    patient.age,
                    patient.gender,
                    json.dumps([c.model_dump() for c in patient.conditions]),
                    json.dumps([m.model_dump() for m in patient.medications]),
                    json.dumps([l.model_dump() for l in patient.labs]),
                    json.dumps([e.model_dump() for e in patient.encounters]),
                    summary_md,
                    generated_at,
                    patient.fhir_path,
                ),
            )

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Retrieve one patient record by ID.

        Returns a dict with all columns including parsed JSON fields,
        or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM patient_summaries WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        # Deserialize JSON fields
        for field in ("conditions_json", "medications_json",
                      "labs_json", "encounters_json"):
            if result.get(field):
                result[field] = json.loads(result[field])
        return result

    def list_patients(self, limit: int = 500) -> list[dict]:
        """Return a lightweight list of all stored patients for the UI dropdown.

        Returns: list of {patient_id, name, age, gender, has_summary} dicts.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT patient_id, name, age, gender,
                       (summary_md IS NOT NULL) AS has_summary
                FROM patient_summaries
                ORDER BY name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_patients(self) -> int:
        """Return total number of stored patient records."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM patient_summaries"
            ).fetchone()[0]

    def count_patients_with_summary(self) -> int:
        """Return number of patients who have a generated summary."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM patient_summaries WHERE summary_md IS NOT NULL"
            ).fetchone()[0]

    # ── Chat history ───────────────────────────────────────────────────────

    def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        patient_id: Optional[str] = None,
        persona: Optional[str] = None,
        user_id: Optional[str] = None,
        sources: Optional[dict | list] = None,
    ) -> None:
        """Append a single message to chat history (saved immediately, per-message).

        `persona` is the asker's role ('patient' | 'care_manager') — used to key
        history so a patient and a care manager see separate threads about the
        same patient. `sources` is an arbitrary JSON-serialisable blob (for
        assistant messages we store evidence/disclaimers/trace for re-rendering).
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_history
                    (session_id, user_id, patient_id, persona, role,
                     content, sources_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    patient_id,
                    persona,
                    role,
                    content,
                    json.dumps(sources) if sources is not None else None,
                    datetime.utcnow().isoformat(),
                ),
            )

    def get_chat_history(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """Return the most recent `limit` messages for a session."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, sources_json, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        messages = [dict(r) for r in reversed(rows)]
        for msg in messages:
            if msg.get("sources_json"):
                msg["sources"] = json.loads(msg["sources_json"])
            msg.pop("sources_json", None)
        return messages

    def delete_patient_chat_history(
        self, patient_id: str, persona: Optional[str] = None
    ) -> int:
        """Delete a patient's stored chat history (optionally only for one persona).

        Returns the number of rows deleted.
        """
        with self._conn() as conn:
            if persona is not None:
                cur = conn.execute(
                    "DELETE FROM chat_history WHERE patient_id = ? AND persona = ?",
                    (patient_id, persona),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM chat_history WHERE patient_id = ?", (patient_id,)
                )
            return cur.rowcount

    def get_patient_chat_history(
        self,
        patient_id: str,
        persona: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return a patient's chat history across all sessions, chronological.

        Filters by persona when provided (so a patient sees only patient-side
        conversations, a care manager only care-manager-side). Each returned dict:
        {role, content, sources, timestamp}.
        """
        query = (
            "SELECT role, content, sources_json, timestamp "
            "FROM chat_history WHERE patient_id = ?"
        )
        params: list = [patient_id]
        if persona is not None:
            query += " AND persona = ?"
            params.append(persona)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        messages = [dict(r) for r in rows]
        for msg in messages:
            if msg.get("sources_json"):
                msg["sources"] = json.loads(msg["sources_json"])
            msg.pop("sources_json", None)
        return messages


# ---------------------------------------------------------------------------
# CLI entry point — for quick setup and validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH
    nadac_csv = "data/cms_medicaid/nadac_2026.csv"
    eligibility_csv = "data/cms_medicaid/eligibility_levels.csv"

    db = MedDocDB(db_path)

    print(f"Creating tables in {db_path} ...")
    db.create_tables()

    print("Loading NADAC pricing ...")
    n = db.load_nadac(nadac_csv)
    print(f"  Loaded {n:,} NDC records")

    print("Loading eligibility ...")
    e = db.load_eligibility(eligibility_csv)
    print(f"  Loaded {e} state records")

    # Quick smoke tests
    print("\nSmoke tests:")
    metformin = db.search_drug_price_by_name("metformin", limit=3)
    for drug in metformin:
        print(f"  {drug['drug_name']:<40} ${drug['price_per_unit']:.5f}/{drug['pricing_unit']}")

    texas = db.get_state_eligibility("Texas")
    print(f"\n  Texas eligibility:")
    print(f"    Children 0-1:     {texas['medicaid_0_1']}")
    print(f"    Pregnant women:   {texas['pregnant_medicaid']}")
    print(f"    Adult expansion:  {texas['expansion_adults']}")

    print("\nDone.")
