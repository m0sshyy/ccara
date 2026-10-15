"""CCaRA — local SQLite assessment history.

Stores each completed risk assessment so the prototype can demonstrate the
"Number of completed risk assessments" / "Number of high-risk users
referred for screening" key metrics from the Lean Model Canvas, and so a
user (or clinic demo) can review past results. Runs entirely on the local
machine — no data leaves the device, matching the report's local-only
deployment note.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "assessments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    label TEXT,
    age INTEGER,
    gender TEXT,
    family_history TEXT,
    smoking_history TEXT,
    alcohol_consumption TEXT,
    obesity_bmi TEXT,
    diet_risk TEXT,
    physical_activity TEXT,
    diabetes TEXT,
    ibd TEXT,
    genetic_mutation TEXT,
    predicted_class TEXT,
    confidence REAL,
    prob_low REAL,
    prob_moderate REAL,
    prob_high REAL,
    urgency_tier TEXT,
    urgency_score REAL,
    top_factors TEXT,
    recommendation TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # TRUNCATE journal mode is used instead of the default (also TRUNCATE)
    # / WAL because WAL relies on shared-memory locking that some network
    # or sandboxed/mounted filesystems don't support cleanly. TRUNCATE is
    # broadly compatible and still crash-safe for a local single-user app.
    conn.execute("PRAGMA journal_mode=TRUNCATE")
    conn.execute(SCHEMA)
    return conn


def save_assessment(*, label: str, form_values: dict, prediction, top_factors: list[dict]) -> int:
    conn = get_connection()
    with conn:
        cur = conn.execute(
            """
            INSERT INTO assessments (
                timestamp, label, age, gender, family_history, smoking_history,
                alcohol_consumption, obesity_bmi, diet_risk, physical_activity,
                diabetes, ibd, genetic_mutation, predicted_class, confidence,
                prob_low, prob_moderate, prob_high, urgency_tier, urgency_score,
                top_factors, recommendation
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                label or None,
                form_values["Age"],
                form_values["Gender"],
                form_values["Family_History"],
                form_values["Smoking_History"],
                form_values["Alcohol_Consumption"],
                form_values["Obesity_BMI"],
                form_values["Diet_Risk"],
                form_values["Physical_Activity"],
                form_values["Diabetes"],
                form_values["Inflammatory_Bowel_Disease"],
                form_values["Genetic_Mutation"],
                prediction.predicted_class,
                prediction.confidence,
                prediction.probabilities.get("Low", 0.0),
                prediction.probabilities.get("Moderate", 0.0),
                prediction.probabilities.get("High", 0.0),
                prediction.urgency_tier,
                prediction.urgency_score,
                json.dumps(top_factors),
                None,
            ),
        )
        return cur.lastrowid


def fetch_history(limit: int | None = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM assessments ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def summary_stats() -> dict:
    df = fetch_history()
    if df.empty:
        return {"total": 0, "by_class": {}, "avg_confidence": 0.0, "high_urgent": 0}
    by_class = df["predicted_class"].value_counts().to_dict()
    high_urgent = int((df["urgency_tier"] == "High-Critical").sum())
    return {
        "total": int(len(df)),
        "by_class": by_class,
        "avg_confidence": float(df["confidence"].mean()),
        "high_urgent": high_urgent,
    }


def clear_history():
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM assessments")
