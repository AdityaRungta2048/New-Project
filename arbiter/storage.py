"""Persistence: an SQLite index plus a JSON file per arbitration.

Every arbitration is written twice — a flat row in SQLite for fast querying and
analytics, and the full nested `ArbitrationResult` as a JSON file for a complete,
human-readable audit trail (Phase 1.2 / 5.1: "Audit trail for every arbitration").
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

from .config import Settings, get_settings
from .models import ArbitrationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS arbitrations (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    original_prompt   TEXT,
    output_excerpt    TEXT NOT NULL,
    quality_score     INTEGER NOT NULL,
    confidence        REAL NOT NULL,
    num_issues_found  INTEGER NOT NULL,
    num_confirmed     INTEGER NOT NULL,
    num_dismissed     INTEGER NOT NULL,
    num_disagreements INTEGER NOT NULL,
    short_circuited   INTEGER NOT NULL,
    degraded          INTEGER NOT NULL,
    result_json       TEXT NOT NULL
);

-- Per-critic rows power the "critic behaviour" analytics (Phase 5.2).
CREATE TABLE IF NOT EXISTS critic_reports (
    arbitration_id  TEXT NOT NULL,
    dimension       TEXT NOT NULL,
    backend         TEXT NOT NULL,
    model           TEXT NOT NULL,
    ok              INTEGER NOT NULL,
    score           INTEGER,
    confidence      REAL,
    num_issues      INTEGER NOT NULL DEFAULT 0,
    num_confirmed   INTEGER NOT NULL DEFAULT 0,
    num_dismissed   INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (arbitration_id) REFERENCES arbitrations(id)
);

CREATE INDEX IF NOT EXISTS idx_reports_dim ON critic_reports(dimension);
CREATE INDEX IF NOT EXISTS idx_arb_created ON arbitrations(created_at);
"""


class Storage:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.db_path = Path(self.settings.db_path)
        self.json_dir = Path(self.settings.json_dir)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.json_dir.mkdir(parents=True, exist_ok=True)
            self._init_db()
        except OSError:
            # Read-only / unavailable filesystem (e.g. serverless like Vercel):
            # fall back to a writable temp dir so the service still functions.
            # The audit trail becomes ephemeral in that case.
            base = Path(tempfile.gettempdir()) / "arbiter"
            self.db_path = base / "arbitrations.sqlite"
            self.json_dir = base / "arbitrations"
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.json_dir.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- writes ------------------------------------------------------------
    def save(self, result: ArbitrationResult) -> None:
        # Full JSON audit file.
        json_path = self.json_dir / f"{result.id}.json"
        json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        verdict = result.verdict
        confirmed_by_dim: dict[str, int] = {}
        for ci in verdict.confirmed_issues:
            for d in ci.raised_by or [ci.dimension]:
                confirmed_by_dim[d.value] = confirmed_by_dim.get(d.value, 0) + 1
        dismissed_by_dim: dict[str, int] = {}
        for df in verdict.dismissed_flags:
            dismissed_by_dim[df.raised_by.value] = dismissed_by_dim.get(df.raised_by.value, 0) + 1

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO arbitrations
                   (id, created_at, original_prompt, output_excerpt, quality_score,
                    confidence, num_issues_found, num_confirmed, num_dismissed,
                    num_disagreements, short_circuited, degraded, result_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.id,
                    result.created_at.isoformat(),
                    result.original_prompt,
                    result.output_text[:280],
                    verdict.quality_score,
                    verdict.confidence,
                    result.num_issues_found,
                    len(verdict.confirmed_issues),
                    len(verdict.dismissed_flags),
                    len(result.disagreements),
                    int(result.short_circuited),
                    int(result.degraded),
                    result.model_dump_json(),
                ),
            )
            conn.execute(
                "DELETE FROM critic_reports WHERE arbitration_id = ?", (result.id,)
            )
            for report in result.reports:
                conn.execute(
                    """INSERT INTO critic_reports
                       (arbitration_id, dimension, backend, model, ok, score,
                        confidence, num_issues, num_confirmed, num_dismissed)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        result.id,
                        report.dimension.value,
                        report.backend,
                        report.model,
                        int(report.ok),
                        report.critique.score if report.ok and report.critique else None,
                        report.critique.confidence if report.ok and report.critique else None,
                        len(report.critique.issues) if report.ok and report.critique else 0,
                        confirmed_by_dim.get(report.dimension.value, 0),
                        dismissed_by_dim.get(report.dimension.value, 0),
                    ),
                )

    # -- reads -------------------------------------------------------------
    def get(self, arbitration_id: str) -> Optional[ArbitrationResult]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM arbitrations WHERE id = ?", (arbitration_id,)
            ).fetchone()
        if not row:
            return None
        return ArbitrationResult.model_validate_json(row["result_json"])

    def list(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, created_at, output_excerpt, quality_score, confidence,
                          num_issues_found, num_confirmed, num_dismissed,
                          num_disagreements, short_circuited, degraded
                   FROM arbitrations ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) AS c FROM arbitrations").fetchone()["c"]
