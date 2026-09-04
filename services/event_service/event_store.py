"""Where alerts go once fired: disk, SQLite, dashboard.

Each sink is best-effort and isolated -- one failing must not stop the others,
so the engine catches per-sink exceptions. The dashboard POST happens on a
daemon thread with a short timeout and is dropped on failure rather than
blocking the detection stream; a durable outbox is deferred by design.

On-disk layout and the SQLite schema notes are in README.md.
"""
import json
import logging
import os
import sqlite3
import threading
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from alert_rules import Alert

log = logging.getLogger("events.store")


class AlertSink(ABC):
    @abstractmethod
    def emit(self, alert: Alert, frame=None) -> None:
        """Persist/forward one alert. `frame` is an annotated BGR copy or None."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class JsonDirSink(AlertSink):
    """One directory per alert: event.json + annotated frame jpeg.

    Suppressed evidence nests under its parent alert. Layout in README.md.
    """

    def __init__(self, root: Path):
        self.dir = root / "alerts"
        self.dir.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: Alert, frame=None) -> None:
        if alert.suppressed:
            d = self.dir / alert.parent_id / "suppressed"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{alert.id}.json").write_text(json.dumps(alert.to_dict(), indent=2))
            frame_path = d / f"{alert.id}_frame_{alert.frame_id}.jpg"
        else:
            d = self.dir / alert.id
            d.mkdir(parents=True, exist_ok=True)
            (d / "event.json").write_text(json.dumps(alert.to_dict(), indent=2))
            frame_path = d / f"frame_{alert.frame_id}.jpg"
        if frame is not None:
            import cv2 as cv          # local import: sink usable without opencv
            cv.imwrite(str(frame_path), frame)


class SqliteSink(AlertSink):
    """One row per alert. WAL mode so dashboard readers never block the writer."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS alerts (
            id          TEXT PRIMARY KEY,
            wall_ts     REAL,
            stream_id   TEXT,
            track_id    INTEGER,
            rule        TEXT,
            score       REAL,
            frame_id    INTEGER,
            event_json  TEXT,
            suppressed  INTEGER DEFAULT 0,
            parent_id   TEXT DEFAULT '',
            frame_age_s REAL DEFAULT -1
        )"""

    def __init__(self, root: Path):
        self._db = sqlite3.connect(str(root / "alerts.db"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(self._SCHEMA)
        # Migrate an older db in place; harmless no-op on a fresh one. Rows
        # written before a column existed keep its DEFAULT (-1 = age unknown).
        for col, decl in (("suppressed", "INTEGER DEFAULT 0"),
                          ("parent_id", "TEXT DEFAULT ''"),
                          ("frame_age_s", "REAL DEFAULT -1")):
            try:
                self._db.execute(f"ALTER TABLE alerts ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass                      # column already exists
        self._db.commit()
        self._lock = threading.Lock()

    def emit(self, alert: Alert, frame=None) -> None:
        with self._lock:
            # Columns named explicitly, NOT positional VALUES(...): a migrated db
            # can have the columns in a different order than _SCHEMA, and adding
            # one must not silently shift every field by a slot.
            self._db.execute(
                "INSERT OR IGNORE INTO alerts "
                "(id, wall_ts, stream_id, track_id, rule, score, frame_id, "
                " event_json, suppressed, parent_id, frame_age_s) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (alert.id, alert.wall_ts, alert.stream_id, alert.track_id,
                 alert.rule, alert.score, alert.frame_id,
                 json.dumps(alert.to_dict()),
                 int(alert.suppressed), alert.parent_id, alert.frame_age_s))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


class DashboardSink(AlertSink):
    """Fire-and-forget POST of the alert JSON to the dashboard."""

    def __init__(self, url: str, timeout_s: float = 3.0):
        self.url = url
        self.timeout_s = timeout_s

    def emit(self, alert: Alert, frame=None) -> None:
        if alert.suppressed:
            return    # cooldown exists to stop dashboard spam; evidence is on disk
        payload = json.dumps(alert.to_dict()).encode()
        threading.Thread(target=self._post, args=(payload, alert.id),
                         daemon=True).start()

    def _post(self, payload: bytes, alert_id: str) -> None:
        try:
            req = urllib.request.Request(
                self.url, data=payload,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=self.timeout_s).close()
        except Exception as e:
            log.warning("dashboard push failed for %s: %s", alert_id, e)


def build_sinks(root: Path) -> list[AlertSink]:
    sinks: list[AlertSink] = [JsonDirSink(root), SqliteSink(root)]
    url = os.getenv("DASHBOARD_URL")
    if url:
        sinks.append(DashboardSink(url))
        log.info("dashboard sink -> %s", url)
    return sinks
