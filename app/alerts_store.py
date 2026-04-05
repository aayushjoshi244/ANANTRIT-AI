import sqlite3
import threading
import time
from typing import Optional


class AlertsStore:
    def __init__(self, db_path: str = "alerts.db") -> None:
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self.lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    person_name TEXT,
                    image_path TEXT,
                    message TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
            conn.close()

    def add_alert(
        self,
        event_type: str,
        person_name: Optional[str] = None,
        image_path: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        with self.lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO alerts (event_type, person_name, image_path, message, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_type, person_name, image_path, message, time.time()),
            )
            conn.commit()
            conn.close()

    def list_alerts(self, limit: int = 50) -> list[dict]:
        with self.lock:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, event_type, person_name, image_path, message, created_at
                FROM alerts
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            conn.close()

        return [
            {
                "id": row[0],
                "event_type": row[1],
                "person_name": row[2],
                "image_path": row[3],
                "message": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]