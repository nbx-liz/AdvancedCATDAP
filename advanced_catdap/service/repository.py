import sqlite3
import json
from typing import Dict, Any, List, Optional, Protocol
from contextlib import contextmanager
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class JobRepository(Protocol):
    """Abstract interface for job persistence."""
    
    def init_storage(self) -> None:
        ...
        
    def save_job(self, job_id: str, dataset_id: str, status: str, params: str) -> None:
        ...
        
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        ...
        
    def update_status(self, job_id: str, status: str, result: Optional[str] = None, 
                      error: Optional[str] = None, progress: Optional[str] = None) -> None:
        ...

class SQLiteJobRepository:
    """SQLite implementation of JobRepository."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()

    def init_storage(self) -> None:
        with self._get_connection() as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        dataset_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        params TEXT,
                        result TEXT,
                        error TEXT,
                        progress TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

    def save_job(self, job_id: str, dataset_id: str, status: str, params: str) -> None:
        try:
            with self._get_connection() as conn:
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO jobs (job_id, dataset_id, status, params, updated_at) 
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (job_id, dataset_id, status, params))
        except sqlite3.Error as e:
            logger.error(f"Database error saving job {job_id}: {e}")
            raise RuntimeError("Database error") from e

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving job {job_id}: {e}")
            return None

    def update_status(self, job_id: str, status: str, result: Optional[str] = None, 
                      error: Optional[str] = None, progress: Optional[str] = None) -> None:
        query_parts = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params = [status]
        
        if result is not None:
            query_parts.append("result = ?")
            params.append(result)
        
        if error is not None:
            query_parts.append("error = ?")
            params.append(error)
            
        if progress is not None:
            query_parts.append("progress = ?")
            params.append(progress)
            
        params.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(query_parts)} WHERE job_id = ?"
        
        try:
            with self._get_connection() as conn:
                with conn:
                    conn.execute(sql, params)
        except sqlite3.Error as e:
            logger.error(f"Database error updating status for {job_id}: {e}")
