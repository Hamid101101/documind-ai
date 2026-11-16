
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

# Load project environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


class PostgreSQLClient:
    """Infrastructure client for PostgreSQL database operations."""

    def __init__(self):
        self.host = self._get_env("POSTGRES_HOST", "localhost")
        self.port = int(self._get_env("POSTGRES_PORT", "5432"))
        self.database = self._get_env("POSTGRES_DB")
        self.user = self._get_env("POSTGRES_USER")
        self.password = self._get_env("POSTGRES_PASSWORD")

        self.connect_timeout = 5

    @staticmethod
    def _get_env(name: str, default: str | None = None) -> str:
        import os

        value = os.getenv(name, default)

        if not value:
            raise ValueError(f"Missing required environment variable: {name}")

        return value

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection, None, None]:
        """
        Open a database connection.

        Commits on successful completion and rolls back on exceptions.
        Always closes the connection.
        """
        with psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
            connect_timeout=self.connect_timeout,
            row_factory=dict_row,
        ) as conn:
            yield conn

    def check_connection(self) -> bool:
        try:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 AS connection_ok")
                    return cursor.fetchone()["connection_ok"] == 1

        except psycopg.Error:
            logger.exception("PostgreSQL connection check failed")
            return False

    def execute(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> int:
        """
        Execute an INSERT, UPDATE or DELETE statement.

        Returns the number of affected rows.
        """
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.rowcount

    def fetch_one(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> dict[str, Any] | None:
        """Execute a query and return one row as a dictionary."""
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()

    def fetch_all(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a query and return all rows as dictionaries."""
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
