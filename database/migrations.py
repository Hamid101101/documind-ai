
import hashlib
import logging
import os
import re
import sys
import time

from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "database" / "migrations"

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "documind"),
    "user": os.getenv("POSTGRES_USER", "documind_user"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "connect_timeout": 10,
    "row_factory": dict_row,
}

MIGRATION_PATTERN = re.compile(r"^\d{3}_[a-z0-9_-]+\.sql$")


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("documind.migrations")


# --------------------------------------------------
# Migration tracking table
# --------------------------------------------------

CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS migrations_logs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    migration_name VARCHAR(255) NOT NULL UNIQUE,
    checksum VARCHAR(64) NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    execution_time_ms BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'success',
    CONSTRAINT valid_migration_status
        CHECK (status = 'success')
);
"""


# --------------------------------------------------
# Database connection
# --------------------------------------------------

def get_connection():
    if not DATABASE_CONFIG["password"]:
        raise RuntimeError("POSTGRES_PASSWORD is not configured.")

    return psycopg.connect(**DATABASE_CONFIG)


# --------------------------------------------------
# Initialize migration tracking
# --------------------------------------------------

def initialize_migrations_table(connection):
    logger.info("Checking migration tracking table...")

    with connection.transaction():
        connection.execute(CREATE_MIGRATIONS_TABLE)

    logger.info("Migration tracking table is ready.")


# --------------------------------------------------
# Discover SQL migrations
# --------------------------------------------------

def discover_migrations():
    if not MIGRATIONS_DIR.exists():
        raise FileNotFoundError(
            f"Migrations directory not found: {MIGRATIONS_DIR}"
        )

    migrations = sorted(
        path
        for path in MIGRATIONS_DIR.iterdir()
        if path.is_file() and MIGRATION_PATTERN.fullmatch(path.name)
    )

    if not migrations:
        logger.warning("No SQL migration files found.")

    return migrations


# --------------------------------------------------
# Calculate migration checksum
# --------------------------------------------------

def calculate_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# --------------------------------------------------
# Retrieve executed migrations
# --------------------------------------------------

def get_executed_migrations(connection):
    rows = connection.execute(
        """
        SELECT migration_name, checksum
        FROM migrations_logs
        WHERE status = 'success'
        """
    ).fetchall()

    return {
        row["migration_name"]: row["checksum"]
        for row in rows
    }


# --------------------------------------------------
# Execute a migration
# --------------------------------------------------

def execute_migration(connection, migration_path, executed):
    migration_name = migration_path.name
    content = migration_path.read_bytes()
    checksum = calculate_checksum(content)

    if migration_name in executed:
        if executed[migration_name] != checksum:
            raise RuntimeError(
                f"Checksum mismatch: {migration_name} "
                "has been modified after execution."
            )

        logger.info("Already executed, skipping: %s", migration_name)
        return

    sql = content.decode("utf-8")

    logger.info("Executing migration: %s", migration_name)

    start_time = time.perf_counter()

    # SQL changes and migration log are committed together.
    with connection.transaction():
        # Serialize migration execution across concurrent runners.
        connection.execute(
            "SELECT pg_advisory_xact_lock(742019, 1)"
        )

        # Recheck after acquiring the lock.
        existing = connection.execute(
            """
            SELECT checksum
            FROM migrations_logs
            WHERE migration_name = %s
            """,
            (migration_name,),
        ).fetchone()

        if existing:
            if existing["checksum"] != checksum:
                raise RuntimeError(
                    f"Checksum mismatch: {migration_name}"
                )

            logger.info("Already executed, skipping: %s", migration_name)
            return

        connection.execute(sql)

        execution_time_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        connection.execute(
            """
            INSERT INTO migrations_logs (
                migration_name,
                checksum,
                execution_time_ms,
                status
            )
            VALUES (%s, %s, %s, 'success')
            """,
            (migration_name, checksum, execution_time_ms),
        )

    executed[migration_name] = checksum

    logger.info(
        "Successfully executed %s (%d ms)",
        migration_name,
        execution_time_ms,
    )


# --------------------------------------------------
# Main migration runner
# --------------------------------------------------

def run_migrations():
    logger.info("Starting DocuMind AI database migrations...")

    with get_connection() as connection:
        # Always initialize the tracking table first.
        initialize_migrations_table(connection)

        migrations = discover_migrations()

        executed = get_executed_migrations(connection)

        for migration in migrations:
            execute_migration(connection, migration, executed)

    logger.info("Database migrations completed successfully.")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    try:
        run_migrations()
    except Exception:
        logger.exception("Database migration process failed.")
        sys.exit(1)
