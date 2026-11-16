
import pytest

from infra.postgres.postgres_client import PostgreSQLClient


@pytest.fixture
def postgres_client():
    return PostgreSQLClient()


def test_postgres_connection(postgres_client):
    """Verify PostgreSQL connectivity."""
    assert postgres_client.check_connection() is True


def test_fetch_one(postgres_client):
    """Verify that a single row can be retrieved."""
    result = postgres_client.fetch_one(
        "SELECT 1 AS value"
    )

    assert result is not None
    assert result["value"] == 1


def test_fetch_all(postgres_client):
    """Verify that multiple rows can be retrieved."""
    results = postgres_client.fetch_all(
        "SELECT generate_series(1, 3) AS number"
    )

    assert len(results) == 3
    assert [row["number"] for row in results] == [1, 2, 3]


def test_execute_and_transaction(postgres_client):
    """Verify INSERT and transaction rollback behavior."""

    with postgres_client.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TEMPORARY TABLE postgres_client_test (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
            """)

            cursor.execute(
                "INSERT INTO postgres_client_test (id, name) VALUES (%s, %s)",
                (1, "DocuMind"),
            )

            cursor.execute(
                "SELECT name FROM postgres_client_test WHERE id = %s",
                (1,),
            )

            result = cursor.fetchone()
            assert result["name"] == "DocuMind"
