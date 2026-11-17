
from fastapi.testclient import TestClient

from backend.app.main import app


class MockChatService:
    """Mock ChatService for API endpoint testing."""

    async def chat(
        self,
        question,
        filters=None,
        top_k=None,
        max_tokens=None,
    ):
        return {
            "answer": "PostgreSQL is a relational database.",
            "sources": [
                {
                    "filename": "database.pdf",
                    "page": 2,
                }
            ],
            "model": "qwen2.5:3b",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }

    async def close(self):
        pass


def main():
    with TestClient(app) as client:
        # Replace the real service with our mock.
        app.state.chat_service = MockChatService()

        # 1. Valid request with optional parameters.
        response = client.post(
            "/api/v1/chat/",
            json={
                "question": "What is PostgreSQL?",
                "top_k": 5,
                "max_tokens": 2000,
            },
        )

        print("Valid request:", response.status_code)
        print("Response:", response.json())

        assert response.status_code == 200
        assert response.json()["answer"] == (
            "PostgreSQL is a relational database."
        )
        assert response.json()["usage"]["total_tokens"] == 120

        # 2. Invalid request: empty question.
        response = client.post(
            "/api/v1/chat/",
            json={"question": ""},
        )

        print("Empty question:", response.status_code)
        assert response.status_code == 422

        # 3. Invalid request: negative top_k.
        response = client.post(
            "/api/v1/chat/",
            json={
                "question": "What is PostgreSQL?",
                "top_k": -1,
            },
        )

        print("Negative top_k:", response.status_code)
        assert response.status_code == 422

        print("\nAll manual API checks passed!")


if __name__ == "__main__":
    main()
