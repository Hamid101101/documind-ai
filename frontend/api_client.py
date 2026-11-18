
import json
import os
from typing import Any
from uuid import UUID

import requests
from dotenv import load_dotenv

load_dotenv()


class APIClientError(Exception):
    """Raised when a backend API request fails."""


class APIClient:
    """HTTP client for communicating with the DocuMind FastAPI backend."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        """Send an HTTP request and handle API errors."""

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs,
            )

            response.raise_for_status()

            if response.status_code == 204:
                return None

            return response.json()

        except requests.HTTPError as exc:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text

            raise APIClientError(
                f"API error ({response.status_code}): {detail}"
            ) from exc

        except requests.RequestException as exc:
            raise APIClientError(
                f"Could not connect to DocuMind API: {exc}"
            ) from exc

    def health_check(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health/ready")

    def upload_document(
        self,
        filename: str,
        file_obj: Any,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/documents/",
            files={
                "file": (filename, file_obj),
            },
            data={
                "metadata": json.dumps(metadata or {}),
            },
        )

    def create_conversation(
        self,
        title: str = "New conversation",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/conversations/",
            json={
                "title": title,
                "metadata": metadata or {},
            },
        )

    def list_conversations(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/v1/conversations/",
            params={
                "page": page,
                "page_size": page_size,
            },
        )

    def get_conversation(
        self,
        conversation_id: UUID | str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/v1/conversations/{conversation_id}",
        )

    def delete_conversation(
        self,
        conversation_id: UUID | str,
    ) -> None:
        self._request(
            "DELETE",
            f"/api/v1/conversations/{conversation_id}",
        )

    def send_message(
        self,
        question: str,
        conversation_id: UUID | str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": question,
        }

        if conversation_id is not None:
            payload["conversation_id"] = str(conversation_id)

        if filters is not None:
            payload["filters"] = filters

        if top_k is not None:
            payload["top_k"] = top_k

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return self._request(
            "POST",
            "/api/v1/chat/",
            json=payload,
        )

    def close(self) -> None:
        self.session.close()
