from frontend.api_client import APIClient, APIClientError


def main() -> None:
    client = APIClient()

    try:
        result = client.health_check()
        print("API health check:", result)

    except APIClientError as exc:
        print("API connection failed:", exc)

    finally:
        client.close()


if __name__ == "__main__":
    main()
