import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_expected_api_key() -> str | None:
    """
    Fetch the expected API key from environment variables.
    """
    return os.getenv("API_KEY")


def validate_api_key(provided_key: str | None) -> bool:
    """
    Simple API key validation.
    """
    expected = get_expected_api_key()
    if not expected:
        # If no key is configured, treat as locked down (no access).
        return False
    if not provided_key:
        return False
    return provided_key == expected

