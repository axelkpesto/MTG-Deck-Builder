"""CLI entrypoint for scheduled API pepper and key rotation."""
import json

from backend.firestore import rotate_pepper_and_keys
def json_default(value):
    """Serialize datetimes and other simple values for CLI output."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    result = rotate_pepper_and_keys()
    print(json.dumps(result, default=json_default, indent=2))


if __name__ == "__main__":
    main()