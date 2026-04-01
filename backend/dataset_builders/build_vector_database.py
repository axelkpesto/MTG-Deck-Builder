"""Build the persisted vector database from the configured MTGJSON dataset."""

from pathlib import Path

from backend.card_data import CardDecoder, CardEncoder
from backend.config import CONFIG
from backend.vector_database import VectorDatabase


def main() -> None:
    """Encode commander-legal cards from AllPrintings and save the vector store."""
    root = Path(__file__).resolve().parents[2]
    source_path = root / CONFIG.datasets["FULL_DATASET_PATH"]
    output_path = root / CONFIG.datasets["VECTOR_DATABASE_PATH"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vector_db = VectorDatabase(CardEncoder(), CardDecoder())
    vector_db.parse_json(str(source_path))
    vector_db.save(str(output_path))

    print(f"Wrote {vector_db.size()} card vectors to {output_path}")


if __name__ == "__main__":
    main()
