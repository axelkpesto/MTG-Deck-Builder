"""Build a lightweight card-name -> image URL dataset for frontend previews."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CONFIG


def image_url_from_scryfall_id(card_id: str) -> str:
    """Build a stable Scryfall image URL from a Scryfall card id."""
    return (
        f"https://api.scryfall.com/cards/{card_id}"
        "?format=image&version=normal"
    )


def add_card_image(image_map: dict[str, list[str]], seen_by_name: dict[str, set[str]], name: str, card_id: str) -> None:
    """Add one card image URL if the name and Scryfall id are valid."""
    card_name = str(name).strip()
    scryfall_id = str(card_id).strip()
    if not card_name or not scryfall_id:
        return

    image_url = image_url_from_scryfall_id(scryfall_id)
    if image_url in seen_by_name[card_name]:
        return

    seen_by_name[card_name].add(image_url)
    image_map[card_name].append(image_url)


def main() -> None:
    all_printings_path = ROOT / CONFIG.datasets["FULL_DATASET_PATH"]
    output_path = ROOT / CONFIG.datasets["CARD_IMAGE_DATASET_PATH"]

    with open(all_printings_path, "r", encoding="utf-8") as f:
        all_printings = json.load(f)

    image_map: dict[str, list[str]] = defaultdict(list)
    seen_by_name: dict[str, set[str]] = defaultdict(set)

    sets = all_printings.get("data", {})
    for set_data in sets.values():
        if not isinstance(set_data, dict):
            continue
        cards = set_data.get("cards", [])
        if not isinstance(cards, list):
            continue

        for card in cards:
            if not isinstance(card, dict):
                continue

            identifiers = card.get("identifiers", {})
            if not isinstance(identifiers, dict):
                identifiers = {}

            add_card_image(
                image_map=image_map,
                seen_by_name=seen_by_name,
                name=str(card.get("name", "")).strip(),
                card_id=str(identifiers.get("scryfallId", "")).strip(),
            )

            face_data = card.get("faceData", [])
            if not isinstance(face_data, list):
                continue
            for face in face_data:
                if not isinstance(face, dict):
                    continue
                face_identifiers = face.get("identifiers", {})
                if not isinstance(face_identifiers, dict):
                    face_identifiers = {}
                add_card_image(
                    image_map=image_map,
                    seen_by_name=seen_by_name,
                    name=str(face.get("name", "")).strip(),
                    card_id=str(face_identifiers.get("scryfallId", "")).strip(),
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(image_map.items())), f, ensure_ascii=True, indent=2)

    print(f"Wrote {len(image_map)} card image entries to {output_path}")


if __name__ == "__main__":
    main()
