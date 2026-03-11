"""Fetch commander deck data from Moxfield and serialize simple deck records."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests import RequestException

from card_data import SimpleDeck
from config import CONFIG

MOXFIELD_BASE = "https://api2.moxfield.com/v2"
DEFAULT_OUT_PATH = CONFIG.datasets["DECKS_DATASET_PATH"]


def _headers() -> Dict[str, str]:
    return {"User-Agent": str(os.environ.get("MOXFIELD_API_KEY"))}


@dataclass(frozen=True)
class FetchConfig:
    """Configuration for paging, filtering, and request behavior."""

    format: str = "commander"
    page_size: int = 100
    timeout_s: int = 30
    sleep_s: float = 0.35
    min_total_cards: int = 100

def _get_json(session: requests.Session, url: str, *, params: Optional[Dict[str, Any]] = None, timeout_s: int = 30) -> Dict[str, Any]:
    """GET a JSON payload and raise on HTTP errors."""
    r = session.get(url, headers=_headers(), params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def fetch_deck_summaries(session: requests.Session, *, page: int, cfg: FetchConfig) -> List[Dict[str, Any]]:
    """Fetch paged deck summaries from Moxfield."""
    url = f"{MOXFIELD_BASE}/decks/all/"
    params = {"format": cfg.format, "page": int(page), "pageSize": int(cfg.page_size)}
    payload = _get_json(session, url, params=params, timeout_s=cfg.timeout_s)
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def fetch_deck_detail(session: requests.Session, *, public_id: str, cfg: FetchConfig) -> Optional[Dict[str, Any]]:
    """Fetch a single deck detail payload by public deck id."""
    url = f"{MOXFIELD_BASE}/decks/all/{public_id}"
    try:
        return _get_json(session, url, timeout_s=cfg.timeout_s)
    except (RequestException, ValueError):
        return None

def parse_simple_deck(deck_detail: Dict[str, Any], *, cfg: FetchConfig) -> Optional[SimpleDeck]:
    """Convert raw deck detail JSON into a `SimpleDeck` instance."""
    if not isinstance(deck_detail, dict):
        return None

    deck_id = str(deck_detail.get("publicId") or "")
    if not deck_id:
        return None

    mainboard = deck_detail.get("mainboard")
    commanders_obj = deck_detail.get("commanders")

    if not isinstance(mainboard, dict) or not isinstance(commanders_obj, dict):
        return None

    commanders = list(commanders_obj.keys())
    if not commanders:
        return None

    cards: List[str] = []
    for card_name, entry in mainboard.items():
        if not card_name:
            continue
        qty = 1
        if isinstance(entry, dict):
            try:
                qty = int(entry.get("quantity", 1))
            except (TypeError, ValueError):
                qty = 1
        if qty > 0:
            cards.extend([str(card_name)] * qty)

    if (len(commanders) + len(cards)) < cfg.min_total_cards:
        return None

    return SimpleDeck(deck_id=deck_id, commanders=commanders, cards=cards)

def save_simple_decks(path: str, decks: Dict[str, SimpleDeck]) -> None:
    """Persist simple decks to disk as JSON keyed by deck id."""
    payload = {deck_id: d.get_attributes() for deck_id, d in decks.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def load_simple_decks(path: str) -> Dict[str, SimpleDeck]:
    """Load simple decks from disk, accepting dict or list JSON layouts."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: Iterable[Dict[str, Any]]
    if isinstance(raw, dict):
        items = raw.values()
    elif isinstance(raw, list):
        items = raw
    else:
        return {}

    out: Dict[str, SimpleDeck] = {}
    for obj in items:
        if not isinstance(obj, dict):
            continue
        try:
            d = SimpleDeck.from_json(obj)
            out[str(d.id)] = d
        except (TypeError, KeyError, ValueError):
            continue
    return out

def build_simple_decks(start_page: int = 1, end_page: int = 50, out_path: str = DEFAULT_OUT_PATH, seed_existing: bool = True, cfg: Optional[FetchConfig] = None) -> Dict[str, SimpleDeck]:
    """Fetch pages of decks, parse them, and continuously save progress."""
    cfg = cfg or FetchConfig()

    decks: Dict[str, SimpleDeck] = load_simple_decks(out_path) if seed_existing else {}
    print(f"Loaded {len(decks)} existing decks from {out_path}" if seed_existing else "Starting fresh.")

    with requests.Session() as session:
        for page in range(int(start_page), int(end_page) + 1):
            summaries = fetch_deck_summaries(session, page=page, cfg=cfg)
            if not summaries:
                print(f"[page {page}] no results")
                continue

            print(f"[page {page}] {len(summaries)} summaries")
            for row in summaries:
                public_id = row.get("publicId")
                if not public_id:
                    continue

                if str(public_id) in decks:
                    continue

                detail = fetch_deck_detail(session, public_id=str(public_id), cfg=cfg)
                if detail is None:
                    continue

                sd = parse_simple_deck(detail, cfg=cfg)
                if sd is None:
                    continue

                decks[str(sd.id)] = sd

                save_simple_decks(out_path, decks)

                time.sleep(cfg.sleep_s)
            save_simple_decks(out_path, decks)
            print(f"[page {page}] total decks saved: {len(decks)}")

    return decks


if __name__ == "__main__":
    build_simple_decks(start_page=1, end_page=50, out_path=DEFAULT_OUT_PATH, seed_existing=True)
