"""Compatibility wrapper for the deck-generation worker module."""

from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from workers.deckgen_worker import *  # noqa: F401,F403
from workers.deckgen_worker import main


if __name__ == "__main__":
    main()
