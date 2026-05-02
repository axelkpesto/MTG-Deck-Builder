"""FastMCP server wrapping the MTG Deck Builder vector database API."""
import os
from typing import Annotated

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

BASE_URL = os.environ.get("FLASK_API_PATH", f"http://localhost:{os.environ.get('DEFAULT_PORT', 5000)}")
API_KEY = os.environ.get("MTG_GLOBAL_API_KEY", "")

mcp = FastMCP("MTG Deck Builder")


def _headers() -> dict[str, str]:
    if API_KEY:
        return {"Authorization": f"Bearer {API_KEY}"}
    return {}


async def _post(path: str, body: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{BASE_URL}{path}", json=body or {}, headers=_headers())
        r.raise_for_status()
        return r.json()

@mcp.tool()
async def get_vector(card_name: Annotated[str, "Exact MTG card name"]) -> dict:
    """Return the raw embedding vector for a single MTG card.

    Returns a dict with 'id' (resolved card name) and 'vector' (list of floats).
    """
    return await _post("/get_vector", {"id": card_name})


@mcp.tool()
async def get_card_description(card_name: Annotated[str, "Exact MTG card name"]) -> dict:
    """Return metadata / description for a single MTG card from the vector database.

    Includes mana cost, type, oracle text, color identity, and other card attributes.
    """
    return await _post("/get_vector_description", {"id": card_name})


@mcp.tool()
async def get_card_descriptions(cards: Annotated[list[str], "List of MTG card names to look up"]) -> dict:
    """Return metadata for a batch of MTG cards in one call.

    Returns a dict with 'found' (name → metadata) and 'missing' (name → error).
    """
    return await _post("/get_vector_descriptions", {"cards": cards})


@mcp.tool()
async def get_random_card() -> dict:
    """Return the name and embedding vector of a randomly sampled MTG card."""
    return await _post("/get_random_vector")


@mcp.tool()
async def get_random_card_description() -> dict:
    """Return the metadata / description of a randomly sampled MTG card."""
    return await _post("/get_random_vector_description")

@mcp.tool()
async def get_similar_cards(card_name: Annotated[str, "Reference MTG card name"], num_cards: Annotated[int, "Number of similar cards to return (1-1000, default 5)"] = 5) -> dict:
    """Find MTG cards most similar to the given card using vector similarity.

    Results are ranked by semantic closeness in embedding space. Returns a dict
    keyed by rank index (0 = most similar), each value being full card metadata.
    """
    return await _post("/get_similar_vectors", {"id": card_name, "num_vectors": num_cards})

@mcp.tool()
async def get_card_tags(card_name: Annotated[str, "MTG card name to predict tags for"], threshold: Annotated[float, "Minimum probability to include a tag (0.0-1.0, default 0.5)"] = 0.5, top_k: Annotated[int, "Max number of top-scoring tags to return (default 8)"] = 8) -> dict:
    """Predict gameplay role tags for a single MTG card.

    Returns 'predicted' (list of tag strings above threshold), 'predicted_scores'
    (tag+score dicts above threshold), 'scores' (top-k tag+score dicts), and
    'threshold' (the value used).

    Example tags: ramp, removal, draw, counter, token, combo, tutor, wipe.
    """
    return await _post("/get_tags", {"id": card_name, "threshold": threshold, "top_k": top_k})


@mcp.tool()
async def get_tags_for_cards(cards: Annotated[list[str], "List of MTG card names to tag"], threshold: Annotated[float, "Minimum probability to include a tag (0.0-1.0, default 0.5)"] = 0.5, top_k: Annotated[int, "Max number of top-scoring tags per card (default 8)"] = 8) -> dict:
    """Predict gameplay role tags for a batch of MTG cards.

    Returns 'found' (name → tag result) and 'missing' (name → error) dicts.
    """
    return await _post("/get_tag_list", {"cards": cards, "threshold": threshold, "top_k": top_k})


@mcp.tool()
async def get_tags_from_vector(vector: Annotated[list[float], "Raw card embedding vector (list of floats)"], threshold: Annotated[float, "Minimum probability to include a tag (0.0-1.0, default 0.5)"] = 0.5, top_k: Annotated[int, "Max number of top-scoring tags to return (default 8)"] = 8) -> dict:
    """Predict gameplay role tags directly from a raw embedding vector.

    Useful when you already have a vector from get_vector and want to avoid a
    second card name lookup. Returns same shape as get_card_tags.
    """
    return await _post("/get_tags_from_vector", {"vector": vector, "threshold": threshold, "top_k": top_k})

@mcp.tool()
async def generate_deck(commander_name: Annotated[str, "Name of the Commander card to build around"]) -> dict:
    """Generate a 100-card Commander deck for the given commander using the GNN model.

    Returns a dict mapping card names to quantities, plus generation statistics.
    The deck generation model must be fully loaded (check get_status first).
    """
    return await _post("/generate_deck", {"id": commander_name})


@mcp.tool()
async def analyze_deck(
    commander_name: Annotated[str, "Name of the Commander card"], cards: Annotated[list[str], "List of non-commander card names in the deck"]) -> dict:
    """Analyze a Commander deck and return comprehensive metrics.

    Metrics include mana curve, color identity distribution, tag distribution
    (ramp, removal, draw counts, etc.), and other SimpleDeckAnalyzer statistics.
    """
    return await _post("/analyze_deck", {"commander": commander_name, "cards": cards})

if __name__ == "__main__":
    mcp.run()
