# Backend Test Suite

Comprehensive `pytest` suite for the MTG Deck Builder backend. One test file per
backend module, with positive and negative cases for every public function.

## Layout

| Test file | Targets |
|---|---|
| `test_card.py` | `card_data.card.Card` — constructor, equality, hash, len, str, `get_attributes`, `to_json` |
| `test_card_fields.py` | `CardFields` vocabularies, set accessors, basic-land helpers, rarity maps, predicates |
| `test_card_fields_tagging.py` | `CardFields.tag_text/tag_subtypes/tag_card`, `parse_mtgjson_card`, `parse_moxfieldapi_card`, `parse_moxfield_group` |
| `test_card_encoder.py` | `CardEncoder.encode` layout, `rarity_to_int`, colorless fallback, subtype bits |
| `test_card_decoder.py` | `CardDecoder.decode/decode_to_dict`, `_title_case`, `int_to_rarity`, `slice`, `item_from_vector`, `constrain_logits`, land / color / mana masks |
| `test_deck.py` | `card_data.deck.Deck` — constructor, equality, len, str, serialization, `to_tensor`, `basic_lands_from_colors`, `shape_deck` |
| `test_simple_deck.py` | `card_data.deck.SimpleDeck` — constructor, equality, len, serialization, `from_json`, `load_json_file`, `shape_deck`, `to_tensor_stack` |
| `test_simple_deck_analyzer.py` | `SimpleDeckAnalyzer.analyze/analyze_tags/analyze_color_distribution/analyze_curve/analyze_lands_and_basics` |
| `test_vector_store.py` | `vector_database.VectorStore` — every public method including similarity, find, filter, save/load |
| `test_vector_database.py` | `VectorDatabase` facade — pass-throughs, `parse_json`, `vector_to_numpy`, `to_index`, `load_static` |
| `test_deckgen_utils.py` | `deckgen.utils` — `mana_value_bucket`, `safe_read_json`, `set_seed`, `clamp_int`, `duplicate_penalty`, `allowed_basic_land_types`, all `extract_*` |
| `test_firebase_auth.py` | `firestore.firebase_auth.generate_api_key`, `validate_api_key` |
| `test_tagging_model_helpers.py` | `ml.tagging_model.MLP`, `VectorsDataset`, `predicted_scores_from_probabilities`, `save_model`/`load_model` |
| `test_vector_db_server_helpers.py` | `api.vector_db_server` pure helpers — `clamp_int/clamp_float`, `format_id`, `parse_card_list_payload`, `parse_required_card_id`, `error`, `resolve_card_id`, `get_api_key_from_request` |
| `test_vector_db_server_routes.py` | Flask routes: `/`, `/help`, `/examples`, `/status`, `/get_vector`, `/get_vector_description`, `/get_vector_descriptions`, `/get_random_vector*`, `/get_similar_vectors`, `/get_tags`, `/get_tag_list`, `/get_tags_from_vector`, `/analyze_deck`, `/generate_deck`, 404 handler |
| `test_config.py` | `config.Config` JSON loader and required keys |

## Running

From the repo root:

```bash
pytest                                # run everything
pytest backend/tests/test_card.py     # single file
pytest -k vector                      # by keyword
pytest -m "not slow"                  # skip slow markers
```

`pytest.ini` lives at the repo root and points `testpaths` at `backend/tests`.

## Fixtures (conftest.py)

- `sample_card`, `legendary_creature_card`, `basic_land_card`, `colorless_card` — `Card` factories
- `encoded_vector_factory` — build encoded card vectors without running the encoder
- `fake_encoder` — lightweight stand-in for `CardEncoder` that skips the SentenceTransformer download
- `fake_decoder` — real `CardDecoder` (no heavy deps)
- `empty_vector_store`, `populated_vector_store` — `VectorStore` instances
- `empty_vector_database`, `populated_vector_database` — `VectorDatabase` instances
- `deterministic_torch_seed` — pin `torch.manual_seed(0)` for the test

## Markers

`slow`, `integration`, `requires_torch`, `requires_flask`, `requires_data` — declared in `pytest.ini` for selective runs in CI.

## Notes on test isolation

- `conftest.py` sets `API_KEY_PEPPER`, `REDIS_URL`, and `AUTHENTICATE=0` env vars before any backend module is imported, so modules that read these at import time work in test mode.
- `test_vector_db_server_routes.py` uses `unittest.mock.patch` to intercept module-level data/model loads in `backend.api.vector_db_server`. The Flask `test_client` then drives the live routes against in-memory fakes.
- `test_tagging_model_helpers.py` does **not** test the training loop or `build_dataset/prepare_dataset` (they require the on-disk vector database). Only the pure structures and prediction post-processing are covered.
