"""Build and query a filtered MTG card dataset from mtgjson exports."""

import os
import re
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from card_data import CardFields

load_dotenv()


class DataSet:
    """Load, parse, and query commander-legal card data."""

    def __init__(self) -> None:
        """Initialize raw set data and parsed card dataframe."""
        self.set_dataframe: pd.Series = self.json_data_setup(os.environ.get("FULL_DATASET_PATH"))
        self.card_set: pd.DataFrame = self.parse_set_data()

    def json_data_setup(self, filename: str) -> pd.Series:
        """Load the source mtgjson file and return the relevant set slice."""
        assert os.path.isfile(filename), f"{filename} not found."
        return pd.read_json(filename)['data'][2:]

    def parse_set_data(self) -> pd.DataFrame:
        """Parse commander-legal paper cards into a normalized dataframe."""
        card_data: Dict[str, List[Any]] = {}
        for game_set in self.set_dataframe:
            for card in game_set['cards']:
                if 'commander' in card['legalities'] and card['legalities']['commander']=="Legal" and 'paper' in card['availability']:
                    cd = CardFields.parse_mtgjson_card(card)
                    attributes = cd.get_attributes()
                    attributes['tags'] = CardFields.tag_card(cd)
                    for attribute, value in attributes.items():
                        if attribute not in card_data:
                            card_data[attribute] = [value]
                        else:
                            card_data[attribute].append(value)

        return pd.DataFrame(card_data).drop_duplicates(subset=['rank'], keep='first').drop_duplicates(subset=['card_name'],keep='first').reset_index()

    def name_query(self, name: str) -> pd.DataFrame:
        """Filter cards by partial name match."""
        return self.card_set[self.card_set['card_name'].str.contains(name)]

    def rank_query(self, rank: int) -> pd.DataFrame:
        """Filter cards by exact rank."""
        return self.card_set[self.card_set['rank']==rank]

    def query(self, queries: List[str]) -> pd.DataFrame:
        """Apply one or more pandas query expressions in sequence."""
        for query in queries:
            self.card_set = self.card_set.query(query)
        return self.card_set

    def write_data_csv(self, filename: str) -> None:
        """Write the current card dataframe to CSV."""
        self.card_set.to_csv(filename, index=False)

    def write_data_json(self, filename: str) -> None:
        """Write the current card dataframe to JSON records."""
        self.card_set.to_json(filename, orient="records", indent=4)

    def get_card_set(self) -> pd.DataFrame:
        """Return the current card dataframe."""
        return self.card_set

    def format_output(self, txt: str) -> str:
        """Normalize output text by stripping punctuation and newlines."""
        return re.sub(r'[\n]', '. ', re.sub(r',\[\]\(\)"\'', '', str(txt)))


if __name__ == "__main__":
    ds = DataSet()
    ds.write_data_json(os.environ.get("CARDS_DATASET"))
