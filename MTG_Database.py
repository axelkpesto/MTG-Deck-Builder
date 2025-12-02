from Card_Lib import CardFields
import os
import re
import pandas as pd
from typing import List, Dict, Any


class DataSet(object):
    def __init__(self) -> None:
        self.SET_DATAFRAME: pd.Series = self.JSON_DATA_SETUP("datasets/AllPrintings.json")
        self.card_set: pd.DataFrame = self.PARSE_SET_DATA()

    def JSON_DATA_SETUP(self, filename: str) -> pd.Series:
        assert os.path.isfile(filename), f"{filename} not found."
        return pd.read_json(filename)['data'][2:]

    def PARSE_SET_DATA(self) -> pd.DataFrame:
        card_data: Dict[str, List[Any]] = dict()
        for game_set in self.SET_DATAFRAME:
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

        return pd.DataFrame(card_data).drop_duplicates(subset=['rank'],keep='first').drop_duplicates(subset=['card_name'],keep='first').reset_index()

    def NAME_QUERY(self, name: str) -> pd.DataFrame:
        return self.card_set[self.card_set['card_name'].str.contains(name)]

    def RANK_QUERY(self, rank: int) -> pd.DataFrame:
        return self.card_set[self.card_set['rank']==rank]

    def QUERY(self, queries: List[str]) -> pd.DataFrame:
        for query in queries:
            self.card_set = self.card_set.query(query)
        return self.card_set

    def WRITE_DATA_CSV(self, filename: str) -> None:
        self.card_set.to_csv(filename, index=False)

    def WRITE_DATA_JSON(self, filename: str) -> None:
        self.card_set.to_json(filename, orient="records", indent=4)

    def GET_CARD_SET(self) -> pd.DataFrame:
        return self.card_set

    def format_output(self, txt: str) -> str:
        return re.sub(r'[\n]', '. ', re.sub(r',\[\]\(\)"\'', '', str(txt)))

if __name__ == "__main__":
    ds = DataSet()
    ds.WRITE_DATA_JSON("datasets/CommanderCards.json")
