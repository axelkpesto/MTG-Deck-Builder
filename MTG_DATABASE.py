from CARD_DATA import Card, CardFields
import os, re
import pandas as pd
import boto3

class DataSet(object):
    def __init__(self):
        self.SET_DATAFRAME = self.JSON_DATA_SETUP("AllPrintings.json")
        self.card_set = self.PARSE_SET_DATA()

    def AWS_DATA_REQUEST(self, file: str) -> pd.DataFrame:
        AWS_S3_BUCKET: str = 'allcarddata'
        REGION_NAME: str = 'us-east-1'

        AWS_ACCESS_KEY_ID: str = str(os.getenv("AWS_ACCESS_KEY_ID"))
        AWS_SECRET_ACCESS_KEY: str = str(os.getenv("AWS_SECRET_ACCESS_KEY"))

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=REGION_NAME
        )

        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=file)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        assert(status==200)
        return pd.read_json(response.get("Body"))['data'][2:]

    def JSON_DATA_SETUP(self, filename: str):
        assert os.path.isfile(filename), f"{filename} not found."
        return pd.read_json(filename)['data'][2:]

    def PARSE_SET_DATA(self) -> pd.DataFrame:
        card_data: dict = dict()
        for game_set in self.SET_DATAFRAME:
            for card in game_set['cards']:
                if 'commander' in card['legalities'] and card['legalities']['commander']=="Legal" and 'paper' in card['availability']:
                    cd = Card(card)
                    attributes = cd.get_attributes()
                    for attribute, value in attributes.items():
                        if attribute not in card_data:
                            card_data[attribute] = [value]
                        else:
                            card_data[attribute].append(value)

        return pd.DataFrame(card_data).drop_duplicates(subset=['rank'],keep='first').drop_duplicates(subset=['card_name'],keep='first').reset_index()

    def NAME_QUERY(self, name: str) -> pd.DataFrame:
        return self.card_set[self.card_set['card_name'].str.contains(name)]

    def RANK_QUERY(self, rank: int):
        return self.card_set[self.card_set['rank']==rank]

    def QUERY(self, queries:list[str]):
        for query in queries:
            self.card_set = self.card_set.query(query)
        return self.card_set

    def WRITE_DATA_CSV(self, filename: str) -> None:
        self.card_set.to_csv(filename, index=False)
    
    def WRITE_DATA_JSON(self, filename: str) -> None:
        self.card_set.to_json(filename, orient="records", indent=4)
    
    def GET_CARD_SET(self): 
        return self.card_set

    def format_output(self, txt: str):
        return re.sub(r'[\n]', '. ', re.sub(r',\[\]\(\)"\'', '', str(txt)))

if __name__ == "__main__":
    ds = DataSet()
    ds.WRITE_DATA_JSON("CommanderCards.json")