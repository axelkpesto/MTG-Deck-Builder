import os
import boto3
import pandas as pd
import time

from Card import Card

pd.set_option('display.max_columns', None)

class DataSet(object):
    def __init__(self):
        start_time = time.time()
        self.SET_DATAFRAME:pd.DataFrame = self.JSON_DATA_SETUP('AllPrintings.json') if os.path.isfile('AllPrintings.json') else self.AWS_DATA_REQUEST("AllPrintings.json")
        print("READ JSON: " + str(time.time()- start_time))

        start_time = time.time()
        self.card_set:pd.DataFrame = self.PARSE_SET_DATA()
        print("PARSE DATA: " + str(time.time()- start_time))

        start_time = time.time()
        self.WRITE_DATA('Training.txt')
        print("WRITE DATA: " + str(time.time()- start_time))

    def AWS_DATA_REQUEST(self, file:str) -> pd.DataFrame:
        AWS_S3_BUCKET:str = 'allcarddata'
        REGION_NAME:str = 'us-east-1'

        AWS_ACCESS_KEY_ID:str = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY:str = os.getenv("AWS_SECRET_ACCESS_KEY")
        
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

    def JSON_DATA_SETUP(self, filename:str):
        assert os.path.isfile(filename), f"{filename} not found."
        return pd.read_json(filename)['data'][2:]
    
    def PARSE_SET_DATA(self) -> pd.DataFrame:
        card_data:dict = dict()
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
    
    def NAME_QUERY(self, name:str, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None: df = self.card_set
        return df[df['card_name'].str.contains(name)]
    
    # def TYPE_QUERY(self, card_type:str, df:pd.DataFrame = None) -> pd.DataFrame:
    #     if df is None: df = self.card_set
    #     return df[[card_type in c for c in list(df['card_types'])]]
    
    # def SUPERTYPE_QUERY(self, supertype:str, df:pd.DataFrame = None) -> pd.DataFrame:
    #     if df is None: df = self.card_set
    #     return df[[supertype in c for c in list(df['card_supertypes'])]]

    # def CREATURE_TYPE_QUERY(self, creature_type:str, df:pd.DataFrame = None) -> pd.DataFrame:
    #     if df is None: df = self.card_set
    #     return df[[creature_type in c for c in list(df['card_subtypes'])]]
    
    def RANK_QUERY(self, rank:int, df:pd.DataFrame = None):
        if df is None: df = self.card_set
        return df[df['rank']==rank]
    
    def QUERY(self, queries:list[str], df:pd.DataFrame = None):
        if df is None: df = self.card_set
        for query in queries:
            df = df.query(query)
        return df

    def WRITE_DATA(self, filename:str, df:pd.DataFrame = None) -> None:
        assert os.path.isfile(filename), f"{filename} not found."
        if df is None: df = self.card_set
        df.to_csv(filename,index=False)

if __name__ == "__main__":
    ds = DataSet()