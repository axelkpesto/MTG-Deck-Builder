#https://mtgjson.com/downloads/all-files/
#https://mtgjson.com/data-models/set/
#https://mtgjson.com/data-models/card/card-set/


import os
import re
import boto3  # type: ignore
import pandas as pd  # type: ignore
import time
import sys

pd.set_option('display.max_columns', None)

class Card(object):
    def __init__(self, cd:dict) -> None:
        if cd is None or {}: return
        self.card_name:str = re.sub(r'[,\n\[\]\(\)\'"]','',str(cd['name'])) if 'name' in cd else None #Name of the Card. None if property doesn't exist.
        self.card_types:list[str] = cd['types']  if 'types' in cd else None #Card Types (EX: CREATURE) of the Card. None if property doesn't exist.
        self.card_supertypes:str = re.sub(r'[,\n\[\]\(\)\'"]','',str(cd['supertypes'])) if 'supertypes' in cd else None #Supertypes (EX: LEGENDARY) of the Card. None if property doesn't exist.
        self.card_subtypes:list[str] = re.sub(r'[,\n\[\]\(\)\'"]','',str(cd['subtypes'])).split(' ')  if 'subtypes' in cd else None #Subtypes (EX: GOBLIN) of the Card. None if property doesn't exist.
        self.mana_cost:int = cd['manaValue']  if 'manaValue' in cd else None #Numerical Mana Cost of the Card. None if property doesn't exist.
        self.mana_cost_exp:str = cd['manaCost']  if 'manaCost' in cd else None #Expanded Mana Cost of the Card. None if property doesn't exist.
        self.color_identity:list[str] = [x for x in list(set(self.mana_cost_exp)) if x.isalpha()]  if 'manaCost' in cd else None #Colors contained in the Expanded Mana Cost of the Card. None if property doesn't exist.
        self.defense:str = cd['defense'] if 'defense' in cd else None #Defense of a Battle Card. None if the Card is not a Battle.
        self.commander_legal:bool = ('commander' in cd['legalities'])  if 'legalities' in cd else None #Commander Legality of the Card. None if property doesn't exist.
        self.keywords:list[str] = re.sub(r'[,\n\[\]\(\)\'"]','',str(cd['supertypes'])).split(' ')  if 'keywords' in cd else None #Keywords (EX: FIRST STRIKE) of the Card. None if property doesn't exist.
        self.rarity:str = cd['rarity']  if 'rarity' in cd else None #Rarity (EX: UNCOMMON) of the Card. None if property doesn't exist.
        self.text:str = re.sub(r'[\n]','. ', re.sub(r'[,\[\]\(\)"]','',str(cd['text'])))  if 'text' in cd else None #Text of the Card. None if property doesn't exist.
        self.rank:int = cd['edhrecRank']  if 'edhrecRank' in cd else None #EDHREC Rank of the Card. None if property doesn't exist.
        self.power:str = cd['power'] if 'Creature' in self.card_types and 'power' in cd else None #Power of a Creature Card. None if Card is not a Creature.
        self.toughness:str = cd['toughness'] if 'Creature' in self.card_types and 'toughness' in cd else None #Toughness of a Creature Card. None if Card is not a Creature.
        self.loyalty:str = cd['loyalty'] if 'Planeswalker' in self.card_types and 'loyalty' in cd else None #Starting Loyalty of a Planeswalker Card. None if Card is not a Planeswalker.
        self.tags:list[str] = list(self.tag_text(self.text).union(self.tag_subtypes(str(self.card_subtypes)))) #Tagging cards for data analysis
        self.associations:list[str] = ['']*len(self.tags) #Empty List for Manual Tagging Ease
        if self.color_identity is not None: self.color_identity.sort() #Sorting Color Identity to ensure consistency in comparison.
        
    def tag_text(self, text:str) -> set[str]:
        if text==None or text=="": return set()
        text = text.replace(self.card_name,'').lower()
        tags = {
            'aggro': [r'haste', r'attack each turn', r'deal \d+ damage'],
            'control': [r'counter target \w spell', r'destroy target', r'exile target', r'return target', r'tap target', r'prevent all damage', r'flash', r'scry',r'can\'t activate',r'can\'t attack',r'can\'t block',r'countered', r'return each', r'destroy that'],
            'combo': [r'You win the game',r'Opponent loses the game',r'for each',r'without paying it\'s mana cost', r'storm',r'cast for {0}',r'The legend rule doesn\'t apply',r'without paying its mana cost'],
            'ramp': [r'add mana', r'search your library for a land', r'landfall', r'add {\w}',r'treasure', r'search your library for a basic land', r'search your library for a (|[a-zA-Z]+) land',r'basic lands',r'basic land'],
            'card_draw': [r'draw (\d+|[a-zA-Z]+) cards', r'draw a card',r'blood token',r'you may play that card this turn'],
            'party': [r'Cleric',r'Rogue',r'Warrior',r'Wizard'],
            'token': [r'token', r'populate', r'copy', r'token creature', r'tokens you control'],
            'life_gain': [r'gain life', r'life equal to', r'whenever you gain life', r'life total', r'lifelink',r'food token',r'gain \d life'],
            'mill': [r'put into your graveyard', r'mill', r'from the top of their library', r'library into graveyard'],
            'discard': [r'discard a card', r'discard cards', r'each player discards', r'from their hand'],
            'reanimator': [r'return target creature card', r'graveyard to the battlefield', r'reanimate', r'bring back', r'from your graveyard'],
            'burn': [r'deal \d+ damage', r'damage to any target', r'damage to each opponent',r'damage to target creature',r'damage to target planeswalker',r'damage to any target', r'deals \d+ damage', r'deals \w damage'],
            'enchantment': [r'enchant', r'aura', r'whenever you cast an enchantment', r'constellation'],
            'equipment': [r'equip', r'attach', r'whenever equipped', r'whenever you attach'],
            'artifact': [r'artifact', r'whenever you cast an artifact', r'metalcraft'],
            'planeswalker': [r'planeswalker', r'loyalty', r'loyalty counter'],
            'tribal': [r'Elf', r'Goblin', r'Zombie', r'Vampire', r'Warrior', r'Merfolk', r'Soldier', r'Dragon', r'Angel', r'Wizard', r'Knight', r'Sliver'],
            'flicker': [r'exile and return', r'flicker', r'blink', r're-enter the battlefield'],
            'voltron': [r'equip', r'attach', r'whenever equipped', r'whenever you attach', r'aura'],
            'infect': [r'infect', r'poison counter', r'proliferate'],
            'stax': [r'tap target', r'opponent can\'t untap', r'opponents can\'t draw', r'whenever an opponent',r'can\'t cast',r'can\'t activate',r'can\'t attack',r'prevent that damage',r'prevent damage',r'deals damage to you'],
            'storm': [r'storm', r'copy this spell', r'copy that spell', r'copy target spell',r'magecraft'],
            'graveyard': [r'graveyard', r'return from your graveyard', r'mill',r'from your graveyard',r'unearth'],
            'sacrifice': [r'sacrifice a creature', r'sacrifice a permanent', r'whenever you sacrifice'],
            'combat': [r'combat damage to a player', r'combat damage to an opponent', r'vigilance', r'combat damage', r'attacks',r'first strike', r'double strike',r'target attacking', r'target blocking', r'whenever you attack', r'whenever  attacks', r'whenever  blocks', r'each combat', r'blocks', r'attacking', r'blocking', r'can\'t be blocked', r'deals damage to a player'],
            'buff': [r'\+\d/\++\d',r'\+\w/\++\w',r'\+\w/\++\d',r'\+\d/\++\w',r'\-\d/\++\d',r'\-\w/\++\w',r'\-\w/\++\d',r'\-\d/\++\w',r'\+\d/\-+\d',r'\+\w/\-+\w',r'\+\w/\-+\d',r'\+\d/\-+\w'],
            'outlaw': [r'Assassin', r'Mercenary', r'Pirate', r'Rogue',r'Warlock'],
            'removal':[r'Destroy target',r'Destroy all',r'Exile target', r'Exile all',r'Remove from the game', r'Removed from the game'],
            'discard':[r'discard',r'madness',r'blood'],
            'etb':[r'enters the battlefield',r'enters under your control', r'enter the battlefield'],
            'library_control':[r'scry',r'on top of your library',r'card into your hand',r'on the top of your library', r'look at the top'],
            'extb':[r'whenever a creature dies',r'whenever another creature dies', r'whenever a creature is put into your graveyard', r'whenever this creature dies', r'when  dies', r'regenerate'],
            'protection':[r'indestructible',r'prevent all damage',r'protection from', r'shroud', r'hexproof', r'you have hexproof', r'you have shroud', r'creatures you control gain indestructible', r'creatures you control gain hexproof', r'creatures you control gain shroud', r'target creature you control gains hexproof']
        }

        tags_joint = {
            'control':[['counter','spell'],],
            'ramp':[['Untap','Land'],['Search your library','land'],['Add','mana'],['Add','{','}']],
            'protection':[['prevent','damage']]
        }

        card_tags:set[str] = set()
        for tag, patterns in tags.items():
            for pattern in patterns:
                if re.search(pattern, str(text), re.IGNORECASE):
                    card_tags.add(str(tag))
        
        for tag, patterns in tags_joint.items():
            for pattern in patterns:
                add = True
                for i in range(len(pattern)):
                    if pattern[i].lower() not in str(text):
                        add = False
                if add: card_tags.add(str(tag))
        return card_tags

        #DOESNT WORK
    def tag_subtypes(self, text:list[str]) -> set[str]:
        if text==None or text==[]: return set()
        text = set([x.lower() for x in text])
        card_tags = set()
        subtype_tags = {
            'tribal': ['Elf', 'Goblin', 'Zombie', 'Vampire', 'Warrior', 'Merfolk', 'Soldier', 'Dragon', 'Angel', 'Wizard', 'Knight', 'Sliver'],
            'outlaw': ['Assassin', 'Mercenary', 'Pirate', 'Rogue','Warlock'],
            'party': ['Cleric','Rogue','Warrior','Wizard']
        }
        
        for tag, patterns in subtype_tags.items():
            for pattern in patterns:
                if pattern in text:
                    card_tags.add(tag)
        return card_tags

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Card):
            return self.card_name == value.card_name
        return False
    
    def __str__(self) -> str:
        return ("NAME: " + self.card_name + "\nCARD TYPES: " + str(self.card_types) + "\nCOLORS: " + str(self.color_identity))

    def __hash__(self):
        return hash(self.card_name)
    
    def get_attributes(self) -> dict:
        """
        get_attributes maps the name of attributes to the value of the attribute.

        return: dictionary of string:attribute
        """
        return {
            'card_name':self.card_name,
            'card_types':self.card_types,
            'card_supertypes':self.card_supertypes,
            'card_subtypes':self.card_subtypes,
            'mana_cost':self.mana_cost,
            'mana_cost_exp':self.mana_cost_exp,
            'color_identity':self.color_identity,
            'defense':self.defense,
            'commander_legal':self.commander_legal,
            'keywords':self.keywords,
            'rarity':self.rarity,
            'text':self.text,
            'rank':self.rank,
            'power':self.power,
            'toughness':self.toughness,
            'loyalty':self.loyalty,
            'tags':self.tags,
            'associations':self.associations
        }

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
    
    def TYPE_QUERY(self, card_type:str, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None: df = self.card_set
        return df[[card_type in c for c in list(df['card_types'])]]
    
    def SUPERTYPE_QUERY(self, supertype:str, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None: df = self.card_set
        return df[[supertype in c for c in list(df['card_supertypes'])]]

    def CREATURE_TYPE_QUERY(self, creature_type:str, df:pd.DataFrame = None) -> pd.DataFrame:
        if df is None: df = self.card_set
        return df[[creature_type in c for c in list(df['card_subtypes'])]]
    
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