import numpy as np
import re

class Card(object):
    def __init__(self, cd:dict) -> None:
        if cd is None or {}: return
        self.card_name: str = str(cd['name']) if 'name' in cd else None #Name of the Card. None if property doesn't exist.
        self.card_types: list[str] =  [str(x).lower() for x in cd['types']]  if 'types' in cd else None #Card Types (EX: CREATURE) of the Card. None if property doesn't exist.
        self.card_supertypes: str = str(cd['supertypes']).lower() if 'supertypes' in cd else None #Supertypes (EX: LEGENDARY) of the Card. None if property doesn't exist.
        self.card_subtypes: list[str] = [str(x).lower() for x in cd['subtypes']]  if 'subtypes' in cd else None #Subtypes (EX: GOBLIN) of the Card. None if property doesn't exist.
        self.mana_cost: int = cd['manaValue']  if 'manaValue' in cd else None #Numerical Mana Cost of the Card. None if property doesn't exist.
        self.mana_cost_exp: str = cd['manaCost']  if 'manaCost' in cd else None #Expanded Mana Cost of the Card. None if property doesn't exist.
        self.color_identity: list[str] = [x.upper() for x in sorted([x for x in list(set(self.mana_cost_exp)) if x.isalpha()])] if 'manaCost' in cd else None #Colors contained in the Expanded Mana Cost of the Card. None if property doesn't exist.
        self.defense: str = cd['defense'] if 'defense' in cd else None #Defense of a Battle Card. None if the Card is not a Battle.
        self.rarity: str = cd['rarity']  if 'rarity' in cd else None #Rarity (EX: UNCOMMON) of the Card. None if property doesn't exist.
        self.text: str = cd['text']  if 'text' in cd else None #Text of the Card. None if property doesn't exist.
        self.rank: int = cd['edhrecRank']  if 'edhrecRank' in cd else None #EDHREC Rank of the Card. None if property doesn't exist.
        self.power: str = cd['power'] if 'Creature' in self.card_types and 'power' in cd else None #Power of a Creature Card. None if Card is not a Creature.
        self.toughness: str = cd['toughness'] if 'Creature' in self.card_types and 'toughness' in cd else None #Toughness of a Creature Card. None if Card is not a Creature.
        self.loyalty: str = cd['loyalty'] if 'Planeswalker' in self.card_types and 'loyalty' in cd else None #Starting Loyalty of a Planeswalker Card. None if Card is not a Planeswalker.
        self.id: str = cd['identifiers']['multiverseId'] if 'identifiers' in cd and 'multiverseId' in cd['identifiers'] else None #MultiverseID Identifier
        self.tags: list[str] = [x.lower() for x in list(self.tag_text(self.text).union(self.tag_subtypes(str(self.card_subtypes))))] #Tagging cards for data analysis
    
    def tag_text(self, text: str) -> set[str]:
        if text==None or text=="": return set()
        text = text.replace(self.card_name,'').lower()

        card_tags: set[str] = set()

        for tag, patterns in CardFields.general_tags().items():
            if tag in card_tags: continue
            for pattern in patterns:
                if pattern.lower() in text:
                    card_tags.add(str(tag).lower())
                    break

        for tag, patterns in CardFields.joint_tags().items():
          if tag in card_tags: continue
          for pattern in patterns:
              add = True
              for i in range(len(pattern)):
                  if pattern[i].lower() not in str(text):
                      add = False
                      break
              if add: 
                card_tags.add(str(tag).lower())
                break

        for tag, patterns in CardFields.regex_tags().items():
          if tag in card_tags: continue
          for pattern in patterns:
              if re.search(pattern, str(text), re.IGNORECASE):
                  card_tags.add(str(tag).lower())
                  break
        
        return card_tags

    def tag_subtypes(self, text: list[str]) -> set[str]:
        if text==None or text==[]: return set()
        card_tags:set[str] = set()

        for tag, patterns in CardFields.subtype_tags().items():
            for pattern in patterns:
                if pattern in text:
                    card_tags.add(str(tag).lower())
                    break
        return card_tags

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Card):
            return self.card_name == value.card_name
        return False

    def __str__(self) -> str:
        return ("NAME: " + self.card_name + "\nCARD TYPES: " + str(self.card_types) + "\nCOLORS: " + str(self.color_identity))

    def __hash__(self) -> int:
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
            'rarity':self.rarity,
            'text':self.text,
            'rank':self.rank,
            'power':self.power,
            'toughness':self.toughness,
            'loyalty':self.loyalty,
            'id':self.id,
            'tags':self.tags,
        }

class CardFields(object):
    __tags_general = {
        'aggro': ['haste', 'attack each turn',],
        'control': ['destroy target', 'exile target', 'return target', 'tap target', 'prevent all damage', 'flash', 'scry','can\'t activate','can\'t attack','can\'t block','countered','return each','destroy that','gain control','tap all','change the target','return all'],
        'combo': ['You win the game','Opponent loses the game','for each','without paying it\'s mana cost', 'storm','cast for {0}','The legend rule doesn\'t apply','without paying its mana cost','rather than paying its mana value','rather than paying it\'s mana value','rather than paying its mana cost','rather than paying it\'s mana cost'],
        'ramp': ['add mana', 'search your library for a land', 'landfall','treasure', 'search your library for a basic land','basic lands','basic land'],
        'card_draw': ['draw a card','blood token','you may play that card this turn'],
        'party': ['Cleric','Rogue','Warrio','Wizard','party'],
        'token': ['token', 'populate', 'copy', 'token creature', 'tokens you control'],
        'life_gain': ['gain life', 'whenever you gain life', 'life total', 'lifelink','food token','you gain that much life',],
        'life_drain':['lose life', 'loses that much life', 'paying life','pays life','pay life'],
        'mill': ['put into your graveyard', 'mill', 'from the top of their library', 'library into graveyard'],
        'discard': ['discard a card', 'discard cards', 'each player discards', 'from their hand'],
        'reanimaton': ['return target creature card', 'graveyard to the battlefield', 'reanimate', 'bring back', 'from your graveyard','regenerate'],
        'burn': ['damage to any target', 'damage to each opponent','damage to target creature','damage to target planeswalker','damage to any target','deals damage equal to'],
        'enchantment': ['enchant', 'aura', 'whenever you cast an enchantment', 'constellation','saga'],
        'equipment': ['equip', 'attach', 'whenever equipped', 'whenever you attach','when equipped'],
        'artifact': ['artifact', 'whenever you cast an artifact', 'metalcraft'],
        'planeswalker': ['planeswalker', 'loyalty', 'loyalty counter'],
        'tribal': ['Elf', 'Goblin', 'Zombie', 'Vampire', 'Warrior', 'Merfolk', 'Soldier', 'Dragon', 'Angel', 'Wizard', 'Knight', 'Sliver'],
        'flicker': ['exile and return', 'flicker', 'blink', 're-enter the battlefield'],
        'voltron': ['equip', 'attach', 'whenever equipped', 'whenever you attach', 'aura'],
        'infect': ['infect', 'poison counter', 'proliferate'],
        'stax': ['tap target', 'opponent can\'t untap', 'opponents can\'t draw', 'whenever an opponent','can\'t cast','can\'t activate','can\'t attack','prevent that damage','prevent damage','deals damage to you'],
        'storm': ['storm', 'copy this spell', 'copy that spell', 'copy target spell','magecraft','copy target instant','copy target sorcery'],
        'graveyard': ['graveyard', 'return from your graveyard', 'mill','from your graveyard','unearth'],
        'sacrifice': ['sacrifice a creature', 'sacrifice a permanent', 'whenever you sacrifice','sacrifice'],
        'combat': ['combat damage to a player', 'combat damage to an opponent', 'vigilance', 'combat damage', 'attacks','first strike', 'double strike','target attacking', 'target blocking', 'whenever you attack', 'whenever  attacks', 'whenever  blocks', 'each combat', 'blocks', 'attacking', 'blocking', 'can\'t be blocked', 'deals damage to a player', 'flying','raid'],
        'outlaw': ['Assassin', 'Mercenary', 'Pirate', 'Rogue','Warlock'],
        'removal':['Destroy target','Destroy all','Exile target', 'Exile all','Remove from the game', 'Removed from the game', 'Destroy',],
        'discard':['discard','madness','blood'],
        'buff':['all tokens you control','all creatures you control'],
        'etb':['enters the battlefield','enters under your control', 'enter the battlefield'],
        'library_control':['scry','on top of your library','card into your hand','on the top of your library', 'look at the top'],
        'extb':['whenever a creature dies','whenever another creature dies', 'whenever a creature is put into your graveyard', 'whenever this creature dies', 'when  dies', 'regenerate'],
        'protection':['indestructible','prevent all damage','protection from', 'shroud', 'hexproof', 'you have hexproof', 'you have shroud', 'creatures you control gain indestructible', 'creatures you control gain hexproof', 'creatures you control gain shroud', 'target creature you control gains hexproof']
    }

    __tags_joint = {
        'control':[['counter','spell'],],
        'ramp':[['Untap','Land'],['Search your library','land'],['Add','mana'],['Add','{','}'],],
        'protection':[['prevent','damage'],],
        'flicker':[['exile','return']],
        'life_gain':[['gain','life']],
        'life_drain':[['lose','life'],['pay','life']],
    }

    __tags_regex = {
        'aggro': [r'attack each turn', r'deal \d+ damage'],
        'combo': [r'cast for {0}'],
        'control': [r'counter target \w spell',],
        'ramp': [r'add {\w}', r'search your library for a (|[a-zA-Z]+) land',],
        'card_draw': [r'draw (\d+|[a-zA-Z]+) cards',],
        'life_gain': [r'gain \d life',r'gain \w life',r'gains \d life',r'gains \w life'],
        'life_drain': [r'loses \d life',r'loses \w life',r'pays \d life',r'pays \w life'],
        'burn': [r'deal \d+ damage', r'deals \d+ damage', r'deals \w damage'],
        'stax': [r'cost {\d+} more to cast'],
        'buff': [r'\+\d/\++\d',r'\+\w/\++\w',r'\+\w/\++\d',r'\+\d/\++\w',r'\-\d/\++\d',r'\-\w/\++\w',r'\-\w/\++\d',r'\-\d/\++\w',r'\+\d/\-+\d',r'\+\w/\-+\w',r'\+\w/\-+\d',r'\+\d/\-+\w'],
    }

    __subtype_tags = {
        'tribal': ['Elf', 'Goblin', 'Zombie', 'Vampire', 'Warrior', 'Merfolk', 'Soldier', 'Dragon', 'Angel', 'Wizard', 'Knight', 'Sliver'],
        'outlaw': ['Assassin', 'Mercenary', 'Pirate', 'Rogue','Warlock'],
        'party': ['Cleric','Rogue','Warrior','Wizard']
    }
    
    __card_types = ['Enchantment', 'Land', 'Artifact', 'Creature', 'Instant', 'Sorcery', 'Planeswalker', 'Battle']
    
    __card_supertypes = ['Legendary', 'Basic']

    __creature_subtypes = ['Advisor', 'Aetherborn', 'Ally', 'Angel', 'Anteater', 'Antelope', 'Ape', 'Archer', 'Archon', 'Artificer', 'Assassin', 'Assembly-Worker', 'Atog', 'Aurochs', 'Avatar', 'Badger', 'Barbarian', 'Basilisk', 'Bat', 'Bear', 'Beast', 'Beeble', 'Berserker', 'Bird', 'Blinkmoth', 'Boar', 'Bringer', 'Brushwagg', 'Camarid', 'Camel', 'Caribou', 'Carrier', 'Cat', 'Centaur', 'Cephalid', 'Chimera', 'Citizen', 'Cleric', 'Cockatrice', \
    'Construct', 'Coward', 'Crab', 'Crocodile', 'Cyclops', 'Dauthi', 'Demon', 'Deserter', 'Devil', 'Dinosaur', 'Djinn', 'Dragon', 'Drake', 'Dreadnought', 'Drone', 'Druid', 'Dryad', 'Dwarf', 'Efreet', 'Elder', 'Eldrazi', 'Elemental', 'Elephant', 'Elf', 'Elk', 'Eye', 'Faerie', 'Ferret', 'Fish', 'Flagbearer', 'Fox', 'Frog', 'Fungus', 'Gargoyle', 'Germ', 'Giant', 'Gnome', 'Goat', 'Goblin', 'God', \
    'Golem', 'Gorgon', 'Graveborn', 'Gremlin', 'Griffin', 'Hag', 'Harpy', 'Hellion', 'Hippo', 'Hippogriff', 'Hormarid', 'Homunculus', 'Horror', 'Horse', 'Hound', 'Human', 'Hydra', 'Hyena', 'Illusion', 'Imp', 'Incarnation', 'Insect', 'Jellyfish', 'Juggernaut', 'Kavu', 'Kirin', 'Kithkin', 'Knight', 'Kobold', 'Kor', 'Kraken', 'Lamia', 'Lammasu', 'Leech', 'Leviathan', 'Lhurgoyf', 'Licid', 'Lizard', 'Manticore', 'Masticore', \
    'Mercenary', 'Merfolk', 'Metathran', 'Minion', 'Minotaur', 'Mole', 'Monger', 'Mongoose', 'Monk', 'Moonfolk', 'Mutant', 'Myr', 'Mystic', 'Naga', 'Nautilus', 'Necron', 'Nephilim', 'Nightmare', 'Nightstalker', 'Ninja', 'Noggle', 'Nomad', 'Nymph', 'Octopus', 'Ogre', 'Ooze', 'Orb', 'Orc', 'Orgg', 'Ouphe', 'Ox', 'Oyster', 'Pegasus', 'Pentavite', 'Pest', 'Phelddagrif', 'Phoenix', 'Pincher', 'Pirate', \
    'Plant', 'Praetor', 'Prism', 'Processor', 'Rabbit', 'Rat', 'Rebel', 'Reflection', 'Rhino', 'Rigger', 'Rogue', 'Sable', 'Salamander', 'Samurai', 'Sand', 'Saproling', 'Satyr', 'Scarecrow', 'Scion', 'Scorpion', 'Scout', 'Serf', 'Serpent', 'Shade', 'Shaman', 'Shapeshifter', 'Sheep', 'Siren', 'Skeleton', 'Slith', 'Sliver', 'Slug', 'Snake', 'Soldier', 'Soltari', 'Spawn', 'Specter', 'Spellshaper', 'Sphinx', 'Spider', 'Spike',\
    'Spirit', 'Splinter', 'Sponge', 'Squid', 'Squirrel', 'Starfish', 'Surrakar', 'Survivor', 'Tetravite', 'Thalakos', 'Thopter', 'Thrull', 'Treefolk', 'Triskelavite', 'Troll', 'Turtle', 'Unicorn', 'Vampire', 'Vedalken', 'Viashino', 'Volver', 'Wall', 'Warrior', 'Weird', 'Werewolf', 'Whale', 'Wizard', 'Wolf', 'Wolverine', 'Wombat', 'Worm', 'Wraith', 'Wurm', 'Yeti', 'Zombie', 'Zubera']

    __artifact_subtypes = ['Blood', 'Clue', 'Food', 'Gold', 'Incubator', 'Junk', 'Map', 'Powerstone', 'Treasure', 'Equipment', 'Fortification', 'Vehicle', 'Attraction', 'Contraption']

    __battle_subtypes = ['Seige']

    __enchantment_subtypes = ['Aura', 'Background', 'Saga', 'Role', 'Shard', 'Cartouche', 'Case', 'Class', 'Curse', 'Rune', 'Shrine']

    __land_subtypes = ['Plains', 'Forest', 'Mountain', 'Island', 'Swamp', 'Cave', 'Desert', 'Gate', 'Lair', 'Locus', 'Mine', 'Power-Plant', 'Sphere', 'Tower', 'Urza']

    __spell_subtypes = ['Adventure', 'Arcane', 'Chorus', 'Lesson', 'Trap']

    __all_subtypes = __creature_subtypes + __artifact_subtypes + __battle_subtypes + __enchantment_subtypes + __land_subtypes + __spell_subtypes

    __color_identities = ['W', 'G', 'U', 'B', 'R', 'C']

    @staticmethod
    def card_types() -> list[str]: return sorted(CardFields.__card_types)

    @staticmethod
    def card_supertypes() -> list[str]: return sorted(CardFields.__card_supertypes)

    @staticmethod
    def card_subtypes() -> list[str]: return CardFields.__all_subtypes

    @staticmethod
    def color_identities() -> list[str]: return sorted(CardFields.__color_identities)

    @staticmethod
    def card_tags() -> list[str]: return sorted(CardFields.__tags_general.keys())
    
    @staticmethod
    def general_tags() -> dict[str,str]: return CardFields.__tags_general

    @staticmethod
    def joint_tags() -> dict[str,str]: return CardFields.__tags_joint

    @staticmethod
    def regex_tags() -> dict[str,str]: return CardFields.__tags_regex
    
    @staticmethod
    def subtype_tags() -> dict[str,str]: return CardFields.__subtype_tags

class CardEncoder(object):
    def __init__(self):
        self.card_types:list[str] = CardFields.card_types()
        self.card_supertypes:list[str] = CardFields.card_supertypes()
        self.all_subtypes:list[str] = CardFields.card_subtypes()
        self.color_identities:list[str] = CardFields.color_identities()
        self.tags:list[str] = CardFields.card_tags()

    def encode(self, crd:Card) -> tuple[str,np.array]:
        ret = []

        cd = [0]*len(self.card_types)
        for i in range(len(self.card_types)):
            if self.card_types[i].lower() in crd.card_types:
                cd[i] = 1
        ret += cd
        cd = [0]*len(self.card_supertypes)
        for i in range(len(self.card_supertypes)):
            if self.card_supertypes[i].lower() in crd.card_supertypes:
                cd[i] = 1
        ret += cd

        cd = [0]*len(self.all_subtypes)
        for i in range(len(self.all_subtypes)):
            if self.all_subtypes[i].lower() in crd.card_subtypes:
                cd[i] = 1
        ret += cd

        ret += [int(crd.mana_cost)]
        
        cd = [0]*len(self.color_identities)
        if crd.color_identity == [] or crd.color_identity==None:
            if crd.mana_cost>0 or crd.mana_cost==None:
                cd[self.color_identities.index("C")] = 1
        else:
            for i in range(len(self.color_identities)):
                if self.color_identities[i].upper() in crd.color_identity:
                    cd[i] = 1
        ret += cd

        ret += [self.rarity_to_int(crd.card_name, crd.rarity.lower())]

        cd = [0]*len(self.tags)
        for i in range(len(self.tags)):
            if self.tags[i].lower() in crd.tags:
                cd[i] = 1
        ret += cd

        return (crd.card_name, np.array(ret))

    def rarity_to_int(self, name:str, rarity:str) -> int:
        match rarity.lower():
            case 'common':
                return 1
            case 'uncommon':
                return 2
            case 'rare':
                return 3
            case 'mythic':
                return 4
            case 'timeshifted':
                return 5
            case 'masterpiece':
                return 6
            case 'special':
                return 7
            case _:
                print("NAME: " + name + "RARITY?: " + str(rarity))
                raise Exception('Rarity not found')

class CardDecoder:
    def __init__(self):
        self.card_types:list[str] = CardFields.card_types()
        self.card_supertypes:list[str] = CardFields.card_supertypes()
        self.all_subtypes:list[str] = CardFields.card_subtypes()
        self.color_identities:list[str] = CardFields.color_identities()
        self.tags:list[str] = CardFields.card_tags()

    def decode_to_string(self,  card_name: str, encoded_vector: np.array) -> str:
        idx = 0

        card_types = [self.card_types[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_types)]) if v == 1]
        idx += len(self.card_types)

        supertypes = [self.card_supertypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_supertypes)]) if v == 1]
        idx += len(self.card_supertypes)

        subtypes = [self.all_subtypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.all_subtypes)]) if v == 1]
        idx += len(self.all_subtypes)

        mana_cost = int(encoded_vector[idx])
        idx += 1

        color_identity = [self.color_identities[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.color_identities)]) if v == 1]
        idx += len(self.color_identities)

        rarity = self.int_to_rarity(encoded_vector[idx])
        idx += 1

        tags = set([self.tags[i].capitalize() for i, v in enumerate(encoded_vector[idx:idx+len(self.tags)]) if v == 1])

        return ("Name: " + str(card_name) + "\n Card Types: " + str(card_types) + "\n Card Supertypes: " + str(supertypes) + "\n Card Subtypes: " + str(subtypes) + \
                "\n Mana Cost: " + str(mana_cost) + "\n Color Identity: " + str(color_identity) + "\n Rarity: " + str(rarity) + "\n Tags: \n" + str(tags))

    def decode_to_dict(self,  card_name: str, encoded_vector: np.array) -> str:
        idx = 0

        card_types = [self.card_types[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_types)]) if v == 1]
        idx += len(self.card_types)

        supertypes = [self.card_supertypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_supertypes)]) if v == 1]
        idx += len(self.card_supertypes)

        subtypes = [self.all_subtypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.all_subtypes)]) if v == 1]
        idx += len(self.all_subtypes)

        mana_cost = int(encoded_vector[idx])
        idx += 1

        color_identity = [self.color_identities[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.color_identities)]) if v == 1]
        idx += len(self.color_identities)

        rarity = self.int_to_rarity(encoded_vector[idx])
        idx += 1

        tags = set([self.tags[i].capitalize() for i, v in enumerate(encoded_vector[idx:idx+len(self.tags)]) if v == 1])

        return ({"Name": str(card_name), "Types": str(card_types), "Supertypes": str(supertypes), "Subtypes": str(subtypes),\
                 "Mana Cost": str(mana_cost), "Color Identity": str(color_identity), "Rarity": str(rarity), "Tags": str(tags)})

    def int_to_rarity(self, rarity_int: int) -> str:
        rarity_map = {
            1: 'Common',
            2: 'Uncommon',
            3: 'Rare',
            4: 'Mythic',
            5: 'Timeshifted',
            6: 'Masterpiece',
            7: 'Special'
        }
        return rarity_map.get(rarity_int, 'Unknown')
