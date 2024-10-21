import numpy as np
import re

class Card(object):
    def __init__(self, cd:dict) -> None:
        if cd is None or {}: return
        self.card_name: str = str(cd['name']) if 'name' in cd else None #Name of the Card. None if property doesn't exist.
        self.card_types: list[str] =  [str(x).lower() for x in cd['types']]  if 'types' in cd else None #Card Types (EX: CREATURE) of the Card. None if property doesn't exist.
        self.card_supertypes: str = cd['supertypes'] if 'supertypes' in cd else None #Supertypes (EX: LEGENDARY) of the Card. None if property doesn't exist.
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
        self.id: str = cd['identifiers']['multiverseId'] if 'identifiers' in cd and 'multiverseId' in cd['identifiers'] else None
        self.tags: list[str] = [x.lower() for x in list(self.tag_text(self.text).union(self.tag_subtypes(str(self.card_subtypes))))] #Tagging cards for data analysis

    def tag_text(self, text: str) -> set[str]:
        if text==None or text=="": return set()
        text = text.replace(self.card_name,'').lower()
        tags_general = {
            'aggro': ['haste', 'attack each turn',],
            'control': ['destroy target', 'exile target', 'return target', 'tap target', 'prevent all damage', 'flash', 'scry','can\'t activate','can\'t attack','can\'t block','countered', 'return each', 'destroy that','gain control','tap all'],
            'combo': ['You win the game','Opponent loses the game','for each','without paying it\'s mana cost', 'storm','cast for {0}','The legend rule doesn\'t apply','without paying its mana cost'],
            'ramp': ['add mana', 'search your library for a land', 'landfall','treasure', 'search your library for a basic land','basic lands','basic land'],
            'card_draw': ['draw a card','blood token','you may play that card this turn'],
            'party': ['Cleric','Rogue','Warrio','Wizard'],
            'token': ['token', 'populate', 'copy', 'token creature', 'tokens you control'],
            'life_gain': ['gain life', 'life equal to', 'whenever you gain life', 'life total', 'lifelink','food token','you gain that much life'],
            'mill': ['put into your graveyard', 'mill', 'from the top of their library', 'library into graveyard'],
            'discard': ['discard a card', 'discard cards', 'each player discards', 'from their hand'],
            'reanimato': ['return target creature card', 'graveyard to the battlefield', 'reanimate', 'bring back', 'from your graveyard'],
            'burn': ['damage to any target', 'damage to each opponent','damage to target creature','damage to target planeswalke','damage to any target'],
            'enchantment': ['enchant', 'aura', 'whenever you cast an enchantment', 'constellation','saga'],
            'equipment': ['equip', 'attach', 'whenever equipped', 'whenever you attach'],
            'artifact': ['artifact', 'whenever you cast an artifact', 'metalcraft'],
            'planeswalke': ['planeswalke', 'loyalty', 'loyalty counte'],
            'tribal': ['Elf', 'Goblin', 'Zombie', 'Vampire', 'Warrio', 'Merfolk', 'Soldie', 'Dragon', 'Angel', 'Wizard', 'Knight', 'Slive'],
            'flicke': ['exile and return', 'flicke', 'blink', 're-enter the battlefield'],
            'voltron': ['equip', 'attach', 'whenever equipped', 'whenever you attach', 'aura'],
            'infect': ['infect', 'poison counte', 'proliferate'],
            'stax': ['tap target', 'opponent can\'t untap', 'opponents can\'t draw', 'whenever an opponent','can\'t cast','can\'t activate','can\'t attack','prevent that damage','prevent damage','deals damage to you'],
            'storm': ['storm', 'copy this spell', 'copy that spell', 'copy target spell','magecraft'],
            'graveyard': ['graveyard', 'return from your graveyard', 'mill','from your graveyard','unearth'],
            'sacrifice': ['sacrifice a creature', 'sacrifice a permanent', 'whenever you sacrifice'],
            'combat': ['combat damage to a playe', 'combat damage to an opponent', 'vigilance', 'combat damage', 'attacks','first strike', 'double strike','target attacking', 'target blocking', 'whenever you attack', 'whenever  attacks', 'whenever  blocks', 'each combat', 'blocks', 'attacking', 'blocking', 'can\'t be blocked', 'deals damage to a playe', 'Flying'],
            'outlaw': ['Assassin', 'Mercenary', 'Pirate', 'Rogue','Warlock'],
            'removal':['Destroy target','Destroy all','Exile target', 'Exile all','Remove from the game', 'Removed from the game'],
            'discard':['discard','madness','blood'],
            'etb':['enters the battlefield','enters under your control', 'enter the battlefield'],
            'library_control':['scry','on top of your library','card into your hand','on the top of your library', 'look at the top'],
            'extb':['whenever a creature dies','whenever another creature dies', 'whenever a creature is put into your graveyard', 'whenever this creature dies', 'when  dies', 'regenerate'],
            'protection':['indestructible','prevent all damage','protection from', 'shroud', 'hexproof', 'you have hexproof', 'you have shroud', 'creatures you control gain indestructible', 'creatures you control gain hexproof', 'creatures you control gain shroud', 'target creature you control gains hexproof']
        }

        tags_joint = {
            'control':[['counter','spell'],],
            'ramp':[['Untap','Land'],['Search your library','land'],['Add','mana'],['Add','{','}'],],
            'protection':[['prevent','damage'],]
        }

        tags_regex = {
            'aggro': [r'attack each turn', r'deal \d+ damage'],
            'combo': [r'cast for {0}'],
            'control': [r'counter target \w spell',],
            'ramp': [r'add {\w}', r'search your library for a (|[a-zA-Z]+) land',],
            'card_draw': [r'draw (\d+|[a-zA-Z]+) cards',],
            'life_gain': [r'gain \d life',],
            'burn': [r'deal \d+ damage', r'deals \d+ damage', r'deals \w damage'],
            'stax': [r'cost {\d+} more to cast'],
            'buff': [r'\+\d/\++\d',r'\+\w/\++\w',r'\+\w/\++\d',r'\+\d/\++\w',r'\-\d/\++\d',r'\-\w/\++\w',r'\-\w/\++\d',r'\-\d/\++\w',r'\+\d/\-+\d',r'\+\w/\-+\w',r'\+\w/\-+\d',r'\+\d/\-+\w'],
        }

        card_tags: set[str] = set()

        for tag, patterns in tags_general.items():
            if tag in card_tags: continue
            for pattern in patterns:
                if pattern in text:
                    card_tags.add(str(tag).lower())
                    break

        for tag, patterns in tags_joint.items():
          if tag in card_tags: continue
          for pattern in patterns:
              add = True
              for i in range(len(pattern)):
                  if pattern[i].lower() not in str(text):
                      add = False
              if add: 
                card_tags.add(str(tag).lower())
                break

        for tag, patterns in tags_regex.items():
          if tag in card_tags: continue
          for pattern in patterns:
              if re.search(pattern, str(text), re.IGNORECASE):
                  card_tags.add(str(tag).lower())
                  break
        
        return card_tags

    def tag_subtypes(self, text: list[str]) -> set[str]:
        if text==None or text==[]: return set()
        card_tags:set[str] = set()
        subtype_tags = {
            'tribal': ['Elf', 'Goblin', 'Zombie', 'Vampire', 'Warrior', 'Merfolk', 'Soldier', 'Dragon', 'Angel', 'Wizard', 'Knight', 'Sliver'],
            'outlaw': ['Assassin', 'Mercenary', 'Pirate', 'Rogue','Warlock'],
            'party': ['Cleric','Rogue','Warrior','Wizard']
        }

        for tag, patterns in subtype_tags.items():
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
    
class CardEncoder(object):
    def __init__(self):
        self.card_types:list[str] = sorted(['Enchantment', 'Land', 'Artifact', 'Creature', 'Instant', 'Sorcery', 'Planeswalker', 'Battle'])
        self.card_supertypes:list[str] = sorted(['Legendary', 'Basic'])

        creature_subtypes:list[str] = sorted(['Advisor', 'Aetherborn', 'Ally', 'Angel', 'Anteater', 'Antelope', 'Ape', 'Archer', 'Archon', 'Artificer', 'Assassin', 'Assembly-Worker', 'Atog', 'Aurochs', 'Avatar', 'Badger', 'Barbarian', 'Basilisk', 'Bat', 'Bear', 'Beast', 'Beeble', 'Berserker', 'Bird', 'Blinkmoth', 'Boar', 'Bringer', 'Brushwagg', 'Camarid', 'Camel', 'Caribou', 'Carrier', 'Cat', 'Centaur', 'Cephalid', 'Chimera', 'Citizen', 'Cleric', 'Cockatrice', 'Construct', 'Coward', 'Crab', 'Crocodile', 'Cyclops', 'Dauthi', 'Demon', 'Deserter', 'Devil', 'Dinosaur', 'Djinn', 'Dragon', 'Drake', 'Dreadnought', 'Drone', 'Druid', 'Dryad', 'Dwarf', 'Efreet', 'Elder', 'Eldrazi', 'Elemental', 'Elephant', 'Elf', 'Elk', 'Eye', 'Faerie', 'Ferret', 'Fish', 'Flagbearer', 'Fox', 'Frog', 'Fungus', 'Gargoyle', 'Germ', 'Giant', 'Gnome', 'Goat', 'Goblin', 'God', 'Golem', 'Gorgon', 'Graveborn', 'Gremlin', 'Griffin', 'Hag', 'Harpy', 'Hellion', 'Hippo', 'Hippogriff', 'Hormarid', 'Homunculus', 'Horror', 'Horse', 'Hound', 'Human', 'Hydra', 'Hyena', 'Illusion', 'Imp', 'Incarnation', 'Insect', 'Jellyfish', 'Juggernaut', 'Kavu', 'Kirin', 'Kithkin', 'Knight', 'Kobold', 'Kor', 'Kraken', 'Lamia', 'Lammasu', 'Leech', 'Leviathan', 'Lhurgoyf', 'Licid', 'Lizard', 'Manticore', 'Masticore', 'Mercenary', 'Merfolk', 'Metathran', 'Minion', 'Minotaur', 'Mole', 'Monger', 'Mongoose', 'Monk', 'Moonfolk', 'Mutant', 'Myr', 'Mystic', 'Naga', 'Nautilus', 'Nephilim', 'Nightmare', 'Nightstalker', 'Ninja', 'Noggle', 'Nomad', 'Nymph', 'Octopus', 'Ogre', 'Ooze', 'Orb', 'Orc', 'Orgg', 'Ouphe', 'Ox', 'Oyster', 'Pegasus', 'Pentavite', 'Pest', 'Phelddagrif', 'Phoenix', 'Pincher', 'Pirate', 'Plant', 'Praetor', 'Prism', 'Processor', 'Rabbit', 'Rat', 'Rebel', 'Reflection', 'Rhino', 'Rigger', 'Rogue', 'Sable', 'Salamander', 'Samurai', 'Sand', 'Saproling', 'Satyr', 'Scarecrow', 'Scion', 'Scorpion', 'Scout', 'Serf', 'Serpent', 'Shade', 'Shaman', 'Shapeshifter', 'Sheep', 'Siren', 'Skeleton', 'Slith', 'Sliver', 'Slug', 'Snake', 'Soldier', 'Soltari', 'Spawn', 'Specter', 'Spellshaper', 'Sphinx', 'Spider', 'Spike', 'Spirit', 'Splinter', 'Sponge', 'Squid', 'Squirrel', 'Starfish', 'Surrakar', 'Survivor', 'Tetravite', 'Thalakos', 'Thopter', 'Thrull', 'Treefolk', 'Triskelavite', 'Troll', 'Turtle', 'Unicorn', 'Vampire', 'Vedalken', 'Viashino', 'Volver', 'Wall', 'Warrior', 'Weird', 'Werewolf', 'Whale', 'Wizard', 'Wolf', 'Wolverine', 'Wombat', 'Worm', 'Wraith', 'Wurm', 'Yeti', 'Zombie', 'Zubera'])
        artifact_subtypes:list[str] = sorted(['Blood', 'Clue', 'Food', 'Gold', 'Incubator', 'Junk', 'Map', 'Powerstone', 'Treasure', 'Equipment', 'Fortification', 'Vehicle', 'Attraction', 'Contraption'])
        battle_subtypes:list[str] = sorted(['Seige'])
        enchantment_subtypes:list[str] = sorted(['Aura', 'Background', 'Saga', 'Role', 'Shard', 'Cartouche', 'Case', 'Class', 'Curse', 'Rune', 'Shrine'])
        land_subtypes:list[str] = sorted(['Plains', 'Forest', 'Mountain', 'Island', 'Swamp', 'Cave', 'Desert', 'Gate', 'Lair', 'Locus', 'Mine', 'Power-Plant', 'Sphere', 'Tower', 'Urza'])
        spell_subtypes:list[str] = sorted(['Adventure', 'Arcane', 'Chorus', 'Lesson', 'Trap'])

        self.large_subtypes:list[str] = creature_subtypes + artifact_subtypes + battle_subtypes + enchantment_subtypes + land_subtypes + spell_subtypes
        self.color_identities:list[str] = sorted(['W', 'G', 'U', 'B', 'R', 'C'])
        self.tags:list[str] = sorted(['Aggro','Control','Combo','Ramp','Card_Draw','Party','Token','Life_Gain','Mill','Discard','Reanimator','Burn','Enchantment','Equipment','Artifact','Planeswalker','Tribal','Flicker','Voltron','Infect','Stax','Storm','Graveyard','Sacrifice','Combat','Buff','Outlaw','Removal','Discard','Etb','Library_Control','Extb','Protection'])
    
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

        cd = [0]*len(self.large_subtypes)
        for i in range(len(self.large_subtypes)):
            if self.large_subtypes[i].lower() in crd.card_subtypes:
                cd[i] = 1
        ret += cd

        ret += [int(crd.mana_cost)]
        
        cd = [0]*len(self.color_identities)
        if crd.color_identity == [] or crd.color_identity==None:
            if crd.mana_cost>0 or crd.mana_cost==None:
                cd[-1] = 1
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
        self.card_types = sorted(['Enchantment', 'Land', 'Artifact', 'Creature', 'Instant', 'Sorcery', 'Planeswalker', 'Battle'])
        self.card_supertypes = sorted(['Legendary', 'Basic'])

        creature_subtypes = sorted(['Advisor', 'Aetherborn', 'Ally', 'Angel', 'Anteater', 'Antelope', 'Ape', 'Archer', 'Archon', 'Artificer', 'Assassin', 'Assembly-Worker', 'Atog', 'Aurochs', 'Avatar', 'Badger', 'Barbarian', 'Basilisk', 'Bat', 'Bear', 'Beast', 'Beeble', 'Berserker', 'Bird', 'Blinkmoth', 'Boar', 'Bringer', 'Brushwagg', 'Camarid', 'Camel', 'Caribou', 'Carrier', 'Cat', 'Centaur', 'Cephalid', 'Chimera', 'Citizen', 'Cleric', 'Cockatrice', 'Construct', 'Coward', 'Crab', 'Crocodile', 'Cyclops', 'Dauthi', 'Demon', 'Deserter', 'Devil', 'Dinosaur', 'Djinn', 'Dragon', 'Drake', 'Dreadnought', 'Drone', 'Druid', 'Dryad', 'Dwarf', 'Efreet', 'Elder', 'Eldrazi', 'Elemental', 'Elephant', 'Elf', 'Elk', 'Eye', 'Faerie', 'Ferret', 'Fish', 'Flagbearer', 'Fox', 'Frog', 'Fungus', 'Gargoyle', 'Germ', 'Giant', 'Gnome', 'Goat', 'Goblin', 'God', 'Golem', 'Gorgon', 'Graveborn', 'Gremlin', 'Griffin', 'Hag', 'Harpy', 'Hellion', 'Hippo', 'Hippogriff', 'Hormarid', 'Homunculus', 'Horror', 'Horse', 'Hound', 'Human', 'Hydra', 'Hyena', 'Illusion', 'Imp', 'Incarnation', 'Insect', 'Jellyfish', 'Juggernaut', 'Kavu', 'Kirin', 'Kithkin', 'Knight', 'Kobold', 'Kor', 'Kraken', 'Lamia', 'Lammasu', 'Leech', 'Leviathan', 'Lhurgoyf', 'Licid', 'Lizard', 'Manticore', 'Masticore', 'Mercenary', 'Merfolk', 'Metathran', 'Minion', 'Minotaur', 'Mole', 'Monger', 'Mongoose', 'Monk', 'Moonfolk', 'Mutant', 'Myr', 'Mystic', 'Naga', 'Nautilus', 'Nephilim', 'Nightmare', 'Nightstalker', 'Ninja', 'Noggle', 'Nomad', 'Nymph', 'Octopus', 'Ogre', 'Ooze', 'Orb', 'Orc', 'Orgg', 'Ouphe', 'Ox', 'Oyster', 'Pegasus', 'Pentavite', 'Pest', 'Phelddagrif', 'Phoenix', 'Pincher', 'Pirate', 'Plant', 'Praetor', 'Prism', 'Processor', 'Rabbit', 'Rat', 'Rebel', 'Reflection', 'Rhino', 'Rigger', 'Rogue', 'Sable', 'Salamander', 'Samurai', 'Sand', 'Saproling', 'Satyr', 'Scarecrow', 'Scion', 'Scorpion', 'Scout', 'Serf', 'Serpent', 'Shade', 'Shaman', 'Shapeshifter', 'Sheep', 'Siren', 'Skeleton', 'Slith', 'Sliver', 'Slug', 'Snake', 'Soldier', 'Soltari', 'Spawn', 'Specter', 'Spellshaper', 'Sphinx', 'Spider', 'Spike', 'Spirit', 'Splinter', 'Sponge', 'Squid', 'Squirrel', 'Starfish', 'Surrakar', 'Survivor', 'Tetravite', 'Thalakos', 'Thopter', 'Thrull', 'Treefolk', 'Triskelavite', 'Troll', 'Turtle', 'Unicorn', 'Vampire', 'Vedalken', 'Viashino', 'Volver', 'Wall', 'Warrior', 'Weird', 'Werewolf', 'Whale', 'Wizard', 'Wolf', 'Wolverine', 'Wombat', 'Worm', 'Wraith', 'Wurm', 'Yeti', 'Zombie', 'Zubera'])
        artifact_subtypes = sorted(['Blood', 'Clue', 'Food', 'Gold', 'Incubator', 'Junk', 'Map', 'Powerstone', 'Treasure', 'Equipment', 'Fortification', 'Vehicle', 'Attraction', 'Contraption'])
        battle_subtypes = sorted(['Seige'])
        enchantment_subtypes = sorted(['Aura', 'Background', 'Saga', 'Role', 'Shard', 'Cartouche', 'Case', 'Class', 'Curse', 'Rune', 'Shrine'])
        land_subtypes = sorted(['Plains', 'Forest', 'Mountain', 'Island', 'Swamp', 'Cave', 'Desert', 'Gate', 'Lair', 'Locus', 'Mine', 'Power-Plant', 'Sphere', 'Tower', 'Urza'])
        spell_subtypes = sorted(['Adventure', 'Arcane', 'Chorus', 'Lesson', 'Trap'])

        self.large_subtypes = creature_subtypes + artifact_subtypes + battle_subtypes + enchantment_subtypes + land_subtypes + spell_subtypes
        self.color_identities = sorted(['W', 'G', 'U', 'B', 'R', 'C'])
        self.tags = sorted(['Aggro','Control','Combo','Ramp','Card_Draw','Party','Token','Life_Gain','Mill','Discard','Reanimator','Burn','Enchantment','Equipment','Artifact','Planeswalker','Tribal','Flicker','Voltron','Infect','Stax','Storm','Graveyard','Sacrifice','Combat','Buff','Outlaw','Removal','Discard','Etb','Library_Control','Extb','Protection'])

    def decode_to_string(self,  card_name: str, encoded_vector: np.array) -> str:
        idx = 0

        card_types = [self.card_types[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_types)]) if v == 1]
        idx += len(self.card_types)

        supertypes = [self.card_supertypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_supertypes)]) if v == 1]
        idx += len(self.card_supertypes)

        subtypes = [self.large_subtypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.large_subtypes)]) if v == 1]
        idx += len(self.large_subtypes)

        mana_cost = int(encoded_vector[idx])
        idx += 1

        color_identity = [self.color_identities[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.color_identities)]) if v == 1]
        idx += len(self.color_identities)

        rarity = self.int_to_rarity(encoded_vector[idx])
        idx += 1

        tags = [self.tags[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.tags)]) if v == 1]


        return ("Name: " + str(card_name) + "\n Card Types: " + str(card_types) + "\n Card Supertypes: " + str(supertypes) + "\n Card Subtypes: " + str(subtypes) + \
                "\n Mana Cost: " + str(mana_cost) + "\n Color Identity: " + str(color_identity) + "\n Rarity: " + str(rarity) + "\n Tags: \n" + str(tags))

    def decode_to_dict(self,  card_name: str, encoded_vector: np.array) -> str:
        idx = 0

        card_types = [self.card_types[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_types)]) if v == 1]
        idx += len(self.card_types)

        supertypes = [self.card_supertypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.card_supertypes)]) if v == 1]
        idx += len(self.card_supertypes)

        subtypes = [self.large_subtypes[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.large_subtypes)]) if v == 1]
        idx += len(self.large_subtypes)

        mana_cost = int(encoded_vector[idx])
        idx += 1

        color_identity = [self.color_identities[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.color_identities)]) if v == 1]
        idx += len(self.color_identities)

        rarity = self.int_to_rarity(encoded_vector[idx])
        idx += 1

        tags = [self.tags[i] for i, v in enumerate(encoded_vector[idx:idx+len(self.tags)]) if v == 1]

    
        return ({"Name":str(card_name),"Types":str(card_types),"Supertypes":str(supertypes),"Subtypes":str(subtypes),\
                 "Mana Cost":str(mana_cost),"Color Identity":str(color_identity),"Rarity":str(rarity),"Tags":str(tags)})

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
        