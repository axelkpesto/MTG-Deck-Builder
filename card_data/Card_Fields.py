from card_data.Card import Card
import re
from typing import Dict, Callable
import numpy as np

class CardFields(object):
    __tags_general = {
        'aggro': ['haste', 'attack each turn', 'additional combat'],
        'control': ['destroy target', 'exile target', 'return target', 'tap target', 'prevent all damage', 'flash', 'scry', 'can\'t activate', 'can\'t attack', 'can\'t block', 'countered', 'return each', 'destroy that', 'gain control', 'tap all', 'change the target', 'return all'],
        'combo': ['You win the game', 'Opponent loses the game', 'for each', 'without paying it\'s mana cost', 'storm', 'cast for {0}', 'The legend rule doesn\'t apply', 'without paying its mana cost', 'rather than paying its mana value', 'rather than paying it\'s mana value', 'rather than paying its mana cost', 'rather than paying it\'s mana cost'],
        'ramp': ['add mana', 'search your library for a land', 'landfall', 'treasure', 'search your library for a basic land', 'basic lands', 'basic land'],
        'card_draw': ['draw a card', 'blood token', 'you may play that card this turn'],
        'party': ['Cleric', 'Rogue', 'Warrior', 'Wizard', 'party'],
        'token': ['token', 'populate', 'copy', 'token creature', 'tokens you control', 'creature token'],
        'life_gain': ['gain life', 'whenever you gain life', 'life total', 'lifelink', 'food token', 'you gain that much life'],
        'life_drain':['lose life', 'loses that much life', 'paying life', 'pays life', 'pay life'],
        'mill': ['put into your graveyard', 'mill', 'from the top of their library', 'library into graveyard'],
        'discard': ['discard a card', 'discard cards', 'each player discards', 'from their hand', 'discard', 'madness', 'blood'],
        'reanimaton': ['return target creature card', 'graveyard to the battlefield', 'reanimate', 'bring back', 'from your graveyard', 'regenerate'],
        'burn': ['damage to any target', 'damage to each opponent', 'damage to target creature', 'damage to target planeswalker', 'damage to any target', 'deals damage equal to'],
        'enchantment': ['enchant', 'aura', 'whenever you cast an enchantment', 'constellation', 'saga'],
        'equipment': ['equip', 'attach', 'whenever equipped', 'whenever you attach', 'when equipped'],
        'artifact': ['artifact', 'whenever you cast an artifact', 'metalcraft'],
        'planeswalker': ['planeswalker', 'loyalty', 'loyalty counter', 'proliforate'],
        'tribal': ['Elf', 'Goblin', 'Zombie', 'Vampire', 'Warrior', 'Merfolk', 'Soldier', 'Dragon', 'Angel', 'Wizard', 'Knight', 'Sliver'],
        'voltron': ['equip', 'attach', 'whenever equipped', 'whenever you attach', 'aura'],
        'stax': ['tap target', 'opponent can\'t untap', 'opponents can\'t draw', 'whenever an opponent', 'can\'t cast', 'can\'t activate', 'can\'t attack', 'prevent that damage', 'prevent damage', 'deals damage to you'],
        'storm': ['storm', 'copy this spell', 'copy that spell', 'copy target spell', 'magecraft', 'copy target instant', 'copy target sorcery', 'copy'],
        'graveyard': ['graveyard', 'return from your graveyard', 'mill', 'from your graveyard', 'unearth'],
        'sacrifice': ['sacrifice a creature', 'sacrifice a permanent', 'whenever you sacrifice', 'sacrifice'],
        'combat': ['combat damage to a player', 'combat damage to an opponent', 'vigilance', 'combat damage', 'attacks', 'first strike', 'double strike', 'target attacking', 'target blocking', 'whenever you attack', 'whenever  attacks', 'whenever  blocks', 'each combat', 'blocks', 'attacking', 'blocking', 'can\'t be blocked', 'deals damage to a player', 'flying', 'raid', 'infect', 'poison counter', 'proliferate', 'with infect', 'flanking', 'horemanship', 'menace', 'deathtouch', 'reach', 'trample'],
        'outlaw': ['Assassin', 'Mercenary', 'Pirate', 'Rogue', 'Warlock'],
        'removal':['Destroy target', 'Destroy all', 'Exile target', 'Exile all', 'Remove from the game', 'Removed from the game', 'Destroy'],
        'buff':['all tokens you control', 'all creatures you control', 'proliferate', 'creatures you control get', 'token creatures you control get'],
        'etb':['enters the battlefield', 'enters under your control', 'enter the battlefield', 'exile and return', 'flicker', 'blink', 're-enter the battlefield'],
        'library_control':['scry', 'on top of your library', 'card into your hand', 'on the top of your library', 'look at the top'],
        'extb':['whenever a creature dies', 'whenever another creature dies', 'whenever a creature is put into your graveyard', 'whenever this creature dies', 'when  dies', 'regenerate', 'would die', 'dies', 'put into a graveyard', ],
        'protection':['indestructible', 'prevent all damage', 'protection from', 'shroud', 'hexproof', 'you have hexproof', 'you have shroud', 'creatures you control gain indestructible', 'creatures you control gain hexproof', 'creatures you control gain shroud', 'target creature you control gains hexproof']
    }
    __tags_general = {tag: [phrase.lower() for phrase in phrases] for tag, phrases in __tags_general.items()}

    __tags_joint = {
        'control':[['counter', 'spell'],],
        'ramp':[['Untap', 'Land'],['Search your library', 'land'],['Add', 'mana'],['Add', '{', '}'],],
        'protection':[['prevent', 'damage'],],
        'etb':[['exile', 'return'], ['enters', 'battlefield']],
        'extb':[['dies'],['put into', 'graveyard'],['when', 'dies'],['graveyard', 'battlefield'], ['would', 'die']],
        'life_gain':[['gain', 'life']],
        'life_drain':[['lose', 'life'],['pay', 'life']],
        'stax':[['whenever','taps']],
        'buff':[['creatures', 'you control', 'get'],['you control', 'get'],],
        'token':[['token', 'creature'],['tokens', 'you control'],],
    }
    __tags_joint = {tag: [[p.lower() for p in pair] for pair in pairs] for tag, pairs in __tags_joint.items()}

    __tags_regex = {
        'aggro': [r'attack each turn', r'deal \d+ damage'],
        'combo': [r'cast for {0}'],
        'control': [r'counter target \w spell'],
        'ramp': [r'add {\w}', r'search your library for a (|[a-zA-Z]+) land'],
        'card_draw': [r'draw (\d+|[a-zA-Z]+) cards'],
        'life_gain': [r'gain \d life',r'gain \w life',r'gains \d life',r'gains \w life'],
        'life_drain': [r'loses \d life',r'loses \w life',r'pays \d life',r'pays \w life'],
        'burn': [r'deal \d+ damage', r'deals \d+ damage', r'deals \w damage'],
        'stax': [r'cost {\d+} more to cast'],
        'buff': [r'\+\d/\++\d',r'\+\w/\++\w',r'\+\w/\++\d',r'\+\d/\++\w',r'\-\d/\++\d',r'\-\w/\++\w',r'\-\w/\++\d',r'\-\d/\++\w',r'\+\d/\-+\d',r'\+\w/\-+\w',r'\+\w/\-+\d',r'\+\d/\-+\w'],
    }
    __tags_regex = {tag: [pat.lower() for pat in pats] for tag, pats in __tags_regex.items()}

    __subtype_tags = {
        'tribal': ['Elf', 'Goblin', 'Zombie', 'Vampire', 'Warrior', 'Merfolk', 'Soldier', 'Dragon', 'Angel', 'Wizard', 'Knight', 'Sliver'],
        'outlaw': ['Assassin', 'Mercenary', 'Pirate', 'Rogue', 'Warlock'],
        'party': ['Cleric', 'Rogue', 'Warrior', 'Wizard'],
        'enchantment': ['Aura', 'Background', 'Saga', 'Role', 'Shard', 'Cartouche', 'Case', 'Class', 'Curse', 'Rune', 'Shrine'],
        'equipment': ['Equipment', 'Fortification', 'Vehicle', 'Attraction', 'Contraption'],
        'artifact': ['Blood', 'Clue', 'Food', 'Gold', 'Incubator', 'Junk', 'Map', 'Powerstone', 'Treasure'],
        'planeswalker': ['Planeswalker'],
        'battle': ['Seige'],
        'voltron': ['Equipment', 'Vehicle', 'Aura', 'Background', 'Saga', 'Role', 'Shard', 'Cartouche', 'Case', 'Class', 'Curse', 'Rune'],
        'discard': ['blood'],
    }
    __subtype_tags = {tag: [s.lower() for s in subs] for tag, subs in __subtype_tags.items()}

    __card_types = [t.lower() for t in ['Enchantment', 'Land', 'Artifact', 'Creature', 'Instant', 'Sorcery', 'Planeswalker', 'Battle']]
    __card_supertypes = [t.lower() for t in['Legendary', 'Basic', 'Snow', 'World', 'Ongoing']]
    __creature_subtypes = [t.lower() for t in['Advisor', 'Aetherborn', 'Ally', 'Angel', 'Anteater', 'Antelope', 'Ape', 'Archer', 'Archon', 'Artificer', 'Assassin', 'Assembly-Worker', 'Atog', 'Aurochs', 'Avatar', 'Badger', 'Barbarian', 'Basilisk', 'Bat', 'Bear', 'Beast', 'Beeble', 'Berserker', 'Bird', 'Blinkmoth', 'Boar', 'Bringer', 'Brushwagg', 'Camarid', 'Camel', 'Caribou', 'Carrier', 'Cat', 'Centaur', 'Cephalid', 'Chimera', 'Citizen', 'Cleric', 'Cockatrice', \
    'Construct', 'Coward', 'Crab', 'Crocodile', 'Cyclops', 'Dauthi', 'Demon', 'Deserter', 'Devil', 'Dinosaur', 'Djinn', 'Dragon', 'Drake', 'Dreadnought', 'Drone', 'Druid', 'Dryad', 'Dwarf', 'Efreet', 'Elder', 'Eldrazi', 'Elemental', 'Elephant', 'Elf', 'Elk', 'Eye', 'Faerie', 'Ferret', 'Fish', 'Flagbearer', 'Fox', 'Frog', 'Fungus', 'Gargoyle', 'Germ', 'Giant', 'Gnome', 'Goat', 'Goblin', 'God', \
    'Golem', 'Gorgon', 'Graveborn', 'Gremlin', 'Griffin', 'Hag', 'Harpy', 'Hellion', 'Hippo', 'Hippogriff', 'Hormarid', 'Homunculus', 'Horror', 'Horse', 'Hound', 'Human', 'Hydra', 'Hyena', 'Illusion', 'Imp', 'Incarnation', 'Insect', 'Jellyfish', 'Juggernaut', 'Kavu', 'Kirin', 'Kithkin', 'Knight', 'Kobold', 'Kor', 'Kraken', 'Lamia', 'Lammasu', 'Leech', 'Leviathan', 'Lhurgoyf', 'Licid', 'Lizard', 'Manticore', 'Masticore', \
    'Mercenary', 'Merfolk', 'Metathran', 'Minion', 'Minotaur', 'Mole', 'Monger', 'Mongoose', 'Monk', 'Moonfolk', 'Mutant', 'Myr', 'Mystic', 'Naga', 'Nautilus', 'Necron', 'Nephilim', 'Nightmare', 'Nightstalker', 'Ninja', 'Noggle', 'Nomad', 'Nymph', 'Octopus', 'Ogre', 'Ooze', 'Orb', 'Orc', 'Orgg', 'Ouphe', 'Ox', 'Oyster', 'Pegasus', 'Pentavite', 'Pest', 'Phelddagrif', 'Phoenix', 'Pincher', 'Pirate', \
    'Plant', 'Praetor', 'Prism', 'Processor', 'Rabbit', 'Rat', 'Rebel', 'Reflection', 'Rhino', 'Rigger', 'Rogue', 'Sable', 'Salamander', 'Samurai', 'Sand', 'Saproling', 'Satyr', 'Scarecrow', 'Scion', 'Scorpion', 'Scout', 'Serf', 'Serpent', 'Shade', 'Shaman', 'Shapeshifter', 'Sheep', 'Siren', 'Skeleton', 'Slith', 'Sliver', 'Slug', 'Snake', 'Soldier', 'Soltari', 'Spawn', 'Specter', 'Spellshaper', 'Sphinx', 'Spider', 'Spike',\
    'Spirit', 'Splinter', 'Sponge', 'Squid', 'Squirrel', 'Starfish', 'Surrakar', 'Survivor', 'Tetravite', 'Thalakos', 'Thopter', 'Thrull', 'Treefolk', 'Triskelavite', 'Troll', 'Turtle', 'Unicorn', 'Vampire', 'Vedalken', 'Viashino', 'Volver', 'Wall', 'Warrior', 'Weird', 'Werewolf', 'Whale', 'Wizard', 'Wolf', 'Wolverine', 'Wombat', 'Worm', 'Wraith', 'Wurm', 'Yeti', 'Zombie', 'Zubera']]
    __artifact_subtypes = [t.lower() for t in['Blood', 'Clue', 'Food', 'Gold', 'Incubator', 'Junk', 'Map', 'Powerstone', 'Treasure', 'Equipment', 'Fortification', 'Vehicle', 'Attraction', 'Contraption']]
    __battle_subtypes = [t.lower() for t in['Seige']]
    __enchantment_subtypes = [t.lower() for t in['Aura', 'Background', 'Saga', 'Role', 'Shard', 'Cartouche', 'Case', 'Class', 'Curse', 'Rune', 'Shrine']]
    __land_subtypes = [t.lower() for t in['Plains', 'Forest', 'Mountain', 'Island', 'Swamp', 'Cave', 'Desert', 'Gate', 'Lair', 'Locus', 'Mine', 'Power-Plant', 'Sphere', 'Tower', 'Urza']]
    __spell_subtypes = [t.lower() for t in['Adventure', 'Arcane', 'Chorus', 'Lesson', 'Trap']]
    __all_subtypes = __creature_subtypes + __artifact_subtypes + __battle_subtypes + __enchantment_subtypes + __land_subtypes + __spell_subtypes
    __color_identities = ['W', 'G', 'U', 'B', 'R', 'C']
    __rarities = ['common', 'uncommon', 'rare', 'mythic', 'timeshifted', 'masterpiece', 'special']
    
    __tags_general_sets = {k: set(v) for k, v in __tags_general.items()}
    __tags_joint_sets   = {k: [set(pair) for pair in v] for k, v in __tags_joint.items()}
    __tags_regex_sets   = {k: set(v) for k, v in __tags_regex.items()}
    __tags_subtype_sets = {k: set(v) for k, v in __subtype_tags.items()}
    __card_types_set = set(__card_types)
    __card_supertypes_set = set(__card_supertypes)
    __creature_subtypes_set = set(__creature_subtypes)
    __artifact_subtypes_set = set(__artifact_subtypes)
    __battle_subtypes_set = set(__battle_subtypes)
    __enchantment_subtypes_set = set(__enchantment_subtypes)
    __land_subtypes_set = set(__land_subtypes)
    __spell_subtypes_set = set(__spell_subtypes)
    __color_identities_set = set(__color_identities)

    __all_tags = set(__tags_general.keys()).union(set(__tags_joint.keys())).union(set(__tags_regex.keys())).union(set(__subtype_tags.keys()))

    __BASIC_LANDS = {"Plains","Island","Swamp","Mountain","Forest","Wastes"}

    __COLOR_BASIC_LAND_MAP = {
        'W': 'Plains',
        'U': 'Island',
        'B': 'Swamp',
        'R': 'Mountain',
        'G': 'Forest',
        'C': 'Wastes'
    }

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
    def general_tags() -> Dict[str, list[str]]: return CardFields.__tags_general

    @staticmethod
    def joint_tags() -> Dict[str, list[list[str]]]: return CardFields.__tags_joint

    @staticmethod
    def regex_tags() -> Dict[str, list[str]]: return CardFields.__tags_regex
    
    @staticmethod
    def subtype_tags() -> Dict[str, list[str]]: return CardFields.__subtype_tags
    
    @staticmethod
    def card_types_set() -> set[str]: return CardFields.__card_types_set

    @staticmethod
    def card_supertypes_set() -> set[str]: return CardFields.__card_supertypes_set

    @staticmethod
    def creature_subtypes_set() -> set[str]: return CardFields.__creature_subtypes_set

    @staticmethod
    def artifact_subtypes_set() -> set[str]: return CardFields.__artifact_subtypes_set

    @staticmethod
    def battle_subtypes_set() -> set[str]: return CardFields.__battle_subtypes_set

    @staticmethod
    def enchantment_subtypes_set() -> set[str]: return CardFields.__enchantment_subtypes_set

    @staticmethod
    def land_subtypes_set() -> set[str]: return CardFields.__land_subtypes_set

    @staticmethod
    def spell_subtypes_set() -> set[str]: return CardFields.__spell_subtypes_set

    @staticmethod
    def color_identities_set() -> set[str]: return CardFields.__color_identities_set

    @staticmethod
    def tags_general_sets() -> Dict[str, set[str]]: return CardFields.__tags_general_sets

    @staticmethod
    def tags_joint_sets() -> Dict[str, list[set[str]]]: return CardFields.__tags_joint_sets

    @staticmethod
    def tags_regex_sets() -> Dict[str, set[str]]: return CardFields.__tags_regex_sets

    @staticmethod
    def tags_subtype_sets() -> Dict[str, set[str]]: return CardFields.__tags_subtype_sets

    @staticmethod
    def rarities() -> list[str]: return sorted(CardFields.__rarities)

    @staticmethod
    def basic_lands() -> set[str]: return CardFields.__BASIC_LANDS

    @staticmethod
    def color_basic_land_map() -> Dict[str, str]: return CardFields.__COLOR_BASIC_LAND_MAP

    @staticmethod
    def rarity_map() -> Dict[int, str]: return dict({(i+1, CardFields.__rarities[i]) for i in range(len(CardFields.__rarities))})

    @staticmethod
    def rarity_to_index() -> Dict[str, int]: return dict({(CardFields.__rarities[i], i+1) for i in range(len(CardFields.__rarities))})

    @staticmethod
    def all_tags() -> set[str]: return CardFields.__all_tags

    @staticmethod
    def pred_dim(dim: int) -> Callable[[str, np.ndarray], bool]:
        return lambda _, v: v.shape[-1] == dim

    @staticmethod
    def pred_regex(pattern: str, flags: int = re.IGNORECASE) -> Callable[[str, np.ndarray], bool]:
        rx = re.compile(pattern, flags)
        return lambda n, _: rx.search(n) is not None

    @staticmethod
    def pred_not(inner: Callable[[str, np.ndarray], bool]) -> Callable[[str, np.ndarray], bool]:
        return lambda n, v: not inner(n, v)

    @staticmethod
    def pred_all(*preds: Callable[[str, np.ndarray], bool]) -> Callable[[str, np.ndarray], bool]:
        return lambda n, v: all(p(n, v) for p in preds)

    @staticmethod
    def pred_any(*preds: Callable[[str, np.ndarray], bool]) -> Callable[[str, np.ndarray], bool]:
        return lambda n, v: any(p(n, v) for p in preds)

    @staticmethod
    def tag_text(name: str, text: str) -> set[str]:
        text = text.replace(name, '').lower()
        tags = set()

        for tag, phrases in CardFields.__tags_general_sets.items():
            if tag in tags: continue
            if any(p.lower() in text for p in phrases):
                tags.add(tag)

        for tag, pairs in CardFields.__tags_joint_sets.items():
            if tag in tags: continue
            for pair in pairs:
                if all(p.lower() in text for p in pair):
                    tags.add(tag)

        for tag, patterns in CardFields.__tags_regex_sets.items():
            if tag in tags: continue
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    tags.add(tag)
        return tags

    @staticmethod
    def tag_subtypes(subtypes: list[str] | None) -> set[str]:
        if not subtypes:
            return set()

        card_tags: set[str] = set()
        for tag, subtype_set in CardFields.tags_subtype_sets().items():
            if any(st.lower() in subtype_set for st in subtypes):
                card_tags.add(tag)
        return card_tags
    
    @staticmethod
    def tag_card(card: Card) -> set[str]:
        return set(card.card_types).union(CardFields.tag_text(card.card_name, card.text).union(CardFields.tag_subtypes(card.card_subtypes)))
    
    @staticmethod
    def parse_mtgjson_card(card: dict) -> Card:
        card_kwargs = {
            'commander_legal': 'commander' in card['legalities'] and card['legalities']['commander']=="Legal",
            'card_name': card.get('name', ''),
            'card_types': [str(s).lower() for s in card.get('types', '')],
            'card_supertypes': [str(s).lower() for s in card.get('supertypes', '')],
            'card_subtypes': [str(s).lower() for s in card.get('subtypes', '')],
            'mana_cost': card.get('manaValue', -1),
            'mana_cost_exp': card.get('manaCost', ''),
            'color_identity': [str(x).upper() for x in sorted([x for x in list(set(card.get('manaCost', ''))) if str(x).isalpha()])],
            'defense': card.get('defense', ''),
            'rarity': card.get('rarity', ''),
            'text': card.get('text', ''),
            'rank': card.get('edhrecRank', ''),
            'power': card.get('power', ''),
            'toughness': card.get('toughness', ''),
            'loyalty': card.get('loyalty', ''),
            'id': card.get('identifiers','').get('scryfallId', ''),
        }
        return Card(**card_kwargs)
    
    @staticmethod
    def parse_moxfieldapi_card(card: dict) -> Card:
        split = card.get('type_line', 'None').split(' — ')
        subsplit = [item.lower() for sublist in [s.split(' ') for s in split] for item in sublist]
        card_kwargs = {
            'commander_legal': 'commander' in card['legalities'] and card['legalities']['commander']=="Legal",
            'card_name': card.get('name', ''),
            'card_types': [s for s in subsplit if s in CardFields.card_types()],
            'card_supertypes': [s for s in subsplit if s in CardFields.card_supertypes()],
            'card_subtypes': [s for s in subsplit if s in CardFields.card_subtypes()],
            'mana_cost': card.get('cmc', '-1'),
            'mana_cost_exp': card.get('mana_cost', ''),
            'color_identity': card.get('color_identity', []),
            'defense': card.get('defense', ''),
            'rarity': card.get('rarity', ''),
            'text': card.get('oracle_text', ''),
            'rank': card.get('edhrec_rank', ''),
            'power': card.get('power', ''),
            'toughness': card.get('toughness', ''),
            'loyalty': card.get('loyalty', ''),
            'id': card.get('scryfall_id', ''),
        }
        return Card(**card_kwargs)
    
    @staticmethod
    def parse_moxfield_group(deck: dict, key: str):
        group = deck.get(key)
        if not group:
            return []

        entries = group if isinstance(group, list) else group.values()
        out = []
        for e in entries:
            if isinstance(e, dict):
                c = e.get("card")
                if isinstance(c, dict):
                    out.append(CardFields.parse_moxfieldapi_card(c))
        return out
