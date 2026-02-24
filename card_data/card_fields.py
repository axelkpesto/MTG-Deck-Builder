"""Canonical card field definitions, tag rules, and parsing helpers."""

import re
from typing import Dict, Callable

import numpy as np

from card_data.card import Card

class CardFields:
    """Static tables and helper methods for card parsing and tagging."""
    __tags_general = {
        'aggro': ['haste', 'attack each turn', 'must attack', 'additional combat', 'extra combat', 'whenever you attack', 'whenever this creature attacks', 'combat phase', 'double strike', 'first strike', 'menace', 'trample', 'can\'t be blocked', 'deals combat damage', 'after this phase', 'until end of combat'],
        'control': ['counter target', 'counterspell', 'countered', 'destroy target', 'destroy all', 'exile target', 'exile all', 'return target', 'return all', 'return each', 'to its owner\'s hand', 'tap target', 'tap all', 'doesn\'t untap', 'can\'t untap', 'gain control', 'change the target', 'prevent all damage', 'prevent that damage', 'can\'t activate', 'can\'t cast', 'can\'t attack', 'can\'t block', 'flash', 'scry'],
        'combo': ['you win the game', 'opponent loses the game', 'without paying its mana cost', 'without paying it\'s mana cost', 'rather than paying its mana cost', 'rather than paying it\'s mana cost', 'rather than paying its mana value', 'rather than paying it\'s mana value', 'the legend rule doesn\'t apply', 'storm', 'cast for {0}', 'repeat this process', 'infinite'],
        'ramp': ['add mana', 'add {', 'mana pool', 'untap target land', 'search your library for a land', 'search your library for a basic land', 'put it onto the battlefield', 'put that card onto the battlefield', 'play an additional land', 'you may play an additional land', 'treasure token', 'create a treasure', 'landfall'],
        'card_draw': ['draw a card', 'draw cards', 'draw two cards', 'draw three cards', 'investigate', 'clue token', 'you may play that card this turn', 'until end of your next turn', 'exile the top card', 'you may cast that card', 'look at the top', 'put it into your hand', 'whenever you draw'],
        'token': ['token', 'tokens you control', 'token creature', 'creature token', 'create a', 'create x', 'populate', 'amass', 'copy', 'create that many'],
        'life_gain': ['gain life', 'whenever you gain life', 'you gain that much life', 'lifelink', 'life total', 'food token'],
        'life_drain': ['lose life', 'loses that much life', 'each opponent loses', 'pay life', 'pays life', 'paying life'],
        'mill': ['mill', 'puts the top', 'from the top of their library', 'put into your graveyard', 'library into graveyard', 'surveil'],
        'discard': ['discard a card', 'discard cards', 'each player discards', 'randomly discards', 'from their hand', 'madness', 'blood'],
        'reanimation': ['reanimate', 'return target creature card', 'return target', 'from your graveyard to the battlefield', 'graveyard to the battlefield', 'put target creature card from your graveyard onto the battlefield', 'bring back', 'unearth'],
        'burn': ['damage to any target', 'damage to each opponent', 'damage to target creature', 'damage to target planeswalker', 'deals damage equal to', 'whenever you cast a spell'],
        'enchantment': ['enchant', 'aura', 'constellation', 'saga', 'whenever you cast an enchantment'],
        'equipment': ['equip', 'attach', 'whenever equipped', 'whenever you attach', 'when equipped'],
        'artifact': ['artifact', 'metalcraft', 'improvise', 'affinity for artifacts', 'whenever you cast an artifact'],
        'planeswalker': ['planeswalker', 'loyalty', 'loyalty counter', 'proliferate'],
        'tribal': ['elf', 'goblin', 'zombie', 'vampire', 'warrior', 'merfolk', 'soldier', 'dragon', 'angel', 'wizard', 'knight', 'sliver'],
        'voltron': ['equip', 'attach', 'aura', 'background', 'whenever equipped'],
        'combat': ['combat damage', 'attacks', 'blocks', 'attacking', 'blocking', 'target attacking', 'target blocking', 'each combat', 'vigilance', 'first strike', 'double strike', 'flying', 'menace', 'deathtouch', 'reach', 'trample', 'infect', 'poison counter', 'proliferate', 'with infect', 'flanking', 'horsemanship'],
        'stax': ['players can\'t', 'opponents can\'t', 'can\'t cast', 'can\'t activate', 'opponent can\'t untap', 'doesn\'t untap', 'skip your untap step', 'opponents can\'t draw', 'can\'t search libraries', 'spells cost', 'costs {', 'enters the battlefield tapped'],
        'storm': ['storm', 'copy this spell', 'copy that spell', 'copy target spell', 'copy target instant', 'copy target sorcery', 'magecraft', 'replicate'],
        'graveyard': ['graveyard', 'from your graveyard', 'mill', 'unearth', 'flashback', 'escape', 'delve', 'dredge'],
        'sacrifice': ['sacrifice a creature', 'sacrifice a permanent', 'sacrifice another', 'whenever you sacrifice', 'sacrifice'],
        'extb': ['whenever a creature dies', 'whenever another creature dies', 'whenever this creature dies', 'when this creature dies', 'dies', 'would die', 'put into a graveyard'],
        'etb': ['enters the battlefield', 'enters under your control', 'exile and return', 'exile then return', 'blink', 'flicker', 're-enter the battlefield'],
        'blink': ['blink', 'flicker', 'exile and return', 'exile then return', 'return it to the battlefield'],
        'removal': ['destroy target', 'destroy all', 'exile target', 'exile all', 'remove from the game', 'destroy'],
        'buff': ['creatures you control get', 'tokens you control get', 'put a +1/+1 counter', 'proliferate'],
        'library_control': ['scry', 'on top of your library', 'look at the top', 'reveal the top', 'put it into your hand'],
        'protection': ['indestructible', 'hexproof', 'shroud', 'protection from', 'prevent all damage', 'you have hexproof', 'you have shroud'],
        'aristocrats': ['whenever a creature you control dies', 'whenever another creature you control dies', 'whenever you sacrifice', 'each opponent loses', 'you gain 1 life'],
        'spellslinger': ['instant or sorcery', 'whenever you cast an instant or sorcery', 'magecraft', 'copy target instant', 'copy target sorcery'],
        'landfall': ['landfall', 'whenever a land enters the battlefield', 'whenever a land enters'],
        'wheel': ['each player discards', 'then draws', 'discards their hand', 'draw that many cards'],
        'theft': ['gain control of', 'control of target', 'until end of turn, untap', 'you may cast', 'you may play'],
        'tutor': ['search your library for a card', 'search your library for an', 'reveal it', 'shuffle'],
        'counters': ['+1/+1 counter', 'proliferate', 'put a counter on', 'remove a counter from']
    }
    __tags_general = {tag: [phrase.lower() for phrase in phrases] for tag, phrases in __tags_general.items()}

    __tags_joint = {
        'control': [
            ['counter', 'target'],
            ['exile', 'target'],
            ['destroy', 'target'],
            ['return', 'to its owner'],
        ],
        'removal': [
            ['destroy', 'target'],
            ['exile', 'target'],
            ['destroy', 'all'],
            ['exile', 'all'],
        ],
        'ramp': [
            ['untap', 'land'],
            ['search your library', 'land'],
            ['put', 'onto the battlefield'],
            ['add', 'mana'],
            ['create', 'treasure'],
            ['play', 'an additional land'],
        ],
        'tutor': [
            ['search your library', 'a card'],
            ['search your library', 'reveal'],
            ['search your library', 'shuffle'],
        ],
        'protection': [
            ['prevent', 'damage'],
            ['gains', 'hexproof'],
            ['gains', 'indestructible'],
        ],
        'etb': [
            ['enters', 'the battlefield'],
            ['enters', 'battlefield'],
        ],
        'blink': [
            ['exile', 'return'],
            ['exile', 'then return'],
        ],
        'extb': [
            ['when', 'dies'],
            ['whenever', 'dies'],
            ['put into', 'graveyard'],
            ['would', 'die'],
        ],
        'aristocrats': [
            ['sacrifice', 'a creature'],
            ['creature', 'dies'],
            ['each opponent', 'loses'],
        ],
        'life_gain': [['gain', 'life']],
        'life_drain': [['lose', 'life'], ['pay', 'life']],
        'wheel': [
            ['each player', 'discards'],
            ['then', 'draw'],
        ],
        'spellslinger': [
            ['instant', 'or', 'sorcery'],
            ['whenever you cast', 'instant'],
            ['whenever you cast', 'sorcery'],
        ],
        'stax': [
            ['players can\'t', 'cast'],
            ['opponents can\'t', 'draw'],
            ['spells cost', 'more'],
        ],
        'buff': [
            ['creatures', 'you control', 'get'],
            ['tokens', 'you control', 'get'],
            ['put', '+1/+1', 'counter'],
        ],
        'token': [
            ['token', 'creature'],
            ['tokens', 'you control'],
            ['create', 'a', 'token'],
        ],
        'landfall': [
            ['whenever', 'a land', 'enters'],
            ['land', 'enters', 'the battlefield'],
        ],
    }
    __tags_joint = {tag: [[p.lower() for p in pair] for pair in pairs] for tag, pairs in __tags_joint.items()}

    __tags_regex = {
        'aggro': [
            r'attack each turn',
            r'additional combat phase',
            r'take an extra combat phase',
            r'deals? \\d+ damage'
        ],
        'combo': [
            r'cast for \\{0\\}',
            r'you win the game',
            r'opponent loses the game',
            r'repeat this process'
        ],
        'control': [
            r'counter target .* spell',
            r'return target .* to its owner\\x27s hand',
            r'exile target .*'
        ],
        'ramp': [
            r'add \\{[wubrgc]\\}',
            r'add \\{\\d+\\}',
            r'search your library for a(?:n)? (?:basic )?(?:[a-zA-Z]+ )?land',
            r'put (?:it|that card) onto the battlefield(?: tapped)?',
            r'play an additional land'
        ],
        'tutor': [
            r'search your library for a card',
            r'search your library for an? (?:artifact|creature|enchantment|instant|sorcery|land|planeswalker) card'
        ],
        'card_draw': [
            r'draw (\\d+|[a-zA-Z]+) cards?',
            r'investigate',
            r'exile the top (\\d+|[a-zA-Z]+) cards? of your library.*you may (?:play|cast)'
        ],
        'life_gain': [
            r'gain \\d+ life',
            r'gain [a-zA-Z]+ life',
            r'gains \\d+ life',
            r'gains [a-zA-Z]+ life'
        ],
        'life_drain': [
            r'loses \\d+ life',
            r'loses [a-zA-Z]+ life',
            r'pays \\d+ life',
            r'pays [a-zA-Z]+ life'
        ],
        'burn': [
            r'deals? \\d+ damage',
            r'deals? [a-zA-Z]+ damage'
        ],
        'stax': [
            r'costs? \\{\\d+\\} more to cast',
            r'players can\\x27t',
            r'opponents can\\x27t'
        ],
        'buff': [
            r'\\+\\d+/\\+\\d+',
            r'\\+\\d+/\\-\\d+',
            r'\\-\\d+/\\+\\d+',
            r'\\+\\w+/\\+\\w+',
            r'\\+\\w+/\\-\\w+',
        ],
        'counters': [
            r'\\+1/\\+1 counter',
            r'proliferate',
            r'put a counter on',
            r'remove a counter from'
        ],
        'wheel': [
            r'each player discards.*then draws',
            r'discards their hand.*draws'
        ],
        'blink': [
            r'exile .* then return .* to the battlefield',
            r'exile .* return .* to the battlefield'
        ],
        'spellslinger': [
            r'instant or sorcery',
            r'whenever you cast an? (?:instant|sorcery)'
        ]
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
        'battle': ['Siege'],
        'voltron': ['Equipment', 'Vehicle', 'Aura', 'Background', 'Saga', 'Role', 'Shard', 'Cartouche', 'Case', 'Class', 'Curse', 'Rune'],
        'discard': ['Blood'],
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
    __BASIC_LANDS = {land.lower() for land in __BASIC_LANDS}

    _BASIC_LANDS_WITH_SNOW = {"Plains","Island","Swamp","Mountain","Forest","Wastes", "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp", "Snow-Covered Mountain", "Snow-Covered Forest"}
    _BASIC_LANDS_WITH_SNOW = {land.lower() for land in _BASIC_LANDS_WITH_SNOW}

    __COLOR_BASIC_LAND_MAP = {
        'W': 'Plains',
        'U': 'Island',
        'B': 'Swamp',
        'R': 'Mountain',
        'G': 'Forest',
        'C': 'Wastes'
    }

    __tag_synergy: Dict[str, set[str]] = {
        "sacrifice": {"aristocrats", "graveyard", "reanimation", "token", "extb"},
        "aristocrats": {"sacrifice", "extb", "token", "graveyard", "reanimation", "life_drain"},
        "graveyard": {"mill", "discard", "reanimation", "sacrifice", "extb"},
        "reanimation": {"graveyard", "mill", "discard", "sacrifice", "extb"},
        "mill": {"graveyard", "reanimation"},
        "discard": {"graveyard", "reanimation", "wheel", "life_drain"},
        "wheel": {"discard", "card_draw", "life_drain"},
        "spellslinger": {"storm", "card_draw", "control"},
        "storm": {"spellslinger", "card_draw"},
        "blink": {"etb", "token", "control"},
        "etb": {"blink", "token"},
        "token": {"buff", "sacrifice", "aristocrats", "etb"},
        "counters": {"buff", "combat"},
        "voltron": {"equipment", "enchantment", "protection", "combat"},
        "control": {"removal", "library_control", "card_draw"},
        "ramp": {"landfall", "card_draw"},
        "landfall": {"ramp", "token", "etb"},
    }

    @staticmethod
    def card_types() -> list[str]:
        """Return normalized card type vocabulary."""
        return sorted(CardFields.__card_types)

    @staticmethod
    def card_supertypes() -> list[str]:
        """Return normalized card supertype vocabulary."""
        return sorted(CardFields.__card_supertypes)

    @staticmethod
    def card_subtypes() -> list[str]:
        """Return all supported card subtype tokens."""
        return CardFields.__all_subtypes

    @staticmethod
    def color_identities() -> list[str]:
        """Return canonical color identity symbols."""
        return sorted(CardFields.__color_identities)

    @staticmethod
    def card_tags() -> list[str]:
        """Return top-level card tags."""
        return sorted(CardFields.__tags_general.keys())

    @staticmethod
    def general_tags() -> Dict[str, list[str]]:
        """Return substring-based tag phrase map."""
        return CardFields.__tags_general

    @staticmethod
    def joint_tags() -> Dict[str, list[list[str]]]:
        """Return joint-phrase tag rules."""
        return CardFields.__tags_joint

    @staticmethod
    def regex_tags() -> Dict[str, list[str]]:
        """Return regex-based tag rules."""
        return CardFields.__tags_regex

    @staticmethod
    def subtype_tags() -> Dict[str, list[str]]:
        """Return subtype-to-tag rule mapping."""
        return CardFields.__subtype_tags

    @staticmethod
    def card_types_set() -> set[str]:
        """Return card types as set."""
        return CardFields.__card_types_set

    @staticmethod
    def card_supertypes_set() -> set[str]:
        """Return card supertypes as set."""
        return CardFields.__card_supertypes_set

    @staticmethod
    def creature_subtypes_set() -> set[str]:
        """Return creature subtypes as set."""
        return CardFields.__creature_subtypes_set

    @staticmethod
    def artifact_subtypes_set() -> set[str]:
        """Return artifact subtypes as set."""
        return CardFields.__artifact_subtypes_set

    @staticmethod
    def battle_subtypes_set() -> set[str]:
        """Return battle subtypes as set."""
        return CardFields.__battle_subtypes_set

    @staticmethod
    def enchantment_subtypes_set() -> set[str]:
        """Return enchantment subtypes as set."""
        return CardFields.__enchantment_subtypes_set

    @staticmethod
    def land_subtypes_set() -> set[str]:
        """Return land subtypes as set."""
        return CardFields.__land_subtypes_set

    @staticmethod
    def spell_subtypes_set() -> set[str]:
        """Return spell subtypes as set."""
        return CardFields.__spell_subtypes_set

    @staticmethod
    def color_identities_set() -> set[str]:
        """Return color identity symbols as set."""
        return CardFields.__color_identities_set

    @staticmethod
    def tags_general_sets() -> Dict[str, set[str]]:
        """Return general tag phrases precompiled as sets."""
        return CardFields.__tags_general_sets

    @staticmethod
    def tags_joint_sets() -> Dict[str, list[set[str]]]:
        """Return joint tag phrase groups as sets."""
        return CardFields.__tags_joint_sets

    @staticmethod
    def tags_regex_sets() -> Dict[str, set[str]]:
        """Return regex patterns grouped by tag."""
        return CardFields.__tags_regex_sets

    @staticmethod
    def tags_subtype_sets() -> Dict[str, set[str]]:
        """Return subtype mapping grouped by tag."""
        return CardFields.__tags_subtype_sets

    @staticmethod
    def rarities() -> list[str]:
        """Return rarity values sorted."""
        return sorted(CardFields.__rarities)

    @staticmethod
    def basic_lands() -> set[str]:
        """Return lowercase names of basic lands."""
        return CardFields.__BASIC_LANDS

    @staticmethod
    def color_basic_land_map() -> Dict[str, str]:
        """Return map of color symbol to basic land name."""
        return CardFields.__COLOR_BASIC_LAND_MAP

    @staticmethod
    def basic_type_name(card_name: str) -> str:
        """Return canonical basic land type from a card name."""
        if card_name not in CardFields.__BASIC_LANDS:
            return None
        for t in ("Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"):
            if t.lower() in card_name.lower():
                return t
        return "Wastes"

    @staticmethod
    def is_basic_land(card_name: str) -> bool:
        """Check whether a card name is a basic land."""
        return card_name.strip().lower() in CardFields.basic_lands()

    @staticmethod
    def basic_land_type(card_name: str) -> str:
        """Return normalized basic land type for a name."""
        return CardFields.basic_type_name(card_name.strip().lower())

    @staticmethod
    def rarity_map() -> Dict[int, str]:
        """Return rarity index-to-name mapping."""
        return {i + 1: CardFields.__rarities[i] for i in range(len(CardFields.__rarities))}

    @staticmethod
    def rarity_to_index() -> Dict[str, int]:
        """Return rarity name-to-index mapping."""
        return {CardFields.__rarities[i]: i + 1 for i in range(len(CardFields.__rarities))}

    @staticmethod
    def all_tags() -> set[str]:
        """Return set of all known tags."""
        return CardFields.__all_tags

    @staticmethod
    def tag_synergy_map() -> Dict[str, set[str]]:
        """Return tag synergy adjacency map."""
        return CardFields.__tag_synergy

    @staticmethod
    def pred_dim(dim: int) -> Callable[[str, np.ndarray], bool]:
        """Create predicate matching vectors by feature dimension."""
        return lambda _, v: v.shape[-1] == dim

    @staticmethod
    def pred_regex(pattern: str, flags: int = re.IGNORECASE) -> Callable[[str, np.ndarray], bool]:
        """Create predicate matching names by regex."""
        rx = re.compile(pattern, flags)
        return lambda n, _: rx.search(n) is not None

    @staticmethod
    def pred_not(inner: Callable[[str, np.ndarray], bool]) -> Callable[[str, np.ndarray], bool]:
        """Negate a name/vector predicate."""
        return lambda n, v: not inner(n, v)

    @staticmethod
    def pred_all(*preds: Callable[[str, np.ndarray], bool]) -> Callable[[str, np.ndarray], bool]:
        """Compose predicates with logical AND."""
        return lambda n, v: all(p(n, v) for p in preds)

    @staticmethod
    def pred_any(*preds: Callable[[str, np.ndarray], bool]) -> Callable[[str, np.ndarray], bool]:
        """Compose predicates with logical OR."""
        return lambda n, v: any(p(n, v) for p in preds)

    @staticmethod
    def tag_text(name: str, text: str) -> set[str]:
        """Infer tags from card name/text using phrase and regex rules."""
        text = text.replace(name, '').lower()
        tags = set()

        for tag, phrases in CardFields.__tags_general_sets.items():
            if tag in tags:
                continue
            if any(p.lower() in text for p in phrases):
                tags.add(tag)

        for tag, pairs in CardFields.__tags_joint_sets.items():
            if tag in tags:
                continue
            for pair in pairs:
                if all(p.lower() in text for p in pair):
                    tags.add(tag)

        for tag, patterns in CardFields.__tags_regex_sets.items():
            if tag in tags:
                continue
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    tags.add(tag)
        return tags

    @staticmethod
    def tag_subtypes(subtypes: list[str] | None) -> set[str]:
        """Infer tags from card subtype names."""
        if not subtypes:
            return set()

        card_tags: set[str] = set()
        for tag, subtype_set in CardFields.tags_subtype_sets().items():
            if any(st.lower() in subtype_set for st in subtypes):
                card_tags.add(tag)
        return card_tags

    @staticmethod
    def tag_card(card: Card) -> set[str]:
        """Compute union of text-derived and subtype-derived tags."""
        return CardFields.tag_text(card.card_name, card.text).union(CardFields.tag_subtypes(card.card_subtypes))

    @staticmethod
    def parse_mtgjson_card(card: dict) -> Card:
        """Parse an MTGJSON card object into a `Card` model."""
        card_kwargs = {
            'commander_legal': 'commander' in card['legalities'] and card['legalities']['commander'] == "Legal" and 'paper' in card['availability'],
            'card_name': card.get('name', ''),
            'card_types': [str(s).lower() for s in card.get('types', '')],
            'card_supertypes': [str(s).lower() for s in card.get('supertypes', '')],
            'card_subtypes': [str(s).lower() for s in card.get('subtypes', '')],
            'mana_cost': card.get('manaValue', -1),
            'mana_cost_exp': card.get('manaCost', ''),
            'color_identity': card.get('colorIdentity', []),
            'defense': card.get('defense', ''),
            'rarity': card.get('rarity', ''),
            'text': card.get('text', ''),
            'rank': card.get('edhrecRank', ''),
            'power': card.get('power', ''),
            'toughness': card.get('toughness', ''),
            'loyalty': card.get('loyalty', ''),
            'card_id': card.get('identifiers','').get('scryfallId', ''),
        }
        return Card(**card_kwargs)

    @staticmethod
    def parse_moxfieldapi_card(card: dict) -> Card:
        """Parse a Moxfield API card object into a `Card` model."""
        split = card.get('type_line', 'None').split(' â€” ')
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
            'card_id': card.get('scryfall_id', ''),
        }
        return Card(**card_kwargs)

    @staticmethod
    def parse_moxfield_group(deck: dict, key: str):
        """Parse one Moxfield deck group into a list of `Card` objects."""
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
