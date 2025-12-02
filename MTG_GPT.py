from mtgsdk import Card
from openai import OpenAI
import os
from typing import List

client: OpenAI = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL: str = "gpt-3.5-turbo"
INITIAL_MESSAGE: str = ""
CONTEXT: List[str] = [
"""
You are a program that helps build magic the gathering commander decks. When building decks, you follow rules of deckbuilding exactly as specified. 
""",
"""
Here are some terms you will need to know for building decks:
- A commander is a legendary permanent (either creature or select planeswalker) that guides the decks identity. When a commander is selected, colors of cards in the deck must follow the commanders color identity. 
- Color identity is the colors of the mana value or mana produced by a card. Lands also have a color identity, which is defined by the mana they generate or fetch for. 
- The types of cards in magic are: lands, artifacts, enchantments, creatures, planeswalkers, instants, sorceries, and battles. 
- Basic lands are lands that say in their supertype: "basic land." Non basic lands are all other lands -- lands that do not have the supertype "basic land." Fetch lands are lands that search through your library for lands. Multicolored lands are lands that produce mana of multiple colors. The sum total of lands is the number of all lands, whether it be basic or non-basic, added together. 
""",
"""
These following rules are extremely important and must be followed exactly. The rules for commander deck building are as follows.:
- A card’s color identity is its color plus the color of any mana symbols in the card’s rules text. A card’s color identity is established before the game begins, and cannot be changed by game effects. The cards in a deck may not have any colors in their color identity which are not in the color identity of the deck’s commander.
- A Commander deck must contain exactly 100 cards, including the commander. If you’re playing a companion, it must adhere to color identity and singleton rules. While it is not part of the deck, it is effectively a 101st card.
- With the exception of basic lands, no two cards in the deck may have the same English name. There may be no repeated cards in the deck other than basic lands. 
- All cards must be legal in the commander format. 
""",
"When building the deck, make sure to include a suitable amount of basic lands, non-basic lands, win conditions, and competitive cards, artifacts, creatures, enchatments, and planeswalkers as you see fit. For reference, most commander decks have a total of 18 lands, including basic and non-basic lands. Do not go over that number of lands.",
"Format the cards listed as (quantity)x (card_name). Do not include any other text in the return message. ",
"""
When building the deck, follow the rules exactly.
"""]
BUILDER_MESSAGE: str = """
Make a Commander Deck using {name}. Gear the deck to work with {name} abilities, and mana cost. The deck must have a suitable amount of lands, cards that fit the {type}s color identity, {color}, and must be fairly competitive. Make sure to include some basic lands, while still keeping the sum total of lands at 18.
The deck should have 18 total lands, and a sum total of 100 cards. If you use fetch lands, the lands must only fetch for lands that are the color identity of the {type}.
"""
PLANESWALKER_COMMANDER_CONDITION: str = ("Can be your Commander").lower()

def chat_with_chatgpt(prompt: str = "") -> str:
    prompt = f"You: {prompt}\nBot:"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
        {"role": "system", "content": CONTEXT[0]},
        {"role": "system", "content": CONTEXT[1]},
        {"role": "system", "content": CONTEXT[2]},
        {"role": "system", "content": CONTEXT[3]},
        {"role": "system", "content": CONTEXT[4]},
        {"role": "system", "content": CONTEXT[5]},
        {"role": "user", "content": prompt},
        ],
        temperature=0.1
    )

    bot_response: str = response.choices[0].message.content
    return bot_response

def get_cards(card_name: str) -> List[str]:
    card_set: List[str] = []
    card_array_creatures: List[Card] = Card.where(name=card_name).where(type="creature").where(supertypes="legendary").where(legality="commander").all()
    card_array_planeswalkers: List[Card] = Card.where(name=card_name).where(type="planeswalker").where(supertypes="legendary").where(legality="commander").all()

    for individual_card in card_array_creatures:
        if individual_card.name not in card_set:
            card_set.append(individual_card.name)

    for individual_card in card_array_planeswalkers:
        if(PLANESWALKER_COMMANDER_CONDITION in individual_card.text.lower()):
            if individual_card.name not in card_set:
                card_set.append(individual_card.name)

    return card_set

while(True):
    user_input = input("Commander HERE: ")
    cards_list = get_cards(user_input)
    if(len(cards_list)>1):
        for each in cards_list:
            print(each)
    else:
        commander = Card.where(name=cards_list[0]).all()[0]
        print(chat_with_chatgpt(BUILDER_MESSAGE.format(name=commander.name, type=commander.type, color=commander.color_identity)))

