from openai import OpenAI
import os
from Card_Lib import Card, CardFields
import pandas as pd
import json

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

def minimized_card_data(card: Card) -> dict:
    return {
        'name': card.card_name,
        'type': card.card_types,
        'colors': card.color_identity,
        'mana_cost': card.mana_cost,
        'text': card.text,
        "pt_or_loyalty": {"power": card.power, "toughness": card.toughness, "loyalty": card.loyalty},
        'CEDH Rec Rank': card.rank,
    }

def rate_card(card: Card, model: str = "gpt-4o-mini") -> int:
    #Work on our string... say that there should be x, y, z quantities of each value
    context: str = f"""
    You are a Magic the Gathering expert.
    Rate the following card on a scale of 1 to 5, where 1 is the worst and  is the best. 

    1: Fringe/very niche; requires heavy build-around; usually outclassed.
    2: Playable but replaceable; fine in specific shells; average to slightly above.
    3: Strong staple-level role player; efficient interaction/mana/advantage; widely playable.
    4: All around strong cards; outclassed by higher tier options
    5: Top-tier staple or commander; warps deckbuilding; consistently game-changing.

    Respond only with a single integer between 1 and 5.
    Consider that the card is being rated for use in a Competitive Commander format deck.
    Ensure that there is a balance of cards being rated 1-5, estimating for a normal distribution across the classes. 
    \n{minimized_card_data(card)}"""
    response = client.chat.completions.create(model=model,messages=[{"role": "user", "content": context}],temperature=0)
    rating = response.choices[0].message.content.strip()
    return rating

def rate_cards(filename: str, num_cards: int = 10, model: str = "gpt-4o-mini") -> dict[str, int]:
    card_data: dict[str, int] = dict()
    card_set = pd.read_json(filename)['data'][2:]
    try:
        for game_set in card_set:
            for card in game_set['cards']:
                if 'commander' in card['legalities'] and card['legalities']['commander']=="Legal" and 'paper' in card['availability']:
                    cd = CardFields.parse_mtgjson_card(card)
                    if cd.card_name not in card_data:
                        card_data[cd.card_name] = rate_card(cd, model=model)
                    if len(card_data) >= num_cards:
                        return card_data
        return card_data
    except Exception as e:
        print(f"Error: {e}")
        save(card_data)

def save(ratings) -> None:
    with open('datasets/RatingData.json', 'w') as f:
        json.dump(ratings, f, indent=4) 

if __name__ == "__main__":
    ratings = rate_cards("datasets/AllPrintings.json", 20000)
    save(ratings)
    
