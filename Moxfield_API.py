import requests
import os
import time
import json
import random
import keyboard
from typing import Dict, List, Any, Optional
from Card_Lib import Deck, CardFields

headers: Dict[str, str] = {"User-Agent": str(os.environ.get("MOXFIELD_KEY"))}

def build_decks(num_pages: int = 2000) -> Dict[str, Deck]:
    decks: Dict[str, Deck] = {}
    for i in range(1, num_pages):
        params = {
            "format": "commander",
            'page': int(random.randint(1,num_pages*10000)),
            'pageSize': 100
        }
        try:
            response = requests.get("https://api2.moxfield.com/v2/decks/all/", headers=headers, timeout=1000, params=params)
            print(f"Page {i}, response code {response.status_code}")
            response_data = response.json()
            for data in response_data["data"]:
                if data['format']=='commander':
                    cards_list = []
                    subrequest = requests.get(f"https://api2.moxfield.com/v2/decks/all/{data["publicId"]}", headers=headers, timeout=1000)
                    time.sleep(0.5)
                    deck = subrequest.json()
                    if 'mainboard' not in deck:
                        continue
                    mainboard = deck['mainboard']
                    for card in list(mainboard.keys()):
                        card_obj = CardFields.parse_moxfieldapi_card(mainboard[card]['card'])
                        cards_list.append((card_obj, mainboard[card_obj.card_name]['quantity']))
                
                    deck_kwargs = {
                        'id': deck.get("publicId",""),
                        'colors': data.get("colors", []),
                        'color_percentages': data.get("colorPercentages",{}),
                        'bracket': data.get("bracket",1),
                        'format': data.get("format", "commander"),
                        'commanders': CardFields.parse_moxfield_group(deck, "commanders"),
                        'companions': CardFields.parse_moxfield_group(deck, "companions"),
                        'mainboard_count': deck.get("mainboardCount", len(cards_list)),
                        'cards': cards_list,
                    }

                    deck_obj = Deck(**deck_kwargs)
                    if len(deck_obj)>=100:
                        decks[deck["name"]] = deck_obj
                    
                    print(deck_obj)
                    time.sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")
            save('datasets/Decks.json', decks)
    return decks

def build_decks_simple(num_pages: int = 2000, input: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    decks_out: Dict[str, Dict[str, Any]] = {} if input is None else dict((deck['id'], deck) for deck in input.values())
    print(f"Starting with {len(decks_out)} decks from input.")
    try:
        for _ in range(1, num_pages):
            params = {
                "format": "commander",
                "page": int(random.randint(1, num_pages * 10000)),
                "pageSize": 100
            }
            try:
                response = requests.get("https://api2.moxfield.com/v2/decks/all/", headers=headers, timeout=1000, params=params)
                response_data = response.json()
            except Exception as e:
                print("Main Request Failed for {response}")
                print(f"Error: {e}")
                save_simple("datasets/DecksSimple.json", decks_out)
                time.sleep(2.5)
                break

            for data in response_data.get("data", []):
                if "commander" not in data.get("format", ""):
                    continue
                
                try:
                    sub = requests.get(f"https://api2.moxfield.com/v2/decks/all/{data['publicId']}", headers=headers, timeout=1000)
                    deck = sub.json()
                except Exception as e:
                    print(f"Subrequest failed for {data['publicId']}, {sub}")
                    print(f"Error: {e}")
                    save_simple("datasets/DecksSimple.json", decks_out)
                    time.sleep(2.5)
                    continue
                
                if keyboard.is_pressed('s'):
                    save_simple("datasets/DecksSimple.json", decks_out)
                    print(f"[hotkey] Saved {len(decks_out)} decks")
                
                if 'mainboard' not in deck:
                    continue

                deck_id = deck.get("publicId", "")
                if deck_id in decks_out:
                    continue
                commanders = list(deck.get("commanders", []).keys())
                cards = [key for key, value in deck["mainboard"].items() for _ in range(int(value.get("quantity", 1)))]
                if(not commanders) or (len(cards)+len(commanders) < 100):
                    continue

                decks_out[deck_id] = {
                    "id": deck_id,
                    "commanders": commanders,
                    "cards": cards
                }

                print(f"Fetched {deck_id}: {commanders[0] if commanders else 'Unknown'} ({len(cards)+len(commanders)} cards), decks so far: {len(decks_out)}")
                time.sleep(2.5)

    except Exception as e:
        print(f"Error: {e}")
        save_simple("datasets/DecksSimple.json", decks_out)

    return decks_out

def save(path: str, decks: Dict[str, Deck]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({key: value.get_attributes() for key, value in decks.items()}, f, indent=4, ensure_ascii=False)

def save_simple(path: str, decks: Dict[str, Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decks, f, indent=4, ensure_ascii=False)

def load_simple(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    decks = build_decks_simple(input=load_simple("datasets/DecksSimple.json"))
    save_simple("datasets/DecksSimple.json", decks)