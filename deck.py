from typing import List

import numpy as np


NUM_CARDS = 52
CARD_NUMBERS = ['Ace'] + [str(i) for i in range(2, 11)] + ['Jack', 'Queen', 'King']
CARD_SUITS = ['Clubs', 'Diamonds', 'Hearts', 'Spades']


def all_cards():
    return range(NUM_CARDS)

def random_card(rng: np.random.RandomState) -> int:
    return rng.randint(NUM_CARDS)

def name_card(card: int) -> str:
    assert 0 <= card < NUM_CARDS
    number = CARD_NUMBERS[card // len(CARD_SUITS)]
    suit = CARD_SUITS[card % len(CARD_SUITS)]
    return f'{number} of {suit} (id={card})'

def draw_card(prev_cards_drawn: List[int], rng: np.random.RandomState) -> int:
    assert len(prev_cards_drawn) < NUM_CARDS

    if len(prev_cards_drawn) < NUM_CARDS // 2:
        while True:  # Rejection sampling
            choice = random_card(rng)
            if choice not in prev_cards_drawn:
                break
    else:
        prev_cards_drawn_set = set(prev_cards_drawn)
        cards_in_deck = np.array([
            card for card in all_cards()
            if card not in prev_cards_drawn_set
        ])
        choice = rng.choice(cards_in_deck)

    return choice