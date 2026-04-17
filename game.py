import abc
from typing import Optional, List
import time

import numpy as np

import deck


NUM_CARDS_TO_PLAY = 5
STARTING_POINTS = 100


class Strategy(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self,
        my_points: int,
        op_points: int,
        prev_cards_drawn: List[int],
        num_remaining_draws: int,
        rng: np.random.RandomState,
    ) -> int:
        ...


def strtime():
    return time.strftime("%Y%m%d %H:%M:%S")


def play_game(player_1: Strategy, player_2: Strategy, rng: np.random.RandomState, verbose=True) -> int:
    player_1_str = f'Player 1 [{player_1}]'
    player_2_str = f'Player 2 [{player_2}]'

    points_1 = STARTING_POINTS
    points_2 = STARTING_POINTS

    cards_drawn = []
    next_card = deck.random_card(rng)

    if verbose:
        print('Starting game')
        print(f'First card [{deck.name_card(next_card)}]')
        print(f'{player_1_str} has {points_1} points')
        print(f'{player_2_str} has {points_2} points')

    for turn in range(1, NUM_CARDS_TO_PLAY):
        prev_card = next_card
        cards_drawn.append(prev_card)

        num_remaining_draws = NUM_CARDS_TO_PLAY - turn

        stake_1 = player_1.evaluate(
            points_1,
            points_2,
            [c for c in cards_drawn],
            num_remaining_draws,
            rng,
        )
        stake_1 = int(stake_1)
        stake_1 = min(max(stake_1, -points_1), points_1)

        stake_2 = player_2.evaluate(
            points_2,
            points_1,
            [c for c in cards_drawn],
            num_remaining_draws,
            rng,
        )
        stake_2 = int(stake_2)
        stake_2 = min(max(stake_2, -points_2), points_2)

        if verbose:
            print(
                f'{player_1_str} stakes {abs(stake_1)}'
                + (' on higher' if stake_1 > 0 else '')
                + (' on lower' if stake_1 < 0 else '')
            )
            print(
                f'{player_2_str} stakes {abs(stake_2)}'
                + (' on higher' if stake_2 > 0 else '')
                + (' on lower' if stake_2 < 0 else '')
            )

        next_card = deck.draw_card(cards_drawn, rng)

        if next_card > prev_card:
            if verbose:
                print(f'Next card [{deck.name_card(next_card)}] is higher')
            points_1 += stake_1
            points_2 += stake_2
        else:
            if verbose:
                print(f'Next card [{deck.name_card(next_card)}] is lower')
            points_1 -= stake_1
            points_2 -= stake_2

        if verbose:
            print(f'{player_1_str} has {points_1} points')
            print(f'{player_2_str} has {points_2} points')

        if points_2 == 0:
            if points_1 == 0:
                if verbose:
                    print('Draw!')
                return 0
            else:
                if verbose:
                    print(f'{player_1_str} wins!')
                return 1
        elif points_1 == 0:
            if verbose:
                print(f'{player_2_str} wins!')
            return -1

    if points_1 > points_2:
        if verbose:
            print(f'{player_1_str} wins!')
        return 1
    elif points_1 < points_2:
        if verbose:
            print(f'{player_2_str} wins!')
        return -1
    else:
        if verbose:
            print('Draw!')
        return 0


def play_tournament(players: List[Strategy], num_tournaments: int, rng: np.random.RandomState):
    num_players = len(players)

    row_wins = np.zeros((num_players, num_players), dtype=np.int64)
    col_wins = np.zeros((num_players, num_players), dtype=np.int64)
    draws = np.zeros((num_players, num_players), dtype=np.int64)

    tournament_idx_pct_prev = 0.0

    time_last = time.time()

    for tournament_idx in range(num_tournaments):
        for idx_1 in range(num_players):
            for idx_2 in range(idx_1, num_players):
                player_1 = players[idx_1]
                player_2 = players[idx_2]

                result = play_game(player_1, player_2, rng, verbose=False)

                if result > 0:
                    row_wins[idx_1, idx_2] += 1
                    col_wins[idx_2, idx_1] += 1
                elif result < 0:
                    row_wins[idx_2, idx_1] += 1
                    col_wins[idx_1, idx_2] += 1
                else:
                    draws[idx_1, idx_2] += 1
                    draws[idx_2, idx_1] += 1

        time_next = time.time()
        time_diff = time_next - time_last
        if time_diff > 1.0:
            tournament_idx_pct = (tournament_idx + 1) / num_tournaments
            if tournament_idx_pct == 1.0 or tournament_idx_pct - tournament_idx_pct_prev >= 0.01:
                tournament_idx_pct_prev = tournament_idx_pct
                print(f'{strtime()} -- Played {tournament_idx + 1} tournaments ({tournament_idx_pct * 100:.0f}%)')
            time_last = time_next

    return row_wins, col_wins, draws


def _log_sigmoid(x):
    return -np.log(1.0 + np.exp(-x))

def _d_log_sigmoid(x):
    return 1.0 / (1.0 + np.exp(x))


def results_to_elo(game_results):
    """Returns elo ratings for each player.
    https://en.wikipedia.org/wiki/Elo_rating_system#Formal_derivation_for_win/loss_games
    """
    import scipy.optimize as opt

    row_wins, col_wins, draws = game_results

    num_players = row_wins.shape[0]

    assert row_wins.shape == (num_players, num_players)
    assert col_wins.shape == (num_players, num_players)
    assert draws.shape == (num_players, num_players)

    # Prior for robustness
    row_wins_adj = row_wins + 0.5 * draws + 0.25
    col_wins_adj = col_wins + 0.5 * draws + 0.25

    def loss(elo):
        elo_diff = elo[:, None] - elo[None, :]

        log_loss_of_row_win = -_log_sigmoid(elo_diff)
        log_loss_of_col_win = -_log_sigmoid(-elo_diff)

        loss = np.sum(row_wins_adj * log_loss_of_row_win + col_wins_adj * log_loss_of_col_win)

        d_log_loss_of_row_win = row_wins_adj
        d_log_loss_of_col_win = col_wins_adj

        d_elo_diff = -d_log_loss_of_row_win * _d_log_sigmoid(elo_diff) + d_log_loss_of_col_win * _d_log_sigmoid(-elo_diff)

        d_elo = np.sum(d_elo_diff, axis=1) - np.sum(d_elo_diff, axis=0)

        return loss, d_elo

    elo_hat = opt.minimize(loss, np.zeros(num_players), jac=True).x
    elo_hat -= elo_hat[0]
    elo_hat *= 400.0 / np.log(10.0)

    return elo_hat


def plot_game_results(players: List[Strategy], game_results, ax=None):
    import matplotlib.pyplot as plt

    elo_scores = results_to_elo(game_results)

    if ax is None:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.subplots()

    num_players = len(players)

    row_wins, col_wins, draws = game_results
    assert row_wins.shape == (num_players, num_players)
    assert col_wins.shape == (num_players, num_players)
    assert draws.shape == (num_players, num_players)

    assert elo_scores.shape == (num_players,)

    elo_order = np.argsort(elo_scores)

    player_names = np.array([str(p) for p in players])[elo_order]

    row_wins = row_wins[elo_order][:, elo_order]
    col_wins = col_wins[elo_order][:, elo_order]
    draws = draws[elo_order][:, elo_order]

    elo_scores = elo_scores[elo_order]

    num_games = row_wins + col_wins + draws

    img_red = col_wins / num_games
    img_green = row_wins / num_games
    img_blue = draws / num_games

    img = np.stack([img_red, img_green, img_blue], axis=-1)

    tick_names = [f'{p_name} [{p_elo:.0f}]' for p_name, p_elo in zip(player_names, elo_scores)]

    ax.imshow(img)
    ax.set_xticks(np.arange(num_players))
    ax.set_xticklabels(tick_names, rotation=90)
    ax.set_yticks(np.arange(num_players))
    ax.set_yticklabels(tick_names)
