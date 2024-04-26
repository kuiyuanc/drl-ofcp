import os
import random
import sys
from enum import Enum

import numpy as np
from numpy.typing import NDArray


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from lib.deuces import Card, Deck, Evaluator
from lib.deuces.evaluator import LookupTable


class Ofcp:
    class Street(Enum):
        FRONT = "front"
        MID = "mid"
        BACK = "back"

    class Action:
        def __init__(self, street: 'Ofcp.Street', card: int) -> None:
            self.street: 'Ofcp.Street' = street
            self.card: int = card

    class Agent:
        def __init__(self) -> None:
            pass

        def __call__(self, state: 'Ofcp', player: 'Ofcp.Player') -> 'Ofcp.Action':
            return random.choice(player.valid_actions())

    class Player:
        def __init__(self, hands: list[int], *, agent: 'Ofcp.Agent') -> None:
            self.streets: dict['Ofcp.Street', list[int]] = {street: [] for street in Ofcp.Street}
            self.hands: list[int] = hands
            self.agent: 'Ofcp.Agent' = agent

        def __call__(self, state: 'Ofcp', card: int) -> 'Ofcp.Action':
            action = self.agent(state, self)
            self.streets[action.street].append(self.hands.pop(self.hands.index(action.card)))
            self.hands.append(card)
            return action

        def set_agent(self, agent: 'Ofcp.Agent') -> None:
            self.agent = agent

        def valid_actions(self) -> tuple['Ofcp.Action', ...]:
            return tuple(Ofcp.Action(street, card) for card in self.hands for street in Ofcp.Street
                         if len(self.streets[street]) < Ofcp.NUM_SLOTS[street])

        def eval(self) -> tuple[dict['Ofcp.Street', int], int, bool]:
            ranks, is_burst = self._ranks()
            return ranks, self._royalties(ranks), is_burst

        def reset(self, hands: list[int]) -> None:
            self.streets = {street: [] for street in Ofcp.Street}
            self.hands = hands

        def _ranks(self) -> tuple[dict['Ofcp.Street', int], bool]:
            ranks = {street: Ofcp.EVALUATOR.evaluate(hand) for street, hand in self.streets.items()}
            is_burst = ranks[Ofcp.Street.FRONT] < ranks[Ofcp.Street.MID] or ranks[Ofcp.Street.MID] < ranks[Ofcp.Street.BACK]
            return {street: Ofcp.RANK_BURST for street in Ofcp.Street} if is_burst else ranks, is_burst

        def _royalties(self, ranks: dict['Ofcp.Street', int]) -> int:
            # TODO: calculate royalties
            return 0

    EVALUATOR = Evaluator()
    NUM_INITIAL_CARDS = 5
    NUM_SLOTS = {Street.FRONT: 3, Street.MID: 5, Street.BACK: 5}
    RANK_BURST = LookupTable.MAX_HIGH_CARD + 1

    def __init__(self, *, num_players: int = 2) -> None:
        if num_players != 2:
            raise NotImplementedError("2 players only")

        self.deck: Deck = Deck()
        self.players: list[Ofcp.Player] = [Ofcp.Player(self.deck.draw(Ofcp.NUM_INITIAL_CARDS), agent=Ofcp.Agent())
                                           for _ in range(num_players)]
        self.turn: int = 0

    def __next__(self) -> bool:
        return self.next()

    def set_player_agent(self, player_id: int, agent: 'Ofcp.Agent') -> None:
        if player_id < 1 or player_id > len(self.players):
            raise ValueError("Invalid player ID")
        self.players[player_id - 1].set_agent(agent)

    def next(self) -> bool:
        self.players[self.turn % len(self.players)](self, self.deck.draw())  # type: ignore
        self.turn += 1
        return self.turn < len(self.players) * sum(Ofcp.NUM_SLOTS.values())

    def eval(self) -> NDArray[np.float64]:
        players_ranks, royalties, bursts = zip(*(player.eval() for player in self.players))

        ranks = np.array(tuple(tuple(player_ranks[street] for street in Ofcp.Street) for player_ranks in players_ranks))
        ranks = ranks == ranks.min(axis=0)
        num_best = np.count_nonzero(ranks, axis=0)
        winner_points = (np.full(len(Ofcp.Street), len(self.players)) - num_best) / num_best
        street_points = ranks @ winner_points + ~ranks @ np.full(len(Ofcp.Street), -1)

        max_street_point = (len(self.players) - 1) * len(Ofcp.Street)
        winner = street_points == max_street_point
        scoops = winner * max_street_point - ~winner * len(Ofcp.Street) if winner.sum() else np.zeros(len(self.players))

        royalties = np.array(royalties, dtype=np.float64)
        royalties -= royalties.mean()

        return np.array(tuple(zip(street_points, scoops, royalties, bursts)))

    def restart(self) -> None:
        self.deck.shuffle()
        for i in range(len(self.players)):
            self.players[i].reset(self.deck.draw(Ofcp.NUM_INITIAL_CARDS))
        self.turn = 0


class OfcpUI(Ofcp):
    DIVIDER = {1: '=' * 30, 2: '-' * 20}

    def __init__(self, *, num_players: int = 2) -> None:
        super().__init__(num_players=num_players)

        self._print_start()

    def next(self) -> bool:
        action = self.players[self.turn % len(self.players)](self, self.deck.draw())  # type: ignore
        self.turn += 1
        not_end = self.turn < len(self.players) * sum(Ofcp.NUM_SLOTS.values())

        self._print_state()
        self._print_action(index=(self.turn - 1) % len(self.players) + 1, action=action)
        self._print_divider(2)

        return not_end

    def eval(self) -> NDArray[np.float64]:
        scores = super().eval()

        self._print_eval(scores)
        self._print_divider(1)

        return scores

    def restart(self) -> None:
        super().restart()

        self._print_start()

    def _print_start(self) -> None:
        self._print_divider(1)
        self._print_state()
        self._print_action()
        self._print_divider(2)

    def _print_divider(self, level: int) -> None:
        print(self.DIVIDER[level])

    def _print_state(self) -> None:
        print(f"turn {self.turn:>2d}")
        for index, player in enumerate(self.players):
            self._print_player(player, index)

    def _print_player(self, player: Ofcp.Player, index: int, *, hide_class: bool = True) -> None:
        print(f"Player {index + 1} ( {player.agent.__class__.__name__} )")
        print(f"\t{"hands":<13} : {''.join(Card.int_to_pretty_str(card) for card in sorted(player.hands))}")
        print(*(f"\t{street.value + " street":<13} : "
                f"{''.join(Card.int_to_pretty_str(card) for card in sorted(player.streets[street])):<45}"
                f"{"" if hide_class else self._hands_to_str(player.streets[street])}"
                for street in Ofcp.Street),
              sep='\n')

    def _hands_to_str(self, hands: list[int]) -> str:
        return Ofcp.EVALUATOR.class_to_string(Ofcp.EVALUATOR.get_rank_class(Ofcp.EVALUATOR.evaluate(hands)))

    def _print_action(self, *, index: int | None = None, action: Ofcp.Action | None = None) -> None:
        print(f"Player {index} put {Card.int_to_pretty_str(action.card)} at {action.street.value} street"
              if index and action else "No action yet")

    def _print_eval(self, scores: NDArray[np.float64]) -> None:
        for index, player in enumerate(self.players):
            self._print_player(player, index, hide_class=False)

        print("Scores :")
        for [index, player], [street, scoop, royalty, burst] in zip(enumerate(self.players), scores):
            print(f"\tPlayer {index + 1} ( {player.agent.__class__.__name__} )")
            print(f"\t\tstreet score : {street}")
            print(f"\t\tscoop        : {scoop}")
            print(f"\t\troyalty      : {royalty}")
            print(f"\t\tburst        : {"Yes" if burst else "No"}")


def test() -> None:
    import time

    begin = time.time()

    ofcp = Ofcp()
    for i in range(int(1e6)):
        while ofcp.next():
            pass
        ofcp.eval()
        ofcp.restart()
        if i % int(1e4) == 0:
            print(f"\r{int(i / 1e4):3d}%", end='')
    print('\r', end='')

    print(f"time : {time.time() - begin:.2f}s")
