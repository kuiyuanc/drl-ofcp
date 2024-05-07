import os
import random
import sys
from enum import Enum, IntEnum

import numpy as np
from numpy.typing import NDArray


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from lib.deuces import Card, Deck, Evaluator
from lib.deuces.evaluator import LookupTable


class Ofcp:
    class Street(Enum):
        FRONT = "front"
        MID   = "mid"
        BACK  = "back"

    class Action:
        def __init__(self, street: 'Ofcp.Street', card: int) -> None:
            self.street: 'Ofcp.Street' = street
            self.card: int = card

    NUM_INITIAL_CARDS = 5
    NUM_SLOTS = {Street.FRONT: 3, Street.MID: 5, Street.BACK: 5}
    EVALUATOR = Evaluator()
    RANK_BURST = LookupTable.MAX_HIGH_CARD + 1
    ROYALTY_LOOKUP = {
        (Street.FRONT, LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_PAIR]): {
            # 6 pair
            5426: 1, 5425: 1, 5424: 1, 5422: 1, 5418: 1, 5411: 1, 5400: 1, 5384: 1, 5362: 1, 5333: 1, 5296: 1, 5250: 1,
            # 7 pair
            5194: 2, 5193: 2, 5192: 2, 5190: 2, 5186: 2, 5179: 2, 5168: 2, 5152: 2, 5130: 2, 5101: 2, 5064: 2, 5018: 2,
            # 8 pair
            4962: 3, 4961: 3, 4960: 3, 4958: 3, 4954: 3, 4947: 3, 4936: 3, 4920: 3, 4898: 3, 4869: 3, 4832: 3, 4786: 3,
            # 9 pair
            4730: 4, 4729: 4, 4728: 4, 4726: 4, 4722: 4, 4715: 4, 4704: 4, 4688: 4, 4666: 4, 4637: 4, 4600: 4, 4554: 4,
            # T pair
            4498: 5, 4497: 5, 4496: 5, 4494: 5, 4490: 5, 4483: 5, 4472: 5, 4456: 5, 4434: 5, 4405: 5, 4368: 5, 4322: 5,
            # J pair
            4266: 6, 4265: 6, 4264: 6, 4262: 6, 4258: 6, 4251: 6, 4240: 6, 4224: 6, 4202: 6, 4173: 6, 4136: 6, 4090: 6,
            # Q pair
            4034: 7, 4033: 7, 4032: 7, 4030: 7, 4026: 7, 4019: 7, 4008: 7, 3992: 7, 3970: 7, 3941: 7, 3904: 7, 3858: 7,
            # K pair
            3802: 8, 3801: 8, 3800: 8, 3798: 8, 3794: 8, 3787: 8, 3776: 8, 3760: 8, 3738: 8, 3709: 8, 3672: 8, 3626: 8,
            # A pair
            3570: 9, 3569: 9, 3568: 9, 3566: 9, 3562: 9, 3555: 9, 3544: 9, 3528: 9, 3506: 9, 3477: 9, 3440: 9, 3394: 9,
        },
        (Street.FRONT, LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_THREE_OF_A_KIND]): {
            2480: 10,
            2413: 11,
            2346: 12,
            2279: 13,
            2212: 14,
            2145: 15,
            2078: 16,
            2011: 17,
            1944: 18,
            1877: 19,
            1810: 20,
            1743: 21,
            1676: 22
        },
        (Street.MID,  LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_THREE_OF_A_KIND]): 2,
        (Street.MID,  LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT]):        4,
        (Street.MID,  LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FLUSH]):           8,
        (Street.MID,  LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FULL_HOUSE]):     12,
        (Street.MID,  LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FOUR_OF_A_KIND]): 20,
        (Street.MID,  LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT_FLUSH]): 30,
        (Street.BACK, LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT]):        2,
        (Street.BACK, LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FLUSH]):           4,
        (Street.BACK, LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FULL_HOUSE]):      6,
        (Street.BACK, LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_FOUR_OF_A_KIND]): 10,
        (Street.BACK, LookupTable.MAX_TO_RANK_CLASS[LookupTable.MAX_STRAIGHT_FLUSH]): 15
    }

    class Agent:
        def __init__(self) -> None:
            pass

        def __call__(self, state: 'Ofcp') -> 'Ofcp.Action':
            return random.choice(state.current_player().valid_actions())

        def copy(self) -> 'Ofcp.Agent':
            return self

    class Player:
        class Eval:
            def __init__(self, *, ranks: dict['Ofcp.Street', int], royalty: int, is_burst: bool) -> None:
                self.ranks = ranks.values()
                self.royalty = royalty
                self.is_burst = is_burst

            def __iter__(self):
                for rank in self.ranks:
                    yield rank
                yield self.royalty
                yield self.is_burst

        def __init__(self, *, hands: list[int], agent: 'Ofcp.Agent') -> None:
            self.streets: dict['Ofcp.Street', list[int]] = {street: [] for street in Ofcp.Street}
            self.hands: list[int] = sorted(hands)
            self.agent: 'Ofcp.Agent' = agent

        def __call__(self, *, state: 'Ofcp', card: int | None = None) -> 'Ofcp.Action':
            action = self.agent(state)
            self.streets[action.street].append(self.hands.pop(self.hands.index(action.card)))
            self.streets[action.street].sort()
            if card:
                self.hands.append(card)
                self.hands.sort()
            return action

        def copy(self) -> 'Ofcp.Player':
            copied = Ofcp.Player(hands=self.hands, agent=self.agent.copy())
            copied.streets = {street: hand.copy() for street, hand in self.streets.items()}
            return copied

        def set_agent(self, agent: 'Ofcp.Agent') -> None:
            self.agent = agent

        def valid_actions(self) -> tuple['Ofcp.Action', ...]:
            return tuple(Ofcp.Action(street, card) for card in self.hands for street in Ofcp.Street
                         if len(self.streets[street]) < Ofcp.NUM_SLOTS[street])

        def eval(self) -> 'Ofcp.Player.Eval':
            ranks, is_burst = self._ranks()
            return Ofcp.Player.Eval(ranks=ranks, royalty=self._royalty(ranks), is_burst=is_burst)

        def reset(self, *, hands: list[int]) -> None:
            self.streets = {street: [] for street in Ofcp.Street}
            self.hands = hands

        def _ranks(self) -> tuple[dict['Ofcp.Street', int], bool]:
            ranks = {street: Ofcp.EVALUATOR.evaluate(hand) for street, hand in self.streets.items()}
            is_burst = ranks[Ofcp.Street.FRONT] < ranks[Ofcp.Street.MID] or ranks[Ofcp.Street.MID] < ranks[Ofcp.Street.BACK]
            return {street: Ofcp.RANK_BURST for street in Ofcp.Street} if is_burst else ranks, is_burst

        def _royalty(self, ranks: dict['Ofcp.Street', int]) -> int:
            classes = {street: Ofcp.EVALUATOR.get_rank_class(rank) if rank < Ofcp.RANK_BURST else len(LookupTable.MAX_TO_RANK_CLASS)
                       for street, rank in ranks.items()}
            front = Ofcp.ROYALTY_LOOKUP.get((Ofcp.Street.FRONT, classes[Ofcp.Street.FRONT]), {}).get(ranks[Ofcp.Street.FRONT], 0)
            mid = Ofcp.ROYALTY_LOOKUP.get((Ofcp.Street.MID, classes[Ofcp.Street.MID]), 0) + (20 if ranks[Ofcp.Street.MID] == 1 else 0)
            end = Ofcp.ROYALTY_LOOKUP.get((Ofcp.Street.BACK, classes[Ofcp.Street.BACK]), 0) + (10 if ranks[Ofcp.Street.BACK] == 1 else 0)
            return front + mid + end

    class Eval:
        def __init__(self, *, street_point: float, scoop: float, royalty: float, is_burst: bool) -> None:
            self.street_point = street_point
            self.scoop = scoop
            self.royalty = royalty
            self.is_burst = is_burst

        def __iter__(self):
            yield self.street_point
            yield self.scoop
            yield self.royalty
            yield self.is_burst

        def __str__(self) -> str:
            return   f"\t\tstreet score : {self.street_point}\n"\
                   + f"\t\tscoop        : {self.scoop}\n"\
                   + f"\t\troyalty      : {self.royalty}\n"\
                   + f"\t\tburst        : {"Yes" if self.is_burst else "No"}"

    def __init__(self, *, num_players: int = 2) -> None:
        if num_players < 2 or num_players > 4:
            raise ValueError("Invalid number of players")

        self.deck: Deck = Deck()
        self.players: list[Ofcp.Player] = [Ofcp.Player(hands=self.deck.draw(Ofcp.NUM_INITIAL_CARDS), agent=Ofcp.Agent())
                                           for _ in range(num_players)]
        self.turn: int = 0

    def __iter__(self) -> 'Ofcp':
        return self

    def __next__(self) -> bool:
        if self.next():
            return True
        raise StopIteration

    def set_player_agent(self, *, player_id: int, agent: 'Ofcp.Agent') -> None:
        if player_id < 1 or player_id > len(self.players):
            raise ValueError("Invalid player ID")
        self.players[player_id - 1].set_agent(agent)

    def next(self) -> bool:
        drawn_card = self.deck.draw() if self.turn / len(self.players) < 8 else None
        self.players[self.turn % len(self.players)](state=self, card=drawn_card)  # type: ignore
        self.turn += 1
        return self.turn < len(self.players) * sum(Ofcp.NUM_SLOTS.values())

    def current_player(self) -> 'Ofcp.Player':
        return self.players[self.turn % len(self.players)]

    def eval(self) -> tuple['Ofcp.Eval', ...]:
        front_ranks, mid_ranks, back_ranks, royalties, bursts = zip(*(player.eval() for player in self.players))

        ranks = np.array(tuple(tuple(player_ranks) for player_ranks in zip(front_ranks, mid_ranks, back_ranks)))
        ranks = ranks == ranks.min(axis=0)
        num_best = np.count_nonzero(ranks, axis=0)
        winner_points = (np.full(len(Ofcp.Street), len(self.players)) - num_best) / num_best
        street_points = ranks @ winner_points + ~ranks @ np.full(len(Ofcp.Street), -1)

        max_street_point = (len(self.players) - 1) * len(Ofcp.Street)
        winner = street_points == max_street_point
        scoops = winner * max_street_point - ~winner * len(Ofcp.Street) if winner.sum() else np.zeros(len(self.players))

        royalties = np.array(royalties, dtype=np.float64)
        royalties -= royalties.mean()

        return tuple(Ofcp.Eval(street_point=street_point, scoop=scoop, royalty=royalty, is_burst=is_burst)
                     for street_point, scoop, royalty, is_burst in zip(street_points, scoops, royalties, bursts))

    def restart(self) -> None:
        self.deck.shuffle()
        for i in range(len(self.players)):
            self.players[i].reset(hands=self.deck.draw(Ofcp.NUM_INITIAL_CARDS))
        self.turn = 0


class OfcpUI(Ofcp):
    DIVIDER = {1: '=' * 80, 2: '-' * 70}

    class Verbosity(IntEnum):
        NONE     = 0
        SCORE    = 1
        FINAL    = 1 << 1
        INTERNAL = 1 << 2
        FULL     = (1 << 32) - 1

    def __init__(self, verbosity: int = Verbosity.FULL, *, num_players: int = 2) -> None:
        super().__init__(num_players=num_players)

        self.verbosity = verbosity

        self._print_start()

    def set_verbosity(self, verbosity: int) -> None:
        self.verbosity = verbosity

    def inherit(self, ofcp: Ofcp) -> None:
        self.turn = ofcp.turn
        self.deck.cards = ofcp.deck.cards.copy()
        self.players = [player.copy() for player in ofcp.players]

    def next(self) -> bool:
        drawn_card = self.deck.draw() if self.turn / len(self.players) < 8 else None
        action = self.players[self.turn % len(self.players)](state=self, card=drawn_card)  # type: ignore
        self.turn += 1
        not_end = self.turn < len(self.players) * sum(Ofcp.NUM_SLOTS.values())

        self._print_round(player_index=(self.turn - 1) % len(self.players) + 1, action=action)

        return not_end

    def eval(self) -> tuple[Ofcp.Eval, ...]:
        evals = super().eval()

        self._print_eval(evals)
        if self.verbosity:
            self._print_divider(1)

        return evals

    def restart(self) -> None:
        super().restart()

        self._print_start()

    def _print_start(self) -> None:
        if self.verbosity:
            self._print_divider(1)
        if (self.verbosity & OfcpUI.Verbosity.INTERNAL):
            self._print_round()

    def _print_round(self, *, player_index: int | None = None, action: Ofcp.Action | None = None) -> None:
        if (self.verbosity & OfcpUI.Verbosity.INTERNAL):
            self._print_state()
            self._print_action(player_index=(self.turn - 1) % len(self.players) + 1, action=action)
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
                f"{''.join(Card.int_to_pretty_str(card) for card in player.streets[street]):<45}"
                f"{"" if hide_class else self._hands_to_str(player.streets[street])}"
                for street in Ofcp.Street),
              sep='\n')

    def _hands_to_str(self, hands: list[int]) -> str:
        return Ofcp.EVALUATOR.class_to_string(Ofcp.EVALUATOR.get_rank_class(Ofcp.EVALUATOR.evaluate(hands)))

    def _print_action(self, *, player_index: int | None = None, action: Ofcp.Action | None = None) -> None:
        print(f"Player {player_index} put {Card.int_to_pretty_str(action.card)} at {action.street.value} street"
              if player_index and action else "No action yet")

    def _print_eval(self, evals: tuple[Ofcp.Eval, ...]) -> None:
        if (self.verbosity & OfcpUI.Verbosity.FINAL):
            for index, player in enumerate(self.players):
                self._print_player(player, index, hide_class=False)

        if (self.verbosity & OfcpUI.Verbosity.SCORE):
            print("Scores :")
            for [index, player], eval in zip(enumerate(self.players), evals):
                print(f"\tPlayer {index + 1} ( {player.agent.__class__.__name__} )")
                print(eval)


def test() -> None:
    import time

    begin = time.time()

    ofcp = OfcpUI()
    while ofcp.next():
        pass
    ofcp.eval()

    print(f"time : {time.time() - begin:.2f}s")


def main():
    test()


if __name__ == "__main__":
    main()
