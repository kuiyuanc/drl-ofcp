import os
import random
import sys
from enum import Enum


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from lib.deuces.evaluator import LookupTable
from lib.deuces import Card, Deck, Evaluator


class Street(Enum):
    FRONT = "front"
    MID = "mid"
    BACK = "back"


class Action:
    def __init__(self, street: Street, card: int) -> None:
        self.street = street
        self.card = card


class Ofcp:
    NUM_INITIAL_CARDS = 5
    MAX_RANK = LookupTable.MAX_THREE_CARD_HIGH_CARD + 1

    def __init__(self, *, num_players: int = 2) -> None:
        if num_players != 2:
            raise NotImplementedError("2 players only")

        self.deck = Deck()
        self.players = [Player(self.deck.draw(Ofcp.NUM_INITIAL_CARDS)) for _ in range(num_players)]
        self.turn = 0

    def next(self) -> bool:
        action = self.players[self.turn % len(self.players)].agent(self)
        self.players[self.turn % len(self.players)](action, self.deck.draw())  # type: ignore
        self.turn += 1
        return self.turn < len(self.players) * sum(Board.NUM_SLOTS.values())

    def valid_actions(self) -> tuple[Action, ...]:
        return self.players[self.turn % len(self.players)].board.valid_actions()

    def eval(self) -> tuple[tuple[int, int, int, bool], ...]:
        players_ranks, players_classes = zip(*(player.board.ranks_classes() for player in self.players))

        bursts = [self._is_burst(player_rank) for player_rank in players_ranks]
        players_ranks = [{street: Ofcp.MAX_RANK for street in Street} if burst else player_ranks
                         for player_ranks, burst in zip(players_ranks, bursts)]

        # TODO: deal with ties
        if False in bursts:
            best_ranks = {street: min(player_ranks[street] for player_ranks in players_ranks) for street in Street}
            street_points = tuple(sum(len(self.players) - 1 if player_ranks[street] == best_ranks[street] else -1
                                      for street in Street)
                                  for player_ranks in players_ranks)
        else:
            street_points = tuple(0 for _ in players_ranks)

        max_street_point = max(street_points)
        scoops = tuple((len(Street) if max_street_point == len(Street) else 0)
                       * (len(self.players) - 1 if street_point == max_street_point else -1)
                       for street_point in street_points)

        # TODO: calculate royalties
        royalties = [0 for _ in self.players]

        return tuple(zip(street_points, scoops, royalties, bursts))

    def restart(self) -> None:
        self.deck = Deck()
        [self.players[i].reset(self.deck.draw(Ofcp.NUM_INITIAL_CARDS)) for i in range(len(self.players))]
        self.turn = 0

    def _is_burst(self, ranks: dict[Street, int]) -> bool:
        return ranks[Street.FRONT] < ranks[Street.MID] or ranks[Street.MID] < ranks[Street.BACK]


class Agent:
    def __init__(self) -> None:
        pass

    def __call__(self, state: Ofcp) -> Action:
        return random.choice(state.valid_actions())


class Board:
    EVALUATOR = Evaluator()
    NUM_SLOTS = {Street.FRONT: 3, Street.MID: 5, Street.BACK: 5}

    def __init__(self, hands: list[int]) -> None:
        self.hands = hands
        self.streets = {street: [] for street in Street}

    def get(self, card: int) -> None:
        self.hands.append(card)

    def put(self, street: Street, card: int) -> None:
        self.streets[street].append(self.hands.pop(self.hands.index(card)))

    def _ranks(self) -> dict[Street, int]:
        return {street: Board.EVALUATOR.evaluate(hand) for street, hand in self.streets.items()}

    def reset(self, hands: list[int]) -> None:
        self.__init__(hands)

    def _classes(self, ranks: dict[Street, int]) -> dict[Street, int]:
        return {street: Board.EVALUATOR.get_rank_class(rank) for street, rank in ranks.items()}

    def ranks_classes(self) -> tuple[dict[Street, int], dict[Street, int]]:
        ranks = self._ranks()
        return ranks, self._classes(ranks)

    def valid_actions(self) -> tuple[Action, ...]:
        return tuple(Action(street, card) for card in self.hands for street in Street
                     if len(self.streets[street]) < Board.NUM_SLOTS[street])


class Player:
    def __init__(self, hands: list[int], *, agent: Agent = Agent()) -> None:
        self.board = Board(hands)
        self.agent = agent

    def __call__(self, action: Action, card: int) -> None:
        self.board.put(action.street, action.card)
        self.board.get(card)

    def reset(self, hands: list[int]) -> None:
        self.board.reset(hands)


class OfcpUI(Ofcp):
    DIVIDER = {1: '=' * 30, 2: '-' * 20}

    def __init__(self, *, num_players: int = 2) -> None:
        super().__init__(num_players=num_players)

        self._print_divider(1)
        self._print_state()
        self._print_action()
        self._print_divider(2)

    def next(self) -> bool:
        action = self.players[self.turn % len(self.players)].agent(self)
        self.players[self.turn % len(self.players)](action, self.deck.draw())  # type: ignore
        self.turn += 1
        not_end = self.turn < len(self.players) * sum(Board.NUM_SLOTS.values())

        self._print_state()
        self._print_action(index=(self.turn - 1) % len(self.players) + 1, action=action)
        self._print_divider(2)

        return not_end

    def eval(self) -> tuple[tuple[int, int, int, bool], ...]:
        scores = super().eval()

        self._print_eval(scores)
        self._print_divider(1)

        return scores

    def restart(self) -> None:
        super().restart()

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

    def _print_player(self, player: Player, index: int, *, hide_class: bool = True) -> None:
        print(f"Player {index + 1} ( {player.agent.__class__.__name__} )")
        print(f"\t{"hands":<13} : {''.join(Card.int_to_pretty_str(card) for card in sorted(player.board.hands))}")
        print(*(f"\t{street.value + " street":<13} : "
                f"{''.join(Card.int_to_pretty_str(card) for card in sorted(player.board.streets[street])):<45}"
                f"{"" if hide_class else self._hands_to_str(player.board.streets[street])}"
                for street in Street),
              sep='\n')

    def _hands_to_str(self, hands: list[int]) -> str:
        return Board.EVALUATOR.class_to_string(Board.EVALUATOR.get_rank_class(Board.EVALUATOR.evaluate(hands)))

    def _print_action(self, *, index: int | None = None, action: Action | None = None) -> None:
        print(f"Player {index} put {Card.int_to_pretty_str(action.card)} at {action.street.value} street"
              if index and action else "No action yet")

    def _print_eval(self, scores: tuple[tuple[int, int, int, bool], ...]) -> None:
        for index, player in enumerate(self.players):
            self._print_player(player, index, hide_class=False)

        # TODO: print scores
        print("Scores :")
        for index, player in enumerate(self.players):
            print(f"\tPlayer {index + 1} ( {player.agent.__class__.__name__} )")
            print(f"\t\tstreet score : {scores[index][0]}")
            print(f"\t\tscoop        : {scores[index][1]}")
            print(f"\t\troyalty      : {scores[index][2]}")
            print(f"\t\tburst        : {scores[index][3]}")


ofcp = OfcpUI()
while ofcp.next():
    pass
ofcp.eval()
