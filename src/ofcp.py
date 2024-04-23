from lib.deuces import Card, Deck, Evaluator
import os
import random
import sys
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Row(Enum):
    FRONT = "front"
    MID = "mid"
    BACK = "back"


class Action:
    def __init__(self, row: Row, card: int) -> None:
        self.row = row
        self.card = card


class Ofcp:
    NUM_INITIAL_CARDS = 5

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

    def restart(self) -> None:
        self.deck = Deck()
        [self.players[i].reset(self.deck.draw(Ofcp.NUM_INITIAL_CARDS)) for i in range(len(self.players))]
        self.turn = 0


class Agent:
    def __init__(self) -> None:
        pass

    def __call__(self, state: Ofcp) -> Action:
        return random.choice(state.valid_actions())


class Board:
    EVALUATOR = Evaluator()
    NUM_SLOTS = {Row.FRONT: 3, Row.MID: 5, Row.BACK: 5}

    def __init__(self, hands: list[int]) -> None:
        self.hands = hands
        self.rows = {row: [] for row in Row}

    def get(self, card: int) -> None:
        self.hands.append(card)

    def put(self, row: Row, card: int) -> None:
        self.rows[row].append(self.hands.pop(self.hands.index(card)))

    def ranks(self) -> dict:
        return {row: Board.EVALUATOR.evaluate(hand) for row, hand in self.rows.items()}

    def reset(self, hands: list[int]) -> None:
        self.__init__(hands)

    def classes(self) -> dict:
        return {row: Board.EVALUATOR.get_rank_class(hand) for row, hand in self.rows.items()}

    def valid_actions(self) -> tuple[Action, ...]:
        return tuple(Action(row, card) for card in self.hands for row in Row if len(self.rows[row]) < Board.NUM_SLOTS[row])


class Player:
    def __init__(self, hands: list[int], *, agent: Agent = Agent()) -> None:
        self.board = Board(hands)
        self.agent = agent

    def __call__(self, action: Action, card: int) -> None:
        self.board.put(action.row, action.card)
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
        self._print_action(action)
        self._print_divider(2)

        if not not_end:
            # TODO: print result
            self._print_divider(1)

        return not_end

    def valid_actions(self) -> tuple[Action, ...]:
        return super().valid_actions()

    def restart(self) -> None:
        super().restart()

        self._print_divider(1)
        self._print_state()
        self._print_action()
        self._print_divider(2)

    def _print_player(self, index: int, player: Player) -> None:
        print(f"Player {index + 1} : {player.agent.__class__.__name__}")
        print(f"\t{"hands":<9} : {''.join(Card.int_to_pretty_str(card) for card in sorted(player.board.hands))}")
        print(*(f"\t{row.value + " row":<9} : {''.join(Card.int_to_pretty_str(card) for card in sorted(player.board.rows[row]))}"
                for row in Row),
              sep='\n')

    def _print_state(self) -> None:
        print(f"turn {self.turn:>2d}")
        for index, player in enumerate(self.players):
            self._print_player(index, player)

    def _print_action(self, action: Action | None = None) -> None:
        print(f"Player {(self.turn - 1) % len(self.players) + 1} put {Card.int_to_pretty_str(action.card)} at {action.row.value} row"
              if action else "No action yet")

    def _print_divider(self, level: int) -> None:
        print(self.DIVIDER[level])


ofcp = OfcpUI()
for _ in range(3):
    while ofcp.next():
        pass
    ofcp.restart()
