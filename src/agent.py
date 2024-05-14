import math
import random

import numpy as np
from numpy.typing import NDArray

from env import Env
from ofcp import OFCP


class Agent(OFCP.Agent):
    def __init__(self, *, player_id: int) -> None:
        super().__init__()

        self.player_id = player_id


class MC(Agent):
    def __init__(self, *, player_id: int, num_simulations: int) -> None:
        super().__init__(player_id=player_id)

        self.num_simulations = num_simulations

    def __call__(self, state: OFCP) -> OFCP.Action:
        valid = state.current_player().valid_actions()
        num_simulations = max(1, self.num_simulations // len(valid))
        return max(valid, key=lambda action: self._simulate(state=state, action=action, num_simulations=num_simulations))

    def _simulate(self, *, state: OFCP, action: OFCP.Action, num_simulations: int) -> float:
        sum_reward = 0
        for _ in range(num_simulations):
            env = Env(state.copy())
            env(action)
            while env:
                env()
            sum_reward += env.reward(player_id=self.player_id)
        return sum_reward

class MCTS(Agent):
    class Node:
        def __init__(self, state: OFCP, *, max_width: int, parent: 'MCTS.Node | None' = None, action: OFCP.Action | None = None) -> None:
            self.parent = parent
            self.children = []

            self.max_width = max_width
            self.action = action

            self.state = state

            self.num_visits = 0
            self.sum_reward = 0

        def select(self, *, explore_coef: float = math.sqrt(2)) -> 'MCTS.Node':
            node = self
            while node.state:
                if node._expandable():
                    return node
                node = node._best_child(explore_coef=explore_coef)
            return node

        def expand(self) -> 'MCTS.Node':
            if not self.state:
                return self

            untried_actions = set(self.state.current_player().valid_actions()) - {child.action for child in self.children}
            action = random.choice(tuple(untried_actions))

            copy = self.state.copy()
            copy(action)

            self.children.append(MCTS.Node(copy, max_width=self.max_width, parent=self, action=action))

            return self.children[-1]

        def simulate(self, player_id: int) -> float:
            env = Env(self.state)
            while env:
                env()
            return env.reward(player_id=player_id)

        def backpropagate(self, reward: float) -> None:
            node = self
            while node:
                node.num_visits, node.sum_reward = node.num_visits + 1, node.sum_reward + reward
                node = node.parent

        def best_action(self) -> OFCP.Action:
            return max(self.children, key=lambda child: child.num_visits).action

        def _expandable(self) -> bool:
            return len(self.children) < min(self.max_width, len(self.state.current_player().valid_actions()))

        def _best_child(self, *, explore_coef: float) -> 'MCTS.Node':
            best_score, best_child = -np.inf, None
            for child in self.children:
                exploit = child.sum_reward / child.num_visits if child.num_visits else np.inf
                explore = explore_coef * math.sqrt(math.log(self.num_visits) / child.num_visits)
                score = exploit + explore

                if score > best_score:
                    best_score, best_child = score, child
            return best_child if best_child else self

    def __init__(self, *, num_simulations: int, max_width: int, player_id: int) -> None:
        super().__init__(player_id=player_id)

        self.num_simulations = num_simulations
        self.max_width = max_width

    def __call__(self, state: OFCP) -> OFCP.Action:
        root = MCTS.Node(state, max_width=self.max_width)
        for _ in range(self.num_simulations):
            # select a node not fully expanded or a node with a terminal state, and expand it
            node = root.select().expand()
            # simulate until a terminal state, and backpropagate the reward
            node.backpropagate(node.simulate(self.player_id))
        return root.best_action()


class D3QN(OFCP.Agent):
    pass
