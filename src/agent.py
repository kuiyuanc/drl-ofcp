from env import Env
from ofcp import OFCP


class Agent(OFCP.Agent):
    def __init__(self) -> None:
        super().__init__()

    def train(self):
        pass


class MCTS(Agent):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, state: OFCP) -> OFCP.Action:
        action: OFCP.Action = super().__call__(state)

        # TODO: implement MCTS inference

        return action


class DQN(Agent):
    def __init__(self) -> None:
        super().__init__()

    def train(self):
        # TODO: implement DQN training
        pass


class PPO(Agent):
    def __init__(self) -> None:
        super().__init__()

    def train(self):
        # TODO: implement PPO training
        pass
