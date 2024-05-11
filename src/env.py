import numpy as np
from numpy.typing import NDArray

from ofcp import OFCP


class Env:
    def __init__(self, *, num_players: int = 2) -> None:
        self.ofcp = OFCP(num_players=num_players)

    def __call__(self, action: OFCP.Action) -> OFCP:
        self.ofcp(action)
        return self.ofcp

    def state(self) -> OFCP:
        return self.ofcp

    def reward(self) -> NDArray[np.float64]:
        return np.zeros(len(self.ofcp.players)) if self.ofcp else np.array(tuple(sum(eval) for eval in self.ofcp.eval()))

    def set_player_agent(self, *, player_id: int, agent: OFCP.Agent) -> None:
        self.ofcp.set_player_agent(player_id=player_id, agent=agent)

    def reset(self) -> None:
        self.ofcp.reset()
