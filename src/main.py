import numpy as np
from numpy.typing import NDArray

from agent import MC, MCTS
from ofcp import OFCP, OFCPUI
from util import timing


def compete(num_competes: int = 100, num_players: int = 2, agents: dict[int, OFCP.Agent] = {}) -> NDArray[np.float64]:
    ofcp = OFCP(num_players=num_players)
    for player_id, agent in agents.items():
        ofcp.set_player_agent(player_id=player_id, agent=agent)

    scores = np.zeros(num_players)
    for i in range(num_competes):
        print(f"\r{i / num_competes * 100:5.2f}%", end='')
        while ofcp:
            ofcp()
        ofcp.eval()
        scores += np.array(tuple(float(eval) for eval in ofcp.eval()), dtype=np.float64)
        ofcp.reset()
    print("\r100.00%")

    return scores


def demo_random() -> None:
    scores = compete()
    print(scores)


def demo_mc() -> None:
    num_simulation, player_id = 1500, 1
    agents: dict[int, OFCP.Agent] = {1: MC(num_simulations=num_simulation, player_id=player_id)}
    scores = compete(agents=agents)
    print(scores)


def demo_mcts() -> None:
    num_simulation, max_width, player_id = 1500, 15, 1
    agents: dict[int, OFCP.Agent] = {1: MCTS(num_simulations=num_simulation, max_width=max_width, player_id=player_id)}
    scores = compete(agents=agents)
    print(scores)


def main():
    timing(demo_random)
    timing(demo_mc)
    timing(demo_mcts)


if __name__ == "__main__":
    main()
