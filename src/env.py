from ofcp import OFCP


class Env:
    def __init__(self):
        self.ofcp = OFCP()

        # TODO: multiple-player extension

    def __call__(self, action: OFCP.Action) -> tuple[OFCP, float, bool]:
        reward, done = 0, True

        # TODO: take action
        # TODO: calculate reward
        # TODO: check if terminal state

        return self.state(), reward, done

    def state(self) -> OFCP:
        return self.ofcp

    def set_player_agent(self, *, player_id: int, agent: OFCP.Agent) -> None:
        self.ofcp.set_player_agent(player_id=player_id, agent=agent)

    def reset(self) -> None:
        self.ofcp.reset()
