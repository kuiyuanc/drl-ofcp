from collections import deque
import random

from env import Env
from ofcp import OFCP
from util import EpsilonGreedy


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
    class Record:
        def __init__(self, state: OFCP, action: OFCP.Action, reward: float, next_state: OFCP) -> None:
            self.state = state
            self.action = action
            self.reward = reward
            self.next_state = next_state

        def __iter__(self):
            yield self.state
            yield self.action
            yield self.reward
            yield self.next_state

    def __init__(self, *, state_size: int, action_size: int) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque()
        self.gamma = 0.95  # discount rate
        self.epsilon_greedy = EpsilonGreedy()
        self.learning_rate = 0.001
        self.model = self._build_model()

    def __call__(self, state: OFCP) -> OFCP.Action:
        # TODO: implement DQN inference
        pass

    def train(self, num_episode: int, *, player_id: int) -> None:
        env = Env()
        env.set_player_agent(player_id=player_id, agent=self)

        for _ in range(num_episode):
            total_reward, state = 0, env.state()

            while state:
                action = self(state)
                next_state = env(action)
                reward = env.reward()[player_id - 1]

                total_reward += reward
                self.memory.append(DQN.Record(state, action, reward, next_state))
                state = next_state

            env.reset()

    def _build_model(self):
        # TODO: build a DQN model
        pass

    def _replay(self, batch_size: int) -> None:
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in batch:
            '''
            target = reward + (self.gamma * np.amax(self.model.predict(next_state)[0]) if state else 0)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            '''
            # TODO: implement DQN replay

        self.epsilon_greedy.decay()
