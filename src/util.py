import time
import random


class EpsilonGreedy:
    def __init__(self, epsilon: float = 1, decay: float = 0.995, min_epsilon: float = 0.05) -> None:
        self._epsilon = epsilon
        self._decay = decay
        self._min = min_epsilon

    def __call__(self, *, explore, exploit):
        return explore if random.random() < self._epsilon else exploit

    def decay(self) -> None:
        self._epsilon = max(self._min, self._epsilon * self._decay)


def timing(f, *args, **kwargs):
    start = time.time()
    res = f(*args, **kwargs)
    print(f"{f.__name__} : {time.time() - start:.2f}s")
    return res
