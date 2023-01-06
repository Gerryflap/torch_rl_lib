import random


class SimpleEnv:
    """
        Simple environment for testing purposes.
        Starts the agent at a random position between -10 and 10,
            then gives the agent 20 steps to get sufficiently close to 0.0 (within 0.1 distance).
        Control is continuous, between bounds [-1, 1].
        Reward is only given at the end of the episode,
            and is given as a negative reward proportional to the distance to the bounds around 0.
        Termination happens either after 20 steps, or when the agent is within the [-0.1, 0.1] bounds .
    """
    def __init__(self):
        self.pos = 0.0
        self.remaining = 0.0
        self.reset()

    def reset(self):
        self.pos = random.random() * 20.0 - 10.0
        self.remaining = 20
        return self.pos, None

    def step(self, action):
        if self.remaining <= 0:
            raise ValueError("Called step after termination!")
        self.pos += min(max(action, -1.0), 1.0)
        term = -0.1 <= self.pos <= 0.1 or self.remaining == 1
        trunc = False
        r = 0 if not (trunc or term) else -max(abs(self.pos) - 0.1, 0.0)
        self.remaining -= 1
        return self.pos, r, term, trunc, None

    def close(self):
        pass

    def render(self):
        pass
