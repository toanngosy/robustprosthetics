from baselines.deepq.replay_buffer import PrioritizedReplayBuffer


class KerasPrioritizedReplayBuffer():
    def __init__(self, limit, alpha):
        self.limit = limit
        self.alpha = alpha
        instance = PrioritizedReplayBuffer(size=self.limit, alpha=self.alpha)

    def append(self, observation, action, reward, terminal, training=True):

