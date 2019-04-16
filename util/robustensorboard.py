import timeit
# TensorBoard to monitor
from keras.callbacks import TensorBoard
import numpy as np


class RobustTensorBoard(TensorBoard):
    """
    Subclassing of tensorboard to log and visualize custom metrics and others
    """
    def __init__(self, *args, **kwargs):

        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}

        super(RobustTensorBoard, self).__init__(*args, **kwargs)

    def _set_env(self, env):
        self.env = env

    def on_step_begin(self, step, logs):
        pass

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])

    def on_episode_begin(self, episode, logs):
        self.episode_start_time = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs={}):
        new_logs = {}
        training_speed = (timeit.default_timer() - self.episode_start_time)/logs["nb_steps"]
        new_logs.update({"total reward p. episode": logs["episode_reward"]})
        new_logs.update({"mean reward p. episode": np.mean(self.rewards[episode])})
        new_logs.update({"training speed p. episode": training_speed})
        new_logs.update({"number steps p. episode": logs["nb_steps"]})

        super(RobustTensorBoard, self).on_epoch_end(episode, new_logs)

        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass
