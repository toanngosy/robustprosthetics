import os

import numpy as np

from keras.callbacks import Callback


class SaveBestEpisode(Callback):
    def __init__(self):
        self.rewards = {}
        self.lastreward = -200

    def on_train_begin(self, logs):
        pass
        
    def on_train_end(self, logs):
        pass

    def on_episode_begin(self, episode, logs):

        self.rewards[episode] = []

        

    def on_episode_end(self, episode, logs):

        '''
        Code for saving up weights if the episode reward is higher than the last one
        '''
        
        if np.sum(self.rewards[episode])>self.lastreward:
            
            previousWeights = 'checkpoint_reward_{}.h5f'.format(self.lastreward)
            if os.path.exists(previousWeights): os.remove(previousWeights)
            self.lastreward = np.sum(self.rewards[episode])
            print("The reward is higher than the best one, saving checkpoint weights")
            newWeights = 'checkpoint_reward_{}.h5f'.format(np.sum(self.rewards[episode]))
            self.model.save_weights(newWeights, overwrite=True)
            
        del self.rewards[episode]

    def on_step_end(self, step, logs):
        episode = logs['episode']

        self.rewards[episode].append(logs['reward'])


