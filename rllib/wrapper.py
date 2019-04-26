import gym
import numpy as np


# wrapper for observation
class NoObstacleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(NoObstacleObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(np.array([0.]*self.get_observation_space_size()),
                                                np.array([0.]*self.get_observation_space_size()))

    def observation(self, observation):
        return observation[:-5]

    def get_observation_space_size(self):
        return 38  # 43-5, 5 is reserved for obstacle position and sth else


class RelativeMassCenterObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(RelativeMassCenterObservationWrapper, self).__init__(env)

    def observation(self, observation):
        observation[1] -= observation[32]  # relative Pelvis Position Ground X axis
        observation[2] -= observation[33]  # relative Pelvis Position Ground Y axis

        observation[18] -= observation[32]  # relative Head Position X axis
        observation[19] -= observation[33]  # relative Head Position Y axis

        observation[20] -= observation[32]  # relative Pelvis Position X axis
        observation[21] -= observation[33]  # relative Pelvis Position Y axis

        observation[22] -= observation[32]  # relative Torso Position X axis
        observation[23] -= observation[33]  # relative Torso Position Y axis

        observation[24] -= observation[32]  # relative Toe_l Position X axis
        observation[25] -= observation[33]  # relative Toe_l Position Y axis

        observation[26] -= observation[32]  # relative Toe_r Position X axis
        observation[27] -= observation[33]  # relative Toe_r Position Y axis

        observation[28] -= observation[32]  # relative Talus_l Position X axis
        observation[29] -= observation[33]  # relative Talus_l Position Y axis

        observation[30] -= observation[32]  # relative Talus_r Position X axis
        observation[31] -= observation[33]  # relative Talus_r Position Y axis

        return observation

# wrapper for reward
class PriorityPelvisRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(PriorityPelvisRewardWrapper, self).__init__(env)

    def reward(self, reward):
        state_desc = self.env.get_state_desc()
        prev_state_desc = self.env.get_prev_state_desc()
        reward = state_desc['body_pos']['pelvis'][0]*10

        if prev_state_desc \
                and ((state_desc["joint_pos"]["ground_pelvis"][1] \
                     - prev_state_desc["joint_pos"]["ground_pelvis"][1]) < 0.005) \
                and (state_desc["body_pos"]["pelvis"][0]) > 0:

            reward -= state_desc["body_pos"]["pelvis"][0] * 20

        return reward

class MassCenterRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(MassCenterRewardWrapper, self).__init__(env)

    def reward(self, reward):
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        if not prev_state_desc:
            return 0
        return state_desc["misc"]["mass_center_pos"][1] - prev_state_desc["misc"]["mass_center_pos"][1]
