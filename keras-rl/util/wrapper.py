import gym
import numpy as np
from osim.env import OsimEnv
from osimwrapper import OsimWrapper

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

# wrapper for reward
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def reward(self, reward):
        foot_split_treshold = 0.9
        foot_split_penalty = 0
        foot_pelvis_penalty = 0
        mass_center_y_penalty = 0
        feet_y_penalty = 0
        legs_directions_penalty = 0
        velocity_feet_penalty = 0

        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()

        # DEFINITION OF USEFUL BODY PARTS
        head = state_desc["body_pos"]["head"]
        pelvis = state_desc["body_pos"]["pelvis"]
        base = [head[0], pelvis[1]]
        talus_r = state_desc["body_pos"]["talus_r"]
        talus_l = state_desc["body_pos"]["talus_l"]
        talus_r_vel = state_desc["body_vel"]["talus_r"]
        talus_l_vel = state_desc["body_vel"]["talus_l"]
        mass_center_pos = state_desc["misc"]["mass_center_pos"]
        mass_center_vel = state_desc["misc"]["mass_center_vel"]



        # LEGS : ensure that at least one leg is moving
        speed_l = np.sqrt(talus_l_vel[0]**2 + talus_l_vel[1]**2)
        speed_r = np.sqrt(talus_r_vel[0]**2 + talus_r_vel[1]**2)
        speed_feet_max = max(speed_l, speed_r)

        if speed_feet_max < 0.2:
            velocity_feet_penalty = -0.1
        else:
            velocity_feet_penalty = 0

        # LEGS : ensure that legs move in different directions :
        if (talus_l_vel[0] < -0.1 and talus_r_vel[0] > 0.1) or (talus_r_vel[0] < -0.1 and talus_l_vel[0] > 0.1):
            legs_directions_penalty = 0
        else:
            legs_directions_penalty = -0.1

        # COMPUTE ANGLE between head and pelvis
        v0 = np.array(base) - np.array(pelvis[0:2])
        v1 = np.array(head[0:2]) - np.array(pelvis[0:2])
        angle_pelvis_head = np.degrees(np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1)))

        if angle_pelvis_head < 0:
            angle_pelvis_head_penalty = -0.1
        elif angle_pelvis_head < 90 and angle_pelvis_head > 70:
            angle_pelvis_head_penalty = 0
        else:
            angle_pelvis_head_penalty = -0.1

        # FEET : Ensure that feet are not too split
        diff_foot = np.hypot(talus_r[0] - talus_l[0], talus_r[1] - talus_l[1])

        if diff_foot > foot_split_treshold:
            foot_split_penalty = -diff_foot

        # FEET : ensure that at least one foot is behind
        diff_foot_l_pelvis = talus_l[0] - pelvis[0]
        diff_foot_r_pelvis = talus_r[0] - pelvis[0]
        diff_foot_pelvis_min = min(diff_foot_l_pelvis, diff_foot_r_pelvis)

        if diff_foot_pelvis_min < 0 and velocity_feet_penalty == 0.1:
            foot_pelvis_penalty = 0
        else:
            foot_pelvis_penalty = -0.1

        # FEET : ensure feet are not too high
        feet_y_max = max(talus_l[1], talus_r[1])
        if feet_y_max > 0.4:
            feet_y_penalty = -0.1

        # MASS CENTER : ensure that mass center isn't too low
        if mass_center_pos[1] < 1.1 and mass_center_pos[1] > 0.85:
            mass_center_y_penalty = 0
        else:
            mass_center_y_penalty = -0.1

        # MASS CENTER : make sure you go forward
        if mass_center_vel[0] > 0.1:
            mass_center_vel_penalty = mass_center_vel[0]
        else:
            mass_center_vel_penalty = -0.1

        reward =  mass_center_pos[0] + legs_directions_penalty + \
            mass_center_vel_penalty + mass_center_y_penalty + \
            angle_pelvis_head_penalty + foot_split_penalty + \
            foot_pelvis_penalty + velocity_feet_penalty + feet_y_penalty

        if mass_center_pos[1] < 0.8 or head[0] < -0.25 or head[1] < 1.35 or diff_foot > 1.3:
            reward = -2

        # print("###############")
        # print("mass_center_pos : ", mass_center_pos)
        # print("angle_pelvis_head_penalty : ", angle_pelvis_head_penalty)
        # print("diff_foot_pelvis : ", foot_pelvis_penalty)
        # print("mass_center_y_penalty : ", mass_center_y_penalty)
        # print("mass_center_vel_penalty : ", mass_center_vel_penalty)
        # print( "velocity_feet_penalty : ", velocity_feet_penalty)
        # print( "legs_directions_penalty : ", legs_directions_penalty)
        # print("reward : ", reward)

        if not prev_state_desc:
            return 0
        return reward


# wrapper for Custom Done Osim
class CustomDoneOsimWrapper(OsimWrapper):
    def __init__(self, env):
        super(CustomDoneOsimWrapper, self).__init__(env)

    def is_done(self): 
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()

        # DEFINITION OF USEFUL BODY PARTS
        head = state_desc["body_pos"]["head"]
        pelvis = state_desc["body_pos"]["pelvis"]
        talus_r = state_desc["body_pos"]["talus_r"]
        talus_l = state_desc["body_pos"]["talus_l"]
        mass_center_pos = state_desc["misc"]["mass_center_pos"]
        diff_foot = np.hypot(talus_r[0] - talus_l[0], talus_r[1] - talus_l[1])

        return mass_center_pos[1] < 0.8 or head[0] < -0.3 or head[1] < 1.35 or diff_foot > 1.3

class AugmentedObservationWrapper(OsimWrapper):
    def __init__(self, env):
        super(CustomDoneOsimWrapper, self).__init__(env)

    def get_observation(self):
        state_desc = self.get_state_desc()

        # Augmented environment from the L2R challenge
        res = []
        pelvis = None
 
        res += state_desc["joint_pos"]["ground_pelvis"] 
        res += state_desc["joint_vel"]["ground_pelvis"]

        for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r",]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]

        for body_part in ["head", "pelvis", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            res += state_desc["body_pos"][body_part][0:2]

        res = res + state_desc["misc"]["mass_center_pos"] + state_desc["misc"]["mass_center_vel"]

        for body_part in ["head", "pelvis", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
            res += state_desc["body_vel"][body_part][0:2] #39-52
            if body_part == "pelvis":
                res += state_desc["body_acc"][body_part][0:2] #53-66
            else:
                res += [state_desc["body_acc"][body_part][i] - state_desc["body_acc"]["pelvis"][i] for i in range(1)]
                res += state_desc["body_acc"][body_part][1:2]
        res += [0]*5
        return res

    def get_observation_space_size(self):
        return 71
