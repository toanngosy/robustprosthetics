import sys
sys.path.append('../../util/')
sys.path.append('../../')

import gym
import numpy as np
import argparse
import os
from osim.env import L2RunEnv, ProstheticsEnv
from osim.http.client import Client
from osim.env.utils.mygym import convert_to_gym
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from sac import SAC
from scale_outputs import scale_range, scale_discrete
#from robustensorboard import RobustTensorBoard
from check_files import check_overwrite
import callback


# #### RECUPERATION DES PARAMETRES #####
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--step', dest='step', action='store', default=100000)
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="default")
args = parser.parse_args()


# Save models ##
if not os.path.exists('models'):
    os.mkdir('models')
    print("Directory " , 'models' ,  " Created ")
MODELS_FOLDER_PATH = './models/' + args.model


# #### CHARGEMENT DE L'ENVIRONNEMENT #####
env = L2RunEnv(visualize=args.visualize, integrator_accuracy = 0.005)


# Examine the action space ##
action_size = env.action_space.shape[0]
print('Size of each action:', action_size)
action_low = env.action_space.low
print('Action low:', action_low)
action_high = env.action_space.high
print('Action high: ', action_high)

# Examine the state space ##
state_size = env.observation_space.shape[0]
print('Size of state:', state_size)

# Redefine action_space to -1/1 (sac implementation needs a symmetric action space) #
env.action_space = ( [-1.0] * env.get_action_space_size(), [1.0] * env.get_action_space_size())
env.action_space = convert_to_gym(env.action_space)


# Set observation space to a length of 43 (43 values are returned, but the initial environnement
# returns a observation space size of 41)
def new_get_observation_space_size():
    return 43

env.get_observation_space_size = new_get_observation_space_size

env.observation_space = ( [0] * env.get_observation_space_size(), [0] * env.get_observation_space_size() )
env.observation_space = convert_to_gym(env.observation_space)

# Create log dir for callback model saving
os.makedirs("./temp_models/", exist_ok=True)
env = Monitor(env, "./temp_models/", allow_early_resets=True)


##### TRAIN #####

if args.train:
    check_overwrite(args.model)
    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard_log/")
    model.learn(total_timesteps=int(args.step), log_interval=10, tb_log_name="log", callback=callback.callback)
    model.save(MODELS_FOLDER_PATH)

#### TEST #####

if not args.train:
    model = SAC.load(MODELS_FOLDER_PATH)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(scale_range(action,-1,1,0,1))
        env.render()
        if done:
            obs = env.reset()



