##### IMPORTATION DES LIBRAIRIES #####
import sys
sys.path.append('../../util/')
sys.path.append('../../models')
sys.path.append('../../')

import numpy as np

from osim.env import L2RunEnv
from osim.http.client import Client
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import argparse
from datetime import datetime
import json
from robustensorboard import RobustTensorBoard
from check_files import check_xml, check_overwrite

# #### RECUPERATION DES PARAMETRES #####
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--step', dest='step', action='store', default=2000)
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="0")
args = parser.parse_args()


# #### Verification fichiers ######
# check_xml()

# #### INITIALISATION DES CONSTANTES #####
with open('parameters.json') as json_file:
    data = json.load(json_file)
    ## Model ##
    SIZE_HIDDEN_LAYER_ACTOR = data['SIZE_HIDDEN_LAYER_ACTOR']
    LR_ACTOR = data['LR_ACTOR']
    SIZE_HIDDEN_LAYER_CRITIC = data['SIZE_HIDDEN_LAYER_CRITIC']
    LR_CRITIC = data['LR_CRITIC']
    DISC_FACT = data['DISC_FACT']
    TARGET_MODEL_UPDATE = data['TARGET_MODEL_UPDATE']
    BATCH_SIZE = data['BATCH_SIZE']
    REPLAY_BUFFER_SIZE = data['REPLAY_BUFFER_SIZE']
    ## Exploration ##
    THETA = data['THETA']
    SIGMA = data['SIGMA']

# # Simulation ##
N_STEPS_TRAIN = int(args.step)
N_EPISODE_TEST = 100
if args.visualize:
    N_EPISODE_TEST = 3
VERBOSE = 1
# 0: pas de descriptif
# 1: descriptif toutes les LOG_INTERVAL steps
# 2: descriptif à chaque épisode
LOG_INTERVAL = 1000

# Save weights ##
FILES_WEIGHTS_NETWORKS = './weights/' + args.model + '.h5f'


# #### CHARGEMENT DE L'ENVIRONNEMENT #####
env = L2RunEnv(visualize=args.visualize, integrator_accuracy = 0.005)
# env.seed(1234)  # for comparison
env.reset()

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


# #### ACTOR / CRITIC #####
# Actor (mu) ##
input_shape = (1, env.observation_space.shape[0] + 2)
observation_input = Input(shape=input_shape, name='observation_input')

print("env.observation_space.shape")
print(env.observation_space.shape)

x = Flatten()(observation_input)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation='relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation='relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation='relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation = 'relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation = 'relu')(x)
x = Dense(action_size, activation='sigmoid')(x)
actor = Model(inputs=observation_input, outputs=x)

opti_actor = Adam(lr=LR_ACTOR)


# Critic (Q) ##
action_input = Input(shape=(action_size,), name='action_input')

x = Flatten()(observation_input)
x = concatenate([action_input, x])
x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation='relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation='relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation='relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation = 'relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation = 'relu')(x)
x = Dense(1, activation='linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

opti_critic = Adam(lr=LR_CRITIC)


# #### SET UP THE AGENT #####
# Initialize Replay Buffer ##
memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=1)
# window_length : usefull for Atari game (cb d'images d'affilé on veut analysé (vitesse de la balle, etc..))

# Random process (exploration) ##
random_process = OrnsteinUhlenbeckProcess(theta=THETA, mu=0, sigma=SIGMA,
                                          size=action_size)

# Paramètres agent DDPG ##
agent = DDPGAgent(nb_actions=action_size, actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory, random_process=random_process,
                  gamma=DISC_FACT, target_model_update=TARGET_MODEL_UPDATE,
                  batch_size=BATCH_SIZE)


agent.compile(optimizer=[opti_critic, opti_actor])

# #### TRAIN #####


logdir = "keras_logs/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
robustensorboard = RobustTensorBoard(log_dir=logdir, hyperparams=data)

if args.train:
    check_overwrite(args.model)
    agent.fit(env, nb_steps=N_STEPS_TRAIN, visualize=args.visualize,
              verbose=VERBOSE, log_interval=LOG_INTERVAL,
              callbacks=[robustensorboard])

    agent.save_weights(FILES_WEIGHTS_NETWORKS, overwrite=True)


#### TEST #####
if not args.train:
   agent.load_weights(FILES_WEIGHTS_NETWORKS)
   agent.test(env, nb_episodes=N_EPISODE_TEST, visualize=args.visualize)