##### IMPORTATION DES LIBRAIRIES #####
import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.optimizers import RMSprop
from rl.agents import DQNAgent, DDPGAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess 
from keras.optimizers import RMSprop
import argparse
import gym
sys.path.append('../../util/')
sys.path.append('../../models')
sys.path.append('../../')
from save_train_test import save_plot_reward, save_result
from check_files import check_xml, check_overwrite
import matplotlib.pyplot as plt

from osim.env import L2RunEnv
from osim.http.client import Client


##### RECUPERATION DES PARAMETRES #####
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="default")
args = parser.parse_args()


##### Verification fichiers ######
check_xml()


##### INITIALISATION DES CONSTANTES #####
## Model ##
SIZE_HIDDEN_LAYER_ACTOR = 300
LR_ACTOR = 0.001
SIZE_HIDDEN_LAYER_CRITIC = 400
LR_CRITIC = 0.001
DISC_FACT = 0.99
TARGET_MODEL_UPDATE = 0.001
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000

## Exploration ##
THETA = 0.15
SIGMA = 0.2

params = [SIZE_HIDDEN_LAYER_ACTOR, LR_ACTOR, SIZE_HIDDEN_LAYER_CRITIC, LR_CRITIC, DISC_FACT, TARGET_MODEL_UPDATE, BATCH_SIZE, REPLAY_BUFFER_SIZE, THETA, SIGMA]

## Simulation ##
N_STEPS_TRAIN = 100000
N_EPISODE_TEST = 100
if args.visualize : N_EPISODE_TEST = 3
VERBOSE = 1
# 0: pas de descriptif
# 1: descriptif toutes les LOG_INTERVAL steps
# 2: descriptif à chaque épisode
LOG_INTERVAL = 10000

## Save weights ##
FILES_WEIGHTS_NETWORKS = './weights/' + args.model + '.h5f'


##### CHARGEMENT DE L'ENVIRONNEMENT #####
env = L2RunEnv(visualize = args.visualize)
# env.seed(1234)  # for comparison
env.reset()

## Examine the action space ##
action_size = env.action_space.shape[0]
print('Size of each action:', action_size)
action_low = env.action_space.low
print('Action low:', action_low)
action_high = env.action_space.high
print('Action high: ', action_high)

## Examine the state space ##
state_size = env.observation_space.shape[0]
print('Size of state:', state_size)


##### ACTOR / CRITIC #####
## Actor (mu) ##
input_shape = (1,env.observation_space.shape[0] + 2)  
observation_input = Input(shape =input_shape, name = 'observation_input')

print("env.observation_space.shape")
print(env.observation_space.shape)

x = Flatten()(observation_input)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation = 'relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation = 'relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation = 'relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation = 'relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_ACTOR, activation = 'relu')(x)
x = Dense(action_size, activation = 'sigmoid')(x)
actor = Model(inputs = observation_input, outputs=x)

opti_actor = Adam(lr = LR_ACTOR)


## Critic (Q) ##
action_input = Input(shape=(action_size,), name='action_input')

x = Flatten()(observation_input)
x = concatenate([action_input, x])
x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation = 'relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation = 'relu')(x)
x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation = 'relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation = 'relu')(x)
# x = Dense(SIZE_HIDDEN_LAYER_CRITIC, activation = 'relu')(x)
x = Dense(1, activation = 'linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

opti_critic = Adam(lr = LR_CRITIC)


##### SET UP THE AGENT #####
## Initialize Replay Buffer ##
memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=1)
# window_length : usefull for Atari game (cb d'images d'affilé on veut analysé (vitesse de la balle, etc..))

## Random process (exploration) ##
random_process = OrnsteinUhlenbeckProcess(theta=THETA, mu=0, sigma=SIGMA, size=action_size)

## Paramètres agent DDPG ##
agent = DDPGAgent(nb_actions=action_size, actor=actor, critic=critic, 
    critic_action_input=action_input,
    memory=memory, random_process=random_process, 
    gamma=DISC_FACT, target_model_update=TARGET_MODEL_UPDATE, 
    batch_size= BATCH_SIZE)

agent.compile(optimizer = [opti_critic, opti_actor], metrics= ['mae'])


##### TRAIN #####
if args.train:
    check_overwrite(args.model)
    history = agent.fit(env, nb_steps=N_STEPS_TRAIN, visualize=args.visualize, verbose=VERBOSE, log_interval = LOG_INTERVAL)
    agent.save_weights(FILES_WEIGHTS_NETWORKS, overwrite=True)
    save_plot_reward(history, args.model, params)


##### TEST #####
if not args.train :
    agent.load_weights(FILES_WEIGHTS_NETWORKS)
    history = agent.test(env, nb_episodes=N_EPISODE_TEST, visualize=args.visualize)
    save_result(history, args.model, params)
