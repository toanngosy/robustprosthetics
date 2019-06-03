##### IMPORTATION DES LIBRAIRIES #####
import sys
sys.path.append('../util/')
sys.path.append('../models')
sys.path.append('../')

import numpy as np
import os
from osim.env import L2RunEnv, ProstheticsEnv
from osim.http.client import Client
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Add, concatenate
from keras.optimizers import Adam, SGD, RMSprop
# from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import argparse
from datetime import datetime
import json
from robustensorboard import RobustTensorBoard
from check_files import check_xml, check_overwrite

from wrapper import *
from CustomDDPG import CustomDDPGAgent


def get_args():
    # #### RECUPERATION DES PARAMETRES #####
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    parser.add_argument('--step', dest='step', action='store', default=10000)
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='train', action='store_false', default=True)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--model', dest='model', action='store', default="0")
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--grid', dest='grid', action='store_true', default=False)
    args = parser.parse_args()
    return args

def get_param():
    #### RECUPERATION DES PARAMETRES DU JSON ####
    with open('parameters.json') as json_file:
        data = json.load(json_file)
    return data


def detect_multiparam(args, data):
    # Détection des paramètres avec plusieurs valeurs
    multi_params = [param for param in data if len(data[param]) > 1]
    if args.grid :
        if len(multi_params) == 0:
            args.grid = False
            print('Vous avez activé l\'argument --grid, pourtant aucun des paramètres ne possède plusieurs valeurs')
    # Vérification qu'aucun paramètre n'ait plusieurs valeurs
    else :
        if len(multi_params) > 0:
            choice = 'null'
            while(choice != 'y' and choice != 'n'):
                choice = input('Vous n\'avez pas activé l\'argument --grid, pourtant au moins un des paramètres possède plusieurs valeurs, voulez-vous l\'activer ? (y/n)  ')
                if(choice == 'y'):
                    args.grid = True
                elif(choice == 'n'):
                    sys.exit()
    
    return multi_params, args
            

def main_function(args, data):
    #### INITIALISATION DES CONSTANTES #####
    ## Model ##
    SIZE_HIDDEN_LAYER_ACTOR = data['SIZE_HIDDEN_LAYER_ACTOR'][0]
    LR_ACTOR = data['LR_ACTOR'][0]
    SIZE_HIDDEN_LAYER_CRITIC = data['SIZE_HIDDEN_LAYER_CRITIC'][0]
    LR_CRITIC = data['LR_CRITIC'][0]
    DISC_FACT = data['DISC_FACT'][0]
    TARGET_MODEL_UPDATE = data['TARGET_MODEL_UPDATE'][0]
    BATCH_SIZE = data['BATCH_SIZE'][0]
    REPLAY_BUFFER_SIZE = data['REPLAY_BUFFER_SIZE'][0]
    ## Exploration ##
    THETA = data['THETA'][0]
    SIGMA = data['SIGMA'][0]
    SIGMA_MIN = data['SIGMA_MIN'][0]
    N_STEPS_ANNEALING = data['N_STEPS_ANNEALING'][0]

    ## Acceleration ##
    ACTION_REPETITION = data['ACTION_REPETITION'][0]
    INTEGRATOR_ACCURACY = data['INTEGRATOR_ACCURACY'][0]

    # # Simulation ##
    N_STEPS_TRAIN = int(args.step)
    N_EPISODE_TEST = 100
    if args.visualize:
        N_EPISODE_TEST = 3
    VERBOSE = 1
    # 0: pas de descriptif
    # 1: descriptif toutes les LOG_INTERVAL steps
    # 2: descriptif à chaque épisode
    LOG_INTERVAL = 500

    # Save weights ##
    if not os.path.exists('weights'):
        os.mkdir('weights')
        print("Directory ", 'weights',  " Created ")
    FILES_WEIGHTS_NETWORKS = './weights/' + args.model + '.h5f'


    # #### CHARGEMENT DE L'ENVIRONNEMENT #####
    env = CustomDoneOsimWrapper(CustomRewardWrapper(RelativeMassCenterObservationWrapper(NoObstacleObservationWrapper(L2RunEnv(visualize=args.visualize, integrator_accuracy=0.005)))))

    env.reset()
    # Examine the action space ##
    action_size = env.action_space.shape[0]
    #action_size = int(env.action_space.shape[0]/2)    pour la symmétrie
    print('Size of each action:', action_size)

    # Examine the state space ##
    state_size = env.observation_space.shape[0]
    print('Size of state:', state_size)

    # #### ACTOR / CRITIC #####

    # Actor (mu) ##
    input_shape = (1, env.observation_space.shape[0])

    observation_input = Input(shape=input_shape, name='observation_input')

    x = Flatten()(observation_input)
    x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
    x = Activation('relu')(x)
    x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
    x = Activation('relu')(x)
    x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
    x = Activation('relu')(x)
    x = Dense(action_size)(x)
    x = Activation('sigmoid')(x)

    actor = Model(inputs=observation_input, outputs=x)
    opti_actor = Adam(lr=LR_ACTOR)


    # Critic (Q) ##
    action_input = Input(shape=(action_size,), name='action_input')

    x = Flatten()(observation_input)
    x = concatenate([action_input, x])
    x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
    x = Activation('relu')(x)
    x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
    x = Activation('relu')(x)
    x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)

    critic = Model(inputs=[action_input, observation_input], outputs=x)

    opti_critic = Adam(lr=LR_CRITIC)


    # #### SET UP THE AGENT #####
    # Initialize Replay Buffer ##
    memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=1)

    # Random process (exploration) ##
    random_process = OrnsteinUhlenbeckProcess(theta=THETA, mu=0, sigma=SIGMA,sigma_min= SIGMA_MIN,
                                            size=action_size, n_steps_annealing=N_STEPS_ANNEALING)


    agent = CustomDDPGAgent(nb_actions=action_size, actor=actor, critic=critic,
                            critic_action_input=action_input,
                            memory=memory, random_process=random_process,
                            gamma=DISC_FACT, target_model_update=TARGET_MODEL_UPDATE,
                            batch_size=BATCH_SIZE)

    agent.compile(optimizer=[opti_critic, opti_actor])


    # #### TRAIN #####
    logdir = "keras_logs/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    robustensorboard = RobustTensorBoard(log_dir=logdir, hyperparams=data)

    if args.train:
        if args.resume:
            agent.load_weights(FILES_WEIGHTS_NETWORKS)
        else:
            check_overwrite(args.model)

        agent.fit(env, nb_steps=N_STEPS_TRAIN, visualize=args.visualize,
                verbose=VERBOSE, log_interval=LOG_INTERVAL,
                callbacks=[robustensorboard], action_repetition = ACTION_REPETITION)

        agent.save_weights(FILES_WEIGHTS_NETWORKS, overwrite=True)


    #### TEST #####
    if not args.train:
        agent.load_weights(FILES_WEIGHTS_NETWORKS)
        agent.test(env, nb_episodes=N_EPISODE_TEST, visualize=args.visualize)


def get_last_model(model):
    compteur = 1
    weights = os.listdir('./weights/')
    for weight in weights:
        new_file = model + "-" + str(compteur) +'_actor.h5f'
        if(new_file == weight):
            compteur += 1
    return compteur


def main():
    args = get_args()
    params = get_param()
    multi_params, args = detect_multiparam(args, params)
    count = get_last_model(args.model)

    if args.grid :
        # Initialisation du compteur
        compteur = {}
        for param in params:
            compteur[param] = 0
        # Boucle principale pour tester toutes les valeurs
        for param in multi_params:
            while compteur[param] < len(params[param]):
                print("Compteur pour le paramètre", param, ": ",compteur[param]+1)
                print("Modele numero :", count)
                args.model = args.model + "-" + str(count)
                # Variable dynamique
                locals()[param] = params[param][compteur[param]]
                main_function(args, params)
                # On passe à la valeur suivante
                compteur[param] += 1
                count += 1


    else :
        args.model = args.model + "-" + str(count)
        main_function(args, params)



if __name__ == "__main__":
    main()