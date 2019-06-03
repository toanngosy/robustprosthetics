def main_function(args):
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
    if args.prosthetic:
        env = CustomRewardWrapper(ProstheticsEnv(visualize=args.visualize, integrator_accuracy=INTEGRATOR_ACCURACY))
    if not args.prosthetic:
        env = CustomDoneOsimWrapper(CustomRewardWrapper(RelativeMassCenterObservationWrapper(NoObstacleObservationWrapper(L2RunEnv(visualize=args.visualize, integrator_accuracy=0.005)))))

    env.reset()
    # Examine the action space ##
    action_size = int(env.action_space.shape[0]/2)
    print('Size of each action:', action_size)

    # Examine the state space ##
    state_size = env.observation_space.shape[0]
    print('Size of state:', state_size)


    # #### ACTOR / CRITIC #####
    # Actor (mu) ##
    if args.prosthetic:
        input_shape = (1, env.observation_space.shape[0])
    if not args.prosthetic:
        input_shape = (1, env.observation_space.shape[0])

    observation_input = Input(shape=input_shape, name='observation_input')

    x = Flatten()(observation_input)

    x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(SIZE_HIDDEN_LAYER_ACTOR)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Dense(action_size)(x)     #without symmetry
    x = Dense(action_size)(x)

    x = Activation('sigmoid')(x)

    actor = Model(inputs=observation_input, outputs=x)

    opti_actor = Adam(lr=LR_ACTOR)


    # Critic (Q) ##
    action_input = Input(shape=(action_size,), name='action_input')

    x = Flatten()(observation_input)
    x = concatenate([action_input, x])

    x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(SIZE_HIDDEN_LAYER_CRITIC)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(1)(x)
    x = Activation('linear')(x)

    critic = Model(inputs=[action_input, observation_input], outputs=x)

    opti_critic = Adam(lr=LR_CRITIC)


    # #### SET UP THE AGENT #####
    # Initialize Replay Buffer ##
    memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=1)

    # Random process (exploration) ##
    # random_process = OrnsteinUhlenbeckProcess(theta=THETA, mu=0, sigma=SIGMA,
    #                                           size=action_size)
    random_process_l = OrnsteinUhlenbeckProcess(theta=THETA, mu=0, sigma=SIGMA,
                                            size=action_size)
    random_process_r = OrnsteinUhlenbeckProcess(theta=THETA, mu=0, sigma=SIGMA,
                                            size=action_size)

    # Paramètres agent DDPG ##
    agent = SymmetricDDPGAgent(nb_actions=action_size, actor=actor, critic=critic,
                            critic_action_input=action_input,
                            memory=memory, random_process_l=random_process_l, random_process_r=random_process_r,
                            gamma=DISC_FACT, target_model_update=TARGET_MODEL_UPDATE,
                            batch_size=BATCH_SIZE, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000, noise_decay=NOISE_DECAY)

    agent.compile(optimizer=[opti_critic, opti_actor])


    # #### TRAIN #####
    logdir = "keras_logs/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    robustensorboard = RobustTensorBoard(log_dir=logdir, hyperparams=data)

    if args.train:
        if args.resume:
            agent.load_weights(FILES_WEIGHTS_NETWORKS)
        else:
            check_overwrite(args.model)

        try:
            agent.fit(env, nb_steps=N_STEPS_TRAIN, visualize=args.visualize,
                    verbose=VERBOSE, log_interval=LOG_INTERVAL,
                    callbacks=[robustensorboard], action_repetition = ACTION_REPETITION)

            agent.save_weights(FILES_WEIGHTS_NETWORKS, overwrite=True)

        except KeyboardInterrupt:
            print("interruption detected , saving weights....")
            agent.save_weights(FILES_WEIGHTS_NETWORKS, overwrite=True)


    #### TEST #####
    if not args.train:
        agent.load_weights(FILES_WEIGHTS_NETWORKS)
        agent.test(env, nb_episodes=N_EPISODE_TEST, visualize=args.visualize)
