def transform2symmetric(obs, action):
    #Symetrize obs and new_obs
    index_sym_obs = [6, 7, 10, 11, 14, 15, 24, 25, 28, 29]
    for tab in [obs]:
        for index in index_sym_obs:
            tab[index], tab[index+2] = tab[index+2], tab[index]

    # Symetrize action
    index_sym_action = [i for i in range(9)]
    for index in index_sym_action:
        action[index], action[index+9] = action[index+9], action[index]

    return obs, action