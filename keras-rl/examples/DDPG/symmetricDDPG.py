from rl.agents import DDPGAgent


class SymmetricDDPGAgent(DDPGAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO

    def process_state_right(self, state):
        state_right = state.copy()
        print(state_right[0])
        state_right[0][6], state_right[0][8] = state_right[0][8], state_right[0][6]
        state_right[0][7], state_right[0][9] = state_right[0][9], state_right[0][7]

        state_right[0][10], state_right[0][12] = state_right[0][12], state_right[0][10]
        state_right[0][11], state_right[0][13] = state_right[0][13], state_right[0][11]

        state_right[0][14], state_right[0][16] = state_right[0][16], state_right[0][14]
        state_right[0][15], state_right[0][17] = state_right[0][17], state_right[0][15]

        state_right[0][24], state_right[0][26] = state_right[0][26], state_right[0][24]
        state_right[0][25], state_right[0][27] = state_right[0][27], state_right[0][25]

        state_right[0][28], state_right[0][30] = state_right[0][30], state_right[0][28]
        state_right[0][29], state_right[0][31] = state_right[0][31], state_right[0][29]

        return state_right

    def select_action(self, state):

        state_left = state
        state_right = self.process_state_right(state)
        print("this asidfjisadfjisajfdijsdfis")
        print(state_left[0][6] - state_right[0][8])

        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    # TODO
    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action
