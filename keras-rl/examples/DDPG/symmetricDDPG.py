from rl.agents import DDPGAgent
import numpy as np
import copy



class SymmetricDDPGAgent(DDPGAgent):
    def __init__(self, noise_decay, random_process_l, random_process_r, **kwargs):
        DDPGAgent.__init__(self, **kwargs)
        self.noise_decay = noise_decay
        self.noise_coeff = noise_decay
        self.random_process_l = random_process_l
        self.random_process_r = random_process_r

    # TODO

    def process_state_right(self, state):
        state_right = copy.deepcopy(state)

        state_right[0][6], state_right[0][8] = state[0][8], state[0][6]
        state_right[0][7], state_right[0][9] = state[0][9], state[0][7]

        state_right[0][10], state_right[0][12] = state[0][12], state[0][10]
        state_right[0][11], state_right[0][13] = state[0][13], state[0][11]

        state_right[0][14], state_right[0][16] = state[0][16], state[0][14]
        state_right[0][15], state_right[0][17] = state[0][17], state[0][15]

        state_right[0][24], state_right[0][26] = state[0][26], state[0][24]
        state_right[0][25], state_right[0][27] = state[0][27], state[0][25]

        state_right[0][28], state_right[0][30] = state[0][30], state[0][28]
        state_right[0][29], state_right[0][31] = state[0][31], state[0][29]
        
        # print("################################")
        # print(state)
        # print(state_right)
        return state_right

    def select_action(self, state, left_leg):

        #TODO ok we have the two obs now

        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training:
            if left_leg:
                noise = self.random_process_l.sample()
            else:
                noise = self.random_process_r.sample()
            
            noise = noise * self.noise_coeff
            assert noise.shape == action.shape
            action += noise
            self.noise_coeff = self.noise_coeff * self.noise_decay
            action += noise

        return action

    # TODO
    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)

        observation_l = state
        observation_r = self.process_state_right(state)
        # print("#############################################################################")
        # print(observation_l)
        # print(observation_r)
        # print(np.asarray(observation_l) - np.asarray(observation_r))

        action_l = self.select_action(observation_l, left_leg=True)
        action_r = self.select_action(observation_r, left_leg=False)  

        # print(action_l)
        # print(action_r)

        # Book-keeping.
        self.recent_observation_l = observation_l[0]
        self.recent_action_l = action_l

        self.recent_observation_r = observation_r[0]
        self.recent_action_r = action_r

        # print("####")
        # print(observation_l)
        # print("####")
        # print(observation_r)


        action = np.concatenate((action_l, action_r), axis=None)
        # print("action ", action)
        return action


    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_observation_l = None
        self.recent_action_l = None
        self.recent_observation_r = None
        self.recent_action_r = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()


    def backward(self, reward, terminal=False):
            # Store most recent experience in memory.
            if self.step % self.memory_interval == 0:
                self.memory.append(self.recent_observation_l, self.recent_action_l, reward, terminal,
                                training=self.training)
                self.memory.append(self.recent_observation_r, self.recent_action_r, reward, terminal,
                        training=self.training)


            metrics = [np.nan for _ in self.metrics_names]
            if not self.training:
                # We're done here. No need to update the experience memory since we only use the working
                # memory to obtain the state over the most recent observations.
                return metrics

            # Train the network on a single stochastic batch.
            can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
            if can_train_either and self.step % self.train_interval == 0:
                experiences = self.memory.sample(self.batch_size)
                assert len(experiences) == self.batch_size

                # Start by extracting the necessary parameters (we use a vectorized implementation).
                state0_batch = []
                reward_batch = []
                action_batch = []
                terminal1_batch = []
                state1_batch = []
                for e in experiences:
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)

                # Prepare and validate parameters.
                state0_batch = self.process_state_batch(state0_batch)
                state1_batch = self.process_state_batch(state1_batch)
                terminal1_batch = np.array(terminal1_batch)
                reward_batch = np.array(reward_batch)
                action_batch = np.array(action_batch)
                assert reward_batch.shape == (self.batch_size,)
                assert terminal1_batch.shape == reward_batch.shape
                assert action_batch.shape == (self.batch_size, self.nb_actions)

                # Update critic, if warm up is over.
                if self.step > self.nb_steps_warmup_critic:
                    target_actions = self.target_actor.predict_on_batch(state1_batch)
                    assert target_actions.shape == (self.batch_size, self.nb_actions)
                    if len(self.critic.inputs) >= 3:
                        state1_batch_with_action = state1_batch[:]
                    else:
                        state1_batch_with_action = [state1_batch]
                    state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                    target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                    assert target_q_values.shape == (self.batch_size,)

                    # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                    # but only for the affected output units (as given by action_batch).
                    discounted_reward_batch = self.gamma * target_q_values
                    discounted_reward_batch *= terminal1_batch
                    assert discounted_reward_batch.shape == reward_batch.shape
                    targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                    # Perform a single batch update on the critic network.
                    if len(self.critic.inputs) >= 3:
                        state0_batch_with_action = state0_batch[:]
                    else:
                        state0_batch_with_action = [state0_batch]
                    state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                    metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                    if self.processor is not None:
                        metrics += self.processor.metrics

                # Update actor, if warm up is over.
                if self.step > self.nb_steps_warmup_actor:
                    # TODO: implement metrics for actor
                    if len(self.actor.inputs) >= 2:
                        inputs = state0_batch[:]
                    else:
                        inputs = [state0_batch]
                    if self.uses_learning_phase:
                        inputs += [self.training]
                    action_values = self.actor_train_fn(inputs)[0]
                    assert action_values.shape == (self.batch_size, self.nb_actions)

            if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
                self.update_target_models_hard()

            return metrics