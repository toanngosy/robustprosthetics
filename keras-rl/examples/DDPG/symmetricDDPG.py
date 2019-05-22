from rl.agents import DDPGAgent
import sys
sys.path.append('../../util')
from transform2symmetric import transform2symmetric

import numpy as np


class SymmetricDDPGAgent(DDPGAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO

    def process_state_right(self, state):
        state_right = state.copy()
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


    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            old_obs = self.recent_observation
            action = self.recent_action
            # new_obs = 
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

            obs_sym, action_sym = transform2symmetric(old_obs, action)
            self.memory.append(obs_sym, action_sym, reward, terminal, training = self.training)

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


    
