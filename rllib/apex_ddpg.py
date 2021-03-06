from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
#from osim.env import L2RunEnv
import gym
import numpy as np
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.utils import merge_dicts

from ray.rllib.agents.ddpg.apex import ApexDDPGTrainer, APEX_DDPG_DEFAULT_CONFIG


# wrapper for observation
#class NoObstacleObservationWrapper(gym.ObservationWrapper):
#    def __init__(self, env):
#        super(NoObstacleObservationWrapper, self).__init__(env)
#        self.observation_space = gym.spaces.Box(np.array([-314]*self.get_observation_space_size()),
#                                                np.array([314]*self.get_observation_space_size()))

#    def observation(self, observation):
#        return observation[:-5]

#    def get_observation_space_size(self):
#        return 38  # 43-5, 5 is reserved for obstacle position and sth else


# init ray
#ray.init(num_cpus=16, num_gpus=4)


# register env to ray
#def env_creator(env_config):
#    return NoObstacleObservationWrapper(L2RunEnv(**env_config))  # return an env instance


#register_env("L2RunEnv", env_creator)


# change config a bit
config = APEX_DDPG_DEFAULT_CONFIG.copy()
config['lr'] = 5e-04
config['twin_q'] = True
#config['model']['use_lstm'] = False
#config['model']['lstm_use_prev_action_reward'] = True
#config['vf_share_layers'] = True
config['num_gpus'] = 4
config['num_gpus_per_worker'] = 0
config['num_workers'] = 8
config['clip_actions'] = True
config['model']['fcnet_activation'] = 'relu'

del config['model']['squash_to_range']
del config['preprocessor_pref']
# set the config for env
env_config = {"env_config": {
        "visualize": False,
        "integrator_accuracy": 0.005}}

# the final config
config_env = merge_dicts(env_config, config)

print(config_env)

# create a trainer
trainer = PPOTrainer(env="L2RunEnv", config=config_env)


# train for 100000 iteration
for i in range(100000):
    result = trainer.train()
    print(pretty_print(result))



{
    'env_config': {
        'visualize': False,
        'integrator_accuracy': 0.005
        },
    'monitor': False, 'log_level': 'INFO',
    'callbacks': {
        'on_episode_start': None, 'on_episode_step': None,
        'on_episode_end': None, 'on_sample_end': None,
        'on_train_result': None, 'on_postprocess_traj': None
        }, 'ignore_worker_failures': False,
    'model': {
        'conv_filters': None,
        'conv_activation': 'relu', 'fcnet_activation': 'tanh',
        'fcnet_hiddens': [256, 256], 'free_log_std': False,
        'use_lstm': True, 'max_seq_len': 20, 'lstm_cell_size': 256,
        'lstm_use_prev_action_reward': True, 'framestack': True,
        'dim': 84, 'grayscale': False, 'zero_mean': True,
        'custom_preprocessor': None, 'custom_model': None, 'custom_options': {}
        },
    'optimizer': {
        'max_weight_sync_delay': 400, 'num_replay_buffer_shards': 4,
        'debug': False
        },
    'gamma': 0.99, 'horizon': None, 'soft_horizon': False, 'env': None,
    'clip_rewards': None, 'clip_actions': True,
    'preprocessor_pref': 'deepmind', 'evaluation_interval': None,
    'evaluation_num_episodes': 10, 'evaluation_config': {}, 'num_workers': 8,
    'num_gpus': 4, 'num_cpus_per_worker': 1, 'num_gpus_per_worker': 0,
    'custom_resources_per_worker': {}, 'num_cpus_for_driver': 1,
    'num_envs_per_worker': 1, 'sample_batch_size': 50, 'train_batch_size': 512,
    'batch_mode': 'truncate_episodes', 'sample_async': False,
    'observation_filter': 'NoFilter', 'synchronize_filters': True,
    'tf_session_args': {
        'intra_op_parallelism_threads': 2, 'inter_op_parallelism_threads': 2,
        'gpu_options': {'allow_growth': True}, 'log_device_placement': False,
        'device_count': {'CPU': 1}, 'allow_soft_placement': True
        },
    'local_evaluator_tf_session_args': {
        'intra_op_parallelism_threads': 8, 'inter_op_parallelism_threads': 8
        },
    'compress_observations': False, 'collect_metrics_timeout': 180,
    'metrics_smoothing_episodes': 100, 'remote_worker_envs': False,
    'remote_env_batch_wait_ms': 0, 'input': 'sampler', 'input_evaluation': ['is', 'wis'],
    'postprocess_inputs': False, 'shuffle_buffer_size': 0, 'output': None,
    'output_compress_columns': ['obs', 'new_obs'],
    'output_max_file_size': 67108864,
    'multiagent': {
        'policy_graphs': {}, 'policy_mapping_fn': None,
        'policies_to_train': None
        },
    'twin_q': False, 'policy_delay': 1, 'smooth_target_policy': False,
    'act_noise': 0.1, 'target_noise': 0.2, 'noise_clip': 0.5,
    'actor_hiddens': [64, 64], 'actor_hidden_activation': 'relu', 'critic_hiddens': [64, 64],
    'critic_hidden_activation': 'relu', 'n_step': 3,
    'schedule_max_timesteps': 100000, 'timesteps_per_iteration': 25000,
    'exploration_fraction': 0.1, 'exploration_final_eps': 0.02,
    'noise_scale': 0.1, 'exploration_theta': 0.15, 'exploration_sigma': 0.2,
    'target_network_update_freq': 500000, 'tau': 0.002, 'parameter_noise': False,
    'buffer_size': 2000000, 'prioritized_replay': True, 'prioritized_replay_alpha': 0.6,
    'prioritized_replay_beta': 0.4, 'prioritized_replay_eps': 1e-06, 'lr': 0.005,
    'actor_loss_coeff': 0.1, 'critic_loss_coeff': 1.0, 'use_huber': False,
    'huber_threshold': 1.0, 'l2_reg': 1e-06, 'grad_norm_clipping': None,
    'learning_starts': 50000, 'optimizer_class': 'AsyncReplayOptimizer',
    'per_worker_exploration': True, 'worker_side_prioritization': True,
    'min_iter_time_s': 30, 'vf_share_layers': True}

