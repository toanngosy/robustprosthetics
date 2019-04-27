from gym import logger
import gym
import numpy as np
from osim.env import OsimEnv

# inspire by gym wrapper

warn_once = True
def deprecated_warn_once(text):
    global warn_once
    if not warn_once: return
    warn_once = False
    logger.warn(text)

class OsimWrapper(OsimEnv):
    env = None

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = getattr(self.env, 'spec', None)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def reward(self):
        if hasattr(self, "_reward"):
            deprecated_warn_once("%s doesn't implement 'reward' method, but it implements deprecated '_reward' method." % type(self))
            self.reward = self._reward
            return self._reward()
        else:
            deprecated_warn_once("%s doesn't implement 'reward' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.reward()

    def is_done(self):
        if hasattr(self, "_is_done"):
            deprecated_warn_once("%s doesn't implement 'is_done' method, but it implements deprecated '_is_done' method." % type(self))
            self.is_done = self._is_done
            return self._is_done()
        else:
            deprecated_warn_once("%s doesn't implement 'is_done' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.is_done()

    def load_model(self, **kwargs):
        if hasattr(self, "_load_model"):
            deprecated_warn_once("%s doesn't implement 'load_model' method, but it implements deprecated '_load_model' method." % type(self))
            self.load_model = self._load_model
            return self._load_model(**kwargs)
        else:
            deprecated_warn_once("%s doesn't implement 'load_model' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.load_model(**kwargs)

    def get_state_desc(self):
        if hasattr(self, "_get_state_desc"):
            deprecated_warn_once("%s doesn't implement 'get_state_desc' method, but it implements deprecated '_get_state_desc' method." % type(self))
            self.get_state_desc = self._get_state_desc
            return self._get_state_desc()
        else:
            deprecated_warn_once("%s doesn't implement 'get_state_desc' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.get_state_desc()

    def get_prev_state_desc(self):
        if hasattr(self, "_get_prev_state_desc"):
            deprecated_warn_once("%s doesn't implement 'get_prev_state_desc' method, but it implements deprecated '_get_prev_state_desc' method." % type(self))
            self.get_prev_state_desc = self._get_prev_state_desc
            return self._get_prev_state_desc()
        else:
            deprecated_warn_once("%s doesn't implement 'get_prev_state_desc' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.get_prev_state_desc()

    def get_observation(self):
        if hasattr(self, "_get_observation"):
            deprecated_warn_once("%s doesn't implement 'get_observation' method, but it implements deprecated '_get_observation' method." % type(self))
            self.get_observation = self._get_observation
            return self._get_observation()
        else:
            deprecated_warn_once("%s doesn't implement 'get_observation' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.get_observation()

    def get_observation_space_size(self):
        if hasattr(self, "_get_observation_space_size"):
            deprecated_warn_once("%s doesn't implement 'get_observation' method, but it implements deprecated '_get_observation_space_size' method." % type(self))
            self.get_observation_space_size = self._get_observation_space_size
            return self._get_observation_space_size()
        else:
            deprecated_warn_once("%s doesn't implement 'get_observation_space_size' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.get_observation_space_size()

    def get_action_space_size(self):
        if hasattr(self, "_get_action_space_size"):
            deprecated_warn_once("%s doesn't implement 'get_action_space_size' method, but it implements deprecated '_get_action_space_size' method." % type(self))
            self.get_action_space_size = self._get_action_space_size
            return self._get_action_space_size()
        else:
            deprecated_warn_once("%s doesn't implement 'get_action_space_size' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.get_action_space_size()

    def step(self, action, **kwargs):
        if hasattr(self, "_step"):
            deprecated_warn_once("%s doesn't implement 'step' method, but it implements deprecated '_step' method." % type(self))
            self.step = self._step
            return self._step(action, **kwargs)
        else:
            deprecated_warn_once("%s doesn't implement 'step' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        if hasattr(self, "_reset"):
            deprecated_warn_once("%s doesn't implement 'reset' method, but it implements deprecated '_reset' method." % type(self))
            self.reset = self._reset
            return self._reset(**kwargs)
        else:
            deprecated_warn_once("%s doesn't implement 'reset' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        if hasattr(self, "_render"):
            deprecated_warn_once("%s doesn't implement 'render' method, but it implements deprecated '_render' method." % type(self))
            self.render = self._render
            return self._render(**kwargs)
        else:
            deprecated_warn_once("%s doesn't implement 'render' method, " % type(self) +
                                 "which is required for wrappers derived directly from Wrapper. Deprecated default implementation is used.")
        return self.env.render(**kwargs)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped
