from typing import Callable
import gym


class Environment:
    def __init__(self, env: gym.Env, feature_func: Callable):
        self.env = env
        self.features = feature_func
        
        # Mirror the main functionality
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset = self.env.reset
        self.step = self.env.step
        
        # Mirror the render function only if it exists
        render_func = getattr(self.env, "render", None)
        self.render = self.env.render if callable(render_func) else None

        # Mirror the set_state function only if it exists
        set_state_func = getattr(self.env, "set_state", None)
        self.set_state = self.env.set_state if callable(set_state_func) else None