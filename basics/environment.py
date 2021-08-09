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
        if callable(render_func):
            self.render = self.env.render
