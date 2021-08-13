"""Environment-related modules."""

from typing import Callable
import gym


class Environment:
    """
    This is a wrapper around an OpenAI Gym environment, so that
    we can store the features function along with the environment itself.
    
    Parameters:
        env (gym.Env): An OpenAi Gym environment.
        features (Callable):  Given a :class:`.Trajectory`, this function
            must return a :class:`numpy.array` of features.
    
    Attributes:
        env (gym.Env): The wrapped environment.
        features (Callable): Features function.
        action_space: Inherits from :py:attr:`env`.
        observation_space: Inherits from :py:attr:`env`.
        reset (Callable): Inherits from :py:attr:`env`.
        step (Callable): Inherits from :py:attr:`env`.
        render (Callable): Inherits from :py:attr:`env`, if it exists; None otherwise.
        render_exists (bool): True if :py:attr:`render` exists.
        close (Callable): Inherits from :py:attr:`env`, if it exists; None otherwise.
        close_exists (bool): True if :py:attr:`close` exists.
    """
    def __init__(self, env: gym.Env, feature_func: Callable):
        self.env = env
        self.features = feature_func

        # Mirror the main functionality
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset = self.env.reset
        self.step = self.env.step

        # Mirror the render function only if it exists
        self.render_exists = callable(getattr(self.env, "render", None))
        self.render = self.env.render if self.render_exists else None

        # Mirror the close function only if it exists
        self.close_exists = callable(getattr(self.env, "close", None))
        self.close = self.env.close if self.close_exists else None
