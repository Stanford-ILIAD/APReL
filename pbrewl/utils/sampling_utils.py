"""
This file contains functions that are useful for learning the weights on the reward function
(see the learning folder)
"""
from typing import Dict
import numpy as np

def uniform_logprior(params: Dict) -> float:
    """
    This is a prior over the belief function for omega, which is uniformly distributed over the unit sphere.
    Args:
        params: a dictionary which contains a possible omega

    Returns: the (unnormalized) log probability of omega, which is 0 (as 0 = log 1) if omega is in the unit sphere,
            and -infty otherwise

    """
    if np.linalg.norm(params['omega']) <= 1:
        return 0.
    return -np.inf


def gaussian_proposal(point: Dict) -> Dict:
    """For the Metropolis-Hastings sampling algorithm, this function generates the next step in the sequence
    assuming that the omega (or other sampled variables) follow a Gaussian distribution."""
    next_point = {}
    for key, value in point.items():
        if getattr(value, "shape", None) is not None:
            shape = list(value.shape)
        elif isinstance(value, list):
            shape = np.array(value).shape
        else:
            shape = [1]
        next_point[key] = value + np.random.randn(*shape) * 0.05
        if key == 'omega':
            next_point[key] /= np.linalg.norm(next_point[key])
    return next_point
