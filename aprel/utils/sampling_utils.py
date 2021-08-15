"""This module contains functions that are useful for the sampling in :class:`.SamplingBasedBelief`."""
from typing import Dict
import numpy as np

def uniform_logprior(params: Dict) -> float:
    """
    This is a log prior belief over the user. Specifically, it is a uniform distribution over ||weights|| <= 1.
    
    Args:
        params (Dict): parameters of the user for which the log prior is going to be calculated.

    Returns: 
        float: the (unnormalized) log probability of weights, which is 0 (as 0 = log 1) if ||weights|| <= 1, and negative infitiny otherwise.
    """
    if np.linalg.norm(params['weights']) <= 1:
        return 0.
    return -np.inf


def gaussian_proposal(point: Dict) -> Dict:
    """
    For the Metropolis-Hastings sampling algorithm, this function generates the next step in the Markov chain,
    with a Gaussian distribution of standard deviation 0.05.
    
    Args:
        point (Dict): the current point in the Markov chain.
    
    Returns:
        Dict: the next point in the Markov chain.
    """
    next_point = {}
    for key, value in point.items():
        if getattr(value, "shape", None) is not None:
            shape = list(value.shape)
        elif isinstance(value, list):
            shape = np.array(value).shape
        else:
            shape = [1]
        next_point[key] = value + np.random.randn(*shape) * 0.05
        if key == 'weights':
            next_point[key] /= np.linalg.norm(next_point[key])
    return next_point
